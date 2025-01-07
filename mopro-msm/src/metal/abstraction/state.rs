use metal::{ComputeCommandEncoderRef, Device, MTLResourceOptions};

use crate::metal::abstraction::errors::MetalError;

use core::{ffi, mem};
use std::env;
use crate::metal::abstraction::limbs_conversion::ToLimbs;

/// Structure for abstracting basic calls to a Metal device and saving the state. Used for
/// implementing GPU parallel computations in Apple machines.
pub struct MetalState {
    pub device: metal::Device,
    pub library: metal::Library,
    pub queue: metal::CommandQueue,
}

impl MetalState {
    /// Creates a new Metal state with an optional `device` (GPU).
    /// If `None` is passed, it will use the **first available device**, not the system's default.
    pub fn new(device: Option<Device>) -> Result<Self, MetalError> {

        // Step 1: Get the device
        // We use first available device instead of system default because it saves 30ms initialization time
        let device: Device = device
            .unwrap_or(
                Device::all().first()
                    .map_or_else(
                        || Err(MetalError::DeviceNotFound()),
                        |d| Ok(d.clone())
                    )?
            );



        // Step 2: Load the Metal library data
        let lib_data = include_bytes!(concat!(env!("OUT_DIR"), "/msm.metallib"));


        // Step 3: Create the Metal library from the loaded data
        let library = device
            .new_library_with_data(lib_data)
            .map_err(MetalError::LibraryError)?;

        // Step 4: Create the command queue
        let queue = device.new_command_queue();

        // Return the Metal state
        Ok(Self {
            device,
            library,
            queue,
        })
    }

    /// Creates a pipeline based on a compute function `kernel` which needs to exist in the state's
    /// library. A pipeline is used for issuing commands to the GPU through command buffers,
    /// executing the `kernel` function.
    pub fn setup_pipeline(
        &self,
        kernel_name: &str,
    ) -> Result<metal::ComputePipelineState, MetalError> {
        let kernel = self
            .library
            .get_function(kernel_name, None)
            .map_err(MetalError::FunctionError)?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(MetalError::PipelineError)?;

        Ok(pipeline)
    }

    /// Allocates `length` bytes of shared memory between CPU and the device (GPU).
    pub fn alloc_buffer<T>(&self, length: usize) -> metal::Buffer {
        let size = mem::size_of::<T>();

        self.device.new_buffer(
            (length * size) as u64,
            MTLResourceOptions::StorageModeShared, // TODO: use managed mode
        )
    }

    /// Allocates `data` in a buffer of shared memory between CPU and the device (GPU).
    pub fn alloc_buffer_data<T>(&self, data: &[T]) -> metal::Buffer {
        let size = mem::size_of::<T>();

        self.device.new_buffer_with_data(
            data.as_ptr() as *const ffi::c_void,
            (data.len() * size) as u64,
            MTLResourceOptions::StorageModeShared, // TODO: use managed mode
        )
    }

    pub fn alloc_buffer_data_direct<T, const N: usize>(&self, items: &[T]) -> metal::Buffer
    where
        T: ToLimbs<N> + Sync,
    {
        let num_items = items.len();
        let limb_size = mem::size_of::<u32>() * N;
        let total_size = num_items * limb_size;

        // Create a GPU buffer with shared storage
        let buffer = self.device.new_buffer(
            total_size as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Safety: Ensure the buffer's contents are safe to access as mutable
        let buffer_ptr = buffer.contents() as *mut u32;

        unsafe {
            items.write_u32_limbs(std::slice::from_raw_parts_mut(buffer_ptr, N * num_items));
        }

        buffer
    }

    pub fn set_bytes<T>(index: usize, data: &[T], encoder: &ComputeCommandEncoderRef) {
        let size = mem::size_of::<T>();

        encoder.set_bytes(
            index as u64,
            (data.len() * size) as u64,
            data.as_ptr() as *const ffi::c_void,
        );
    }

    /// Creates a command buffer and a compute encoder in a pipeline, optionally issuing `buffers`
    /// to it.
    pub fn setup_command(
        &self,
        pipeline: &metal::ComputePipelineState,
        buffers: Option<&[(u64, &metal::Buffer)]>,
    ) -> (&metal::CommandBufferRef, &metal::ComputeCommandEncoderRef) {
        let command_buffer = self.queue.new_command_buffer();
        let command_encoder = command_buffer.new_compute_command_encoder();
        command_encoder.set_compute_pipeline_state(pipeline);

        if let Some(buffers) = buffers {
            for (i, buffer) in buffers.iter() {
                command_encoder.set_buffer(*i, Some(buffer), 0);
            }
        }

        (command_buffer, command_encoder)
    }

    /// Returns a vector of a copy of the data that `buffer` holds, interpreting it into a specific
    /// type `T`.
    ///
    /// BEWARE: this function uses an unsafe function for retrieveing the data, if the buffer's
    /// contents don't match the specified `T`, expect undefined behaviour. Always make sure the
    /// buffer you are retreiving from holds data of type `T`.
    pub fn retrieve_contents<T: Clone>(buffer: &metal::Buffer) -> Vec<T> {
        let ptr = buffer.contents() as *const T;
        let len = buffer.length() as usize / mem::size_of::<T>();
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

        slice.to_vec()
    }
}
