use std::process::Output;
use metal::{Buffer, MTLSize};
use objc::rc::autoreleasepool;
use std::time::Instant;
use crate::msm::metal::abstraction::state::MetalState;
use crate::msm::metal::msm::{MetalMsmConfig, MetalMsmInstance};

// Helper function to round up `x` to the next power of two
fn next_power_of_two(mut x: u64) -> u64 {
    if x == 0 {
        return 1;
    }
    x -= 1;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x + 1
}

/// Dispatch the "local sort" kernel that sorts each sub-block of size `block_size`.
fn dispatch_local_sort(
    config: &MetalMsmConfig,
    data_buffer: &metal::Buffer,
    length: usize,
    block_size: u32,
) {
    // We'll compute how many threadgroups we need:
    // Each threadgroup handles one block of `block_size`.
    let total_elems = length as u32;
    let num_threadgroups = (total_elems + block_size - 1) / block_size;

    // We assume each threadgroup spawns `block_size` threads (if possible).
    // But watch out for device limits on max threadgroup size.
    let threads_per_group = block_size.min(config.pipelines.local_sort_buckets_indices.max_total_threads_per_threadgroup() as u32);

    autoreleasepool(|| {

        let total_elems_buf  = config.state.alloc_buffer_data(&[total_elems]);
        let block_size_buf   = config.state.alloc_buffer_data(&[block_size]);

        let (command_buffer, command_encoder) = config.state.setup_command(
            &config.pipelines.local_sort_buckets_indices,
            Some(&[
                (0, &data_buffer),
                (1, &total_elems_buf),
                (2, &block_size_buf),
            ]),
        );

        // Shared memory size needed
        let shared_mem_size = (block_size as usize) * std::mem::size_of::<u32>() * 2; // 2 u32 per item
        command_encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);

        let tg_count = MTLSize::new(num_threadgroups as u64, 1, 1);
        let tg_size  = MTLSize::new(threads_per_group as u64, 1, 1);

        command_encoder.dispatch_thread_groups(tg_count, tg_size);
        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    });
}

fn dispatch_merge_pass(
    config: &MetalMsmConfig,
    output_buffer: &Buffer,
    input_buffer: &Buffer,
    length: usize,
    curr_block_size: u32,
) {
    // Merging sub-blocks of size curr_block_size into 2*curr_block_size
    let total_elems = length as u32;
    println!("Dispatching merge pass: total_elems = {}, curr_block_size = {}", total_elems, curr_block_size);
    let num_threadgroups = (total_elems + (curr_block_size * 2) - 1) / (curr_block_size * 2);

    let threads_per_group = curr_block_size.min(config.pipelines.merge_sort_buckets_indices.max_total_threads_per_threadgroup() as u32);

    // The shared memory usage is tile_size * 2 * sizeof(uint2).
    // We'll pick tile_size=256 or so inside the kernel.
    // That needs e.g. 256*2*8=4096 bytes, which fits 2^12.
    let tile_size: usize = 4; // must match the kernel
    let shared_mem_size = 4 * tile_size * 2 * std::mem::size_of::<u32>(); // tile_size*2*(u32 per item?), or tile_size * 2 * 8 if directly for uint2
    // but we rely on the kernel to interpret it. We'll be safe with a bit overhead.


    autoreleasepool(|| {

        let total_elems_buf  = config.state.alloc_buffer_data(&[total_elems]);
        let block_size_buf   = config.state.alloc_buffer_data(&[curr_block_size]);

        let (command_buffer, command_encoder) = config.state.setup_command(
            &config.pipelines.merge_sort_buckets_indices,
            Some(&[
                (0, &input_buffer),
                (1, &output_buffer),
                (2, &total_elems_buf),
                (3, &block_size_buf),
            ]),
        );


        // Set the shared memory size for the tile
        command_encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);

        let tg_count = MTLSize::new(num_threadgroups as u64, 1, 1);
        let tg_size  = MTLSize::new(threads_per_group as u64, 1, 1);

        command_encoder.dispatch_thread_groups(tg_count, tg_size);
        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    });
}

/// Executes the `sort_buckets` Metal shader kernel.
/// Sorts the `(uint2)` data in `buckets_indices_buffer` by `.x` ascending.
///
/// # Arguments
///
/// * `config` - The Metal MSM configuration containing pipelines and state.
/// * `instance` - The MSM instance containing data and parameters.
///
/// # Returns
///
/// * `()` - The function modifies the `buckets_indices_buffer` in-place.
pub(crate) fn sort_buckets_indices(
    config: &MetalMsmConfig,
    instance: &MetalMsmInstance,
) {
    let length = instance.params.instances_size as usize;
    println!("Sorting {} buckets", length);
    let block_size = 4u32;
    // Ensure block_size * sizeof(uint2) < 4096 => 256 * 8 = 2048, so we fit in 4KB shared memory.

    let padded_length = next_power_of_two(length as u64) as usize;

    // --- 1) Local Sort pass (sort each block of size=block_size)
    dispatch_local_sort(
        config,
        &instance.data.buckets_indices_buffer,
        length,
        block_size
    );

    // --- 2) Merge pass: doubling from block_size to 2 * block_size, etc
    let mut buffer_a = &instance.data.buckets_indices_buffer;
    let mut buffer_b = &config.state.alloc_buffer::<u32>((instance.params.instances_size * instance.params.num_window * 2) as usize);
    let mut curr_size = block_size as usize;
    let mut pass = 0;
    while curr_size <= padded_length {
        {
            println!("Merge pass {pass}: curr_size = {}, padded_length = {}", curr_size, padded_length);
            let buffer_a = MetalState::retrieve_contents::<u32>(&buffer_a);
            let buffer_b = MetalState::retrieve_contents::<u32>(&buffer_b);

            let unflattened_a = buffer_a.chunks_exact(2).collect::<Vec<_>>();
            let unflattened_b = buffer_b.chunks_exact(2).collect::<Vec<_>>();

            println!("Buffer A: {:?}", unflattened_a);
            println!("Buffer B: {:?}", unflattened_b);

            pass += 1;
        }
        dispatch_merge_pass(
            config,
            if pass % 2 == 0 { buffer_a } else { buffer_b },
            if pass % 2 == 0 { buffer_b } else { buffer_a },
            length,
            curr_size as u32
        );
        // // If we will have the final result in buffer_b, do another round with the same window size
        // // By doing another round, we can ensure that the final result is in buffer_a
        // if curr_size * 2 >= padded_length {
        //     (buffer_a, buffer_b) = (buffer_b, buffer_a);
        //     dispatch_merge_pass(
        //         config,
        //         (buffer_a, buffer_b),
        //         length,
        //         curr_size as u32
        //     );
        // }

        curr_size *= 2;
    }

    {
        println!("After Merge Pass");
        let buffer_a = MetalState::retrieve_contents::<u32>(&buffer_a);
        let buffer_b = MetalState::retrieve_contents::<u32>(&buffer_b);

        let unflattened_a = buffer_a.chunks_exact(2).collect::<Vec<_>>();
        let unflattened_b = buffer_b.chunks_exact(2).collect::<Vec<_>>();

        println!("Buffer A: {:?}", unflattened_a);
        println!("Buffer B: {:?}", unflattened_b);

        pass += 1;
    }



}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::msm::metal::msm::{encode_instances, setup_metal_state};
    use std::collections::HashSet;
    use std::fmt::format;
    use itertools::Itertools;
    use proptest::prelude::*;
    use proptest::collection::vec as prop_vec;
    use rand::SeedableRng;
    use crate::msm::metal::abstraction::state::MetalState;
    use crate::msm::metal::tests::init_logger;

    /// We define a small GPU wrapper for testing
    fn sort_on_gpu(config: &MetalMsmConfig, instance: &mut MetalMsmInstance) -> Vec<u32> {
        // Sort on the GPU
        sort_buckets_indices(config, instance);

        // Read results back from GPU
        MetalState::retrieve_contents::<u32>(&instance.data.buckets_indices_buffer)
    }

    #[test]
    fn test_sort_buckets_indices_small() {
        // Setup: small data
        let mut config = setup_metal_state();

        // Suppose we have a small set of pairs (u32, u32):
        // (15, 2), (10, 1), (500, 3), (0, 0) ...
        // We'll place them in instance.data.buckets_indices_buffer as a flat Vec<u32>
        // let mut data = vec![
        //     15, 2,
        //     10, 1,
        //     500, 3,
        //     0, 0,
        //     10, 2,
        //     10, 3,
        //     65, 1,
        //     94, 2,
        // ];
        let mut data: Vec<u32> = [[22u32, 2], [208, 29], [379, 20], [383, 18], [34, 17], [273, 20], [275, 15], [455, 26], [0, 27], [84, 3], [382, 29], [428, 14], [85, 15], [162, 4], [443, 27], [496, 2], [154, 3], [186, 16], [253, 28], [296, 28], [100, 25], [235, 31], [243, 8], [433, 23], [47, 7], [141, 16], [216, 9], [298, 28], [7, 2], [277, 24], [342, 8], [371, 11]]
            .map(Vec::from).to_vec().iter().flatten().map(|i| i.clone()).collect();

        // Build a mock instance
        let mut instance = create_test_instance(&mut config, data.clone());

        // GPU sort
        let gpu_sorted_flat = sort_on_gpu(&config, &mut instance);

        let gpu_sorted = gpu_sorted_flat
            .chunks_exact(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect::<Vec<(u32, u32)>>();

        // Check that no items were lost
        assert_eq!(gpu_sorted.len(), data.len() / 2);
        let expected_hashset: HashSet<(u32, u32)> = data
            .chunks_exact(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect();
        let gpu_hashset: HashSet<(u32, u32)> = gpu_sorted.iter().cloned().collect();
        assert_eq!(gpu_hashset, expected_hashset);

        println!("GPU sorted: {:?}", gpu_sorted);
        let expected_sorted: Vec<(u32, u32)> = data.chunks(2).sorted_by_key(|v| v[0]).map(|v| (v[0], v[1])).collect();
        println!("Expected:   {:?}", expected_sorted);

        // Check that the elements in the gpu_sorted are sorted
        let mut prev = (0, 0);
        for (x, y) in gpu_sorted {
            assert!(prev.0 <= x, "GPU sort failed: {:?} > {:?}", prev, (x, y));
            prev = (x, y);
        }

    }

    proptest! {
        #[test]
        fn test_sort_buckets_indices_large(
            seed in any::<u64>(),
            log_length in 5usize..9,
            lenght_mul in 1usize..8,
            offset in 0usize..8,
        ) {
            init_logger();

            let mut config = setup_metal_state();

            // Generate random pairs
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let length = (1 << log_length) * lenght_mul + offset;
            log::debug!("Length: {}", length);
            let mut data = Vec::with_capacity(length * 2);
            (0..length).for_each(|_| {
                data.push(rng.gen::<u32>() % (length as u32 * 16));
                data.push(rng.gen::<u32>() % length as u32) ;
            });

            // Build a mock instance
            let mut instance = create_test_instance(&mut config, data.clone());

            // GPU sort
            let gpu_sorted_flat = sort_on_gpu(&config, &mut instance);

            let gpu_sorted = gpu_sorted_flat
                .chunks_exact(2)
                .map(|chunk| (chunk[0], chunk[1]))
                .collect::<Vec<(u32, u32)>>();

            // Check that no items were lost
            assert_eq!(gpu_sorted.len(), data.len() / 2);
            let expected_hashset: HashSet<(u32, u32)> = data
                .chunks_exact(2)
                .map(|chunk| (chunk[0], chunk[1]))
                .collect();
            let gpu_hashset: HashSet<(u32, u32)> = gpu_sorted.iter().cloned().collect();
            // Find the difference between the two hashsets
            let expected_diff: HashSet<_> = expected_hashset.difference(&gpu_hashset).collect();
            let gpu_diff: HashSet<_> = gpu_hashset.difference(&expected_hashset).collect();
            assert!(expected_diff.is_empty() && gpu_diff.is_empty(), "GPU sort failed: {:?} != {:?}", expected_diff, gpu_diff);
            // Check that the elements in the gpu_sorted are sorted
            let mut prev = (0, 0);
            for (x, y) in gpu_sorted {
                if x == 0 && y == 0 {
                    continue;
                }
                assert!(prev.0 <= x, "GPU sort failed: {:?} > {:?}. Length: {}", prev, (x, y), length);
                prev = (x, y);
            }

            log::debug!("GPU sort matches CPU sort for length {}", length);
        }
    }

    /// Creates a MetalMsmInstance with the given (u32) data in the buckets_indices_buffer.
    /// This is a mock/truncated approach. Adjust as needed to fit your actual code.
    fn create_test_instance(
        config: &mut MetalMsmConfig,
        data: Vec<u32>,
    ) -> MetalMsmInstance {
        // We'll create a minimal MetalMsmInstance with only the buckets_indices_buffer set
        use crate::msm::metal::msm::{MetalMsmParams, MetalMsmData};

        let length = data.len() / 2;
        let buckets_indices_buffer = config.state.alloc_buffer_data(&data);

        let instance_data = MetalMsmData {
            window_size_buffer: config.state.alloc_buffer_data(&[0]),
            instances_size_buffer: config.state.alloc_buffer_data(&[length as u32]),
            window_starts_buffer: config.state.alloc_buffer_data(&[0]),
            scalar_buffer: config.state.alloc_buffer::<u32>(0),
            base_buffer: config.state.alloc_buffer::<u32>(0),
            num_windows_buffer: config.state.alloc_buffer_data(&[1u32]),
            buckets_matrix_buffer: config.state.alloc_buffer::<u32>(0),
            buckets_indices_buffer,
            res_buffer: config.state.alloc_buffer::<u32>(0),
            result_buffer: config.state.alloc_buffer::<u32>(0),
        };

        let instance_params = MetalMsmParams {
            instances_size: length as u32,
            buckets_size: 0,
            window_size: 0,
            num_window: 1,
        };

        MetalMsmInstance {
            data: instance_data,
            params: instance_params,
        }
    }
}