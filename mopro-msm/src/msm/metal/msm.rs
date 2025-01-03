mod prepare_buckets_indices;

use std::ops::Add;
use std::sync::{Arc, Mutex};
use crate::msm::metal::abstraction::{
    errors::MetalError,
    limbs_conversion::{FromLimbs},
    state::*,
};
use crate::msm::utils::{benchmark::BenchmarkResult, preprocess};
use ark_std::{vec::Vec};
// For benchmarking
use std::time::{Duration, Instant};
use crate::msm::metal::abstraction::limbs_conversion::{PointGPU, ScalarGPU};
use crate::msm::utils::preprocess::{get_or_create_msm_instances, MsmInstance};
use metal::*;
use objc::rc::autoreleasepool;
use rand::rngs::OsRng;
use rayon::prelude::{ParallelSliceMut, ParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};
use crate::msm::metal::msm::prepare_buckets_indices::prepare_buckets_indices;

pub struct MetalMsmData {
    pub window_size_buffer: Buffer,
    pub instances_size_buffer: Buffer,
    pub window_starts_buffer: Buffer,
    pub scalar_buffer: Buffer,
    pub base_buffer: Buffer,
    pub num_windows_buffer: Buffer,
    pub buckets_indices_buffer: Buffer,
    pub buckets_matrix_buffer: Buffer,
    pub res_buffer: Buffer,
    pub result_buffer: Buffer,
    // pub debug_buffer: Buffer,
}

pub struct MetalMsmParams {
    pub instances_size: u32,
    pub buckets_size: u32,
    pub window_size: u32,
    pub num_window: u32,
}

pub struct MetalMsmPipeline {
    pub prepare_buckets_indices: ComputePipelineState,
    pub sort_buckets: ComputePipelineState,
    pub bucket_wise_accumulation: ComputePipelineState,
    pub sum_reduction: ComputePipelineState,
    pub final_accumulation: ComputePipelineState,

    // New one-shot kernel:
    pub msm_all_gpu: ComputePipelineState,
}

pub struct MetalMsmConfig {
    pub state: MetalState,
    pub pipelines: MetalMsmPipeline,
}

pub struct MetalMsmInstance {
    pub data: MetalMsmData,
    pub params: MetalMsmParams,
}

// Helper function for getting the windows size
// TODO - find out the heuristic
fn ln_without_floats(a: usize) -> usize {
    // log2(a) * ln(2)
    (ark_std::log2(a) * 69 / 100) as usize
}

pub fn setup_metal_state() -> MetalMsmConfig {
    let state = MetalState::new(None).unwrap();
    let final_accumulation = state.setup_pipeline("final_accumulation").unwrap();

    // TODO:
    let prepare_buckets_indices = state.setup_pipeline("prepare_buckets_indices").unwrap();
    let sort_buckets = state.setup_pipeline("sort_buckets_indices_multi").unwrap();
    let bucket_wise_accumulation = state.setup_pipeline("bucket_wise_accumulation").unwrap();
    let sum_reduction = state.setup_pipeline("sum_reduction").unwrap();

    // Our new kernel:
    let msm_all_gpu = state.setup_pipeline("msm_all_gpu").unwrap();

    MetalMsmConfig {
        state,
        pipelines: MetalMsmPipeline {
            prepare_buckets_indices,
            sort_buckets,
            bucket_wise_accumulation,
            sum_reduction,
            final_accumulation,
            msm_all_gpu,
        },
    }
}


pub fn encode_instances<P: PointGPU<NP> + Sync, S: ScalarGPU<NS> + Sync, const NP: usize, const NS: usize>(
    points: &[P],
    scalars: &[S],
    config: &mut MetalMsmConfig,
    window_size: Option<u32>
) -> MetalMsmInstance {
    let modulus_bit_size = S::MODULUS_BIT_SIZE;

    let instances_size = ark_std::cmp::min(points.len(), scalars.len());
    let window_size = if let Some(window_size) = window_size {
        window_size
    } else {
        if instances_size < 32 {
            3
         } else {
            17 // TODO - learn how to calculate this
        }
    };
    let buckets_size = (1 << window_size) - 1;
    let window_starts: Vec<u32> = (0..modulus_bit_size as u32).step_by(window_size as usize).collect();
    let num_windows = window_starts.len();

    // flatten scalar and base to Vec<u32> for GPU usage
    let flatten_start = Instant::now();

    // Preallocate scalars and bases
    let mut scalars_limbs = vec![0u32; scalars.len() * NS];
    let mut bases_limbs = vec![0u32; points.len() * NP];

    // Fill scalars_limbs using write_u32_limbs in parallel
    scalars
        .par_iter()
        .zip(scalars_limbs.par_chunks_mut(NS))
        .for_each(|(scalar, chunk)| {
            scalar.write_u32_limbs(chunk.try_into().unwrap());
        });

    // Fill bases_limbs using write_u32_limbs in parallel
    points
        .par_iter()
        .zip(bases_limbs.par_chunks_mut(NP))
        .for_each(|(point, chunk)| {
            point.write_u32_limbs(chunk.try_into().unwrap());
        });

    log::debug!("Encoding flatten time: {:?}", flatten_start.elapsed());


    // store params to GPU shared memory
    let store_params_start = Instant::now();
    let window_size_buffer = config.state.alloc_buffer_data(&[window_size as u32]);
    let instances_size_buffer = config.state.alloc_buffer_data(&[instances_size as u32]);
    let scalar_buffer = config.state.alloc_buffer_data(&scalars_limbs);
    let base_buffer = config.state.alloc_buffer_data(&bases_limbs);
    let num_windows_buffer = config.state.alloc_buffer_data(&[num_windows as u32]);
    let buckets_matrix_buffer = config
        .state
        .alloc_buffer::<u32>(buckets_size * num_windows * 8 * 3);
    let res_buffer = config.state.alloc_buffer::<u32>(num_windows * 8 * 3);
    let result_buffer = config.state.alloc_buffer::<u32>(8 * 3);
    // convert window_starts to u32 to give the exact storage need for GPU
    let window_starts_buffer = config.state.alloc_buffer_data(&window_starts);
    // prepare bucket_size * num_windows * 2
    let buckets_indices_buffer = config
        .state
        .alloc_buffer::<u32>(instances_size * num_windows * 2);
    log::debug!("Store params time: {:?}", store_params_start.elapsed());

    // // debug
    // let debug_buffer = config.state.alloc_buffer::<u32>(2048);

    MetalMsmInstance {
        data: MetalMsmData {
            window_size_buffer,
            instances_size_buffer,
            window_starts_buffer,
            scalar_buffer,
            base_buffer,
            num_windows_buffer,
            buckets_matrix_buffer,
            buckets_indices_buffer,
            res_buffer,
            result_buffer,
            // debug_buffer,
        },
        params: MetalMsmParams {
            instances_size: instances_size as u32,
            buckets_size: buckets_size as u32,
            window_size: window_size as u32,
            num_window: num_windows as u32,
        },
    }
}

pub fn exec_metal_commands<P: FromLimbs>(
    config: &MetalMsmConfig,
    instance: MetalMsmInstance,
) -> Result<P, MetalError> {
    let data = &instance.data;
    let params = &instance.params;

    let prepare_time = Instant::now();
    prepare_buckets_indices(&config, &instance);
    log::debug!("Prepare buckets indices time: {:?}", prepare_time.elapsed());


    {
        let length = (params.instances_size * params.num_window) as usize;

        let next_power_of_two = |x: u64| -> u64 {
            let mut x = x;
            x -= 1;
            x |= x >> 1;
            x |= x >> 2;
            x |= x >> 4;
            x |= x >> 8;
            x |= x >> 16;
            x |= x >> 32;
            x += 1;
            x
        };

        // For bitonic sort, we want length to be a power-of-two.
        // If it's not, we can pad. For correctness, do so:
        let padded_length = next_power_of_two(length as u64) as usize;

        // Our chosen local block size:
        //   - typically 1024 is a good block size on many GPUs.
        let local_block_size = 1024;

        // Step 1: local sort pass
        // We'll dispatch enough threadgroups to cover (padded_length / local_block_size).
        // For each threadgroup, 'block_size = local_block_size', sorting that chunk in local memory.

        // We'll define a small buffer for 'block_size', 'stage', etc. repeatedly.
        // We'll define a function to dispatch one pass:
        let mut pass_number = 0;
        let now = Instant::now();

        // local function to dispatch one pass
        let mut dispatch_pass = |block_size: u32| {
            let total_elems = padded_length as u32;

            // # of threadgroups
            let num_threadgroups = (total_elems + block_size - 1) / block_size;
            let mtl_threadgroups = MTLSize::new(num_threadgroups as u64, 1, 1);

            // threads_per_group = block_size, or clamp if device can't handle that many threads in one group
            let max_tg = config.pipelines.sort_buckets.max_total_threads_per_threadgroup();
            let threads_per_group = block_size.min(max_tg as u32);

            let mtl_threads_per_group = MTLSize::new(threads_per_group as u64, 1, 1);

            // Alloc small buffers for total_elems, block_size, stage, substage
            let total_elems_buf  = config.state.alloc_buffer_data(&[total_elems]);
            let block_size_buf   = config.state.alloc_buffer_data(&[block_size]);
            let stage_buf        = config.state.alloc_buffer_data(&[pass_number]);
            let substage_buf     = config.state.alloc_buffer_data(&[0u32]);

            let (command_buffer, command_encoder) = config.state.setup_command(
                &config.pipelines.sort_buckets, // <--- The new pipeline
                Some(&[
                    (0, &data.buckets_indices_buffer), // device uint2 *data
                    (1, &total_elems_buf),
                    (2, &block_size_buf),
                    (3, &stage_buf),
                    (4, &substage_buf),
                ]),
            );

            // We need threadgroup memory for 'block_size' * sizeof(uint2).
            // block_size is up to 1024 typically, so 1024 * 8 = 8192 bytes
            let shared_mem_size = (block_size as usize) * std::mem::size_of::<u32>() * 2; // 2 u32 per uint2
            command_encoder.set_threadgroup_memory_length(0, shared_mem_size as u64);

            command_encoder.dispatch_thread_groups(mtl_threadgroups, mtl_threads_per_group);
            command_encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            pass_number += 1;
        };

        // --- PASS A) local sort sub-blocks of 1024 (or fewer if near end)
        dispatch_pass(local_block_size);

        // --- PASS B) Merging passes
        // We double the sub-block size each pass until >= padded_length.
        let mut size = local_block_size as usize;
        while size < padded_length {
            let new_size = size * 2;
            dispatch_pass(new_size as u32);
            size = new_size;
        }

        // Done. If original length < padded_length, the "extra" region is sorted sentinel elements,
        // so your real data is still sorted in the first 'length' portion.

        log::debug!("Sort buckets indices time: {:?}", now.elapsed());
    }

    {
        // 1. Calculate total bucket IDs
        let total_buckets = params.buckets_size * params.num_window;

        // 2. Limit maximum GPU threads (e.g., 2^17)
        let max_gpu_threads = 2 << 17;
        let actual_threads  = total_buckets.min(max_gpu_threads) as NSUInteger;

        // 3. Threads per group
        let threads_per_group = config
            .pipelines
            .bucket_wise_accumulation
            .max_total_threads_per_threadgroup();

        // 4. Number of threadgroups
        let num_thread_groups = (actual_threads + threads_per_group - 1) / threads_per_group;

        // 5. Prepare MTLSize
        let mtl_threadgroups = MTLSize::new(num_thread_groups, 1, 1);
        let mtl_threads_per_group = MTLSize::new(threads_per_group, 1, 1);

        // 6. Create small buffer for '_actual_threads'
        let actual_threads_buffer = config.state.alloc_buffer_data(&[actual_threads]);

        // 7. Dispatch GPU accumulation kernel
        let bucket_wise_time = Instant::now();
        autoreleasepool(|| {
            let (command_buffer, command_encoder) = config.state.setup_command(
                &config.pipelines.bucket_wise_accumulation,
                Some(&[
                    (0, &data.instances_size_buffer),     // _instances_size
                    (1, &data.num_windows_buffer),        // _num_windows
                    (2, &data.base_buffer),               // p_buff
                    (3, &data.buckets_indices_buffer),    // buckets_indices
                    (4, &data.buckets_matrix_buffer),     // buckets_matrix
                    (5, &actual_threads_buffer),          // _actual_threads
                    // If needed, also pass 'buckets_indices_len' if it differs from total_buckets
                ]),
            );

            command_encoder.dispatch_thread_groups(mtl_threadgroups, mtl_threads_per_group);
            command_encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        let bucket_wise_elapsed = bucket_wise_time.elapsed();
        log::debug!(
            "Bucket-wise accumulation time (using {} threads, chunked): {:?}",
            actual_threads,
            bucket_wise_elapsed
        );
    }

    {
    // Number of windows = number of thread groups
        let num_thread_groups = params.num_window as NSUInteger;

        // Each thread group is responsible for one window.
        // Tune this value as needed for performance.
        let desired_point_additions_per_thread = 16;
        let threads_per_group = ((1 << params.window_size as NSUInteger + desired_point_additions_per_thread - 1) / desired_point_additions_per_thread)
            .min(config.pipelines.sum_reduction.max_total_threads_per_threadgroup());
        log::debug!("(reduction) max thread per threadgroup: {:?}", config.pipelines.sum_reduction.max_total_threads_per_threadgroup());

        log::debug!("(reduction) num thread groups: {:?}", num_thread_groups);
        log::debug!("(reduction) threads per group: {:?}", threads_per_group);
        log::debug!("(reduction) point additions per thread: {:?} (desired {desired_point_additions_per_thread})", (1 << params.window_size) / threads_per_group);

        // Prepare Metal size structs
        let mtl_threadgroups = MTLSize::new(num_thread_groups, 1, 1);
        let mtl_threads_per_group = MTLSize::new(threads_per_group, 1, 1);

        // We no longer need a buffer for 'max_thread_size'
        let reduction_time = Instant::now();
        autoreleasepool(|| {
            let (command_buffer, command_encoder) = config.state.setup_command(
                &config.pipelines.sum_reduction,
                Some(&[
                    (0, &data.window_size_buffer),  // c
                    (1, &data.scalar_buffer),       // k_buff
                    (2, &data.base_buffer),         // p_buff
                    (3, &data.buckets_matrix_buffer),
                    (4, &data.res_buffer),
                    // No buffer(5) needed anymore
                ]),
            );

            // Dispatch: one threadgroup per window, 'threads_per_group' threads in each group
            command_encoder.dispatch_thread_groups(mtl_threadgroups, mtl_threads_per_group);
            command_encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        log::debug!("Reduction time: {:?}", reduction_time.elapsed());
    }

    {
        // Sequentially accumulate the msm results on GPU
        let final_time = Instant::now();
        autoreleasepool(|| {
            let (command_buffer, command_encoder) = config.state.setup_command(
                &config.pipelines.final_accumulation,
                Some(&[
                    (0, &data.window_size_buffer),
                    (1, &data.window_starts_buffer),
                    (2, &data.num_windows_buffer),
                    (3, &data.res_buffer),
                    (4, &data.result_buffer),
                ]),
            );
            command_encoder.dispatch_thread_groups(MTLSize::new(1, 1, 1), MTLSize::new(1, 1, 1));
            command_encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });
        log::debug!("Final accumulation time: {:?}", final_time.elapsed());
    }

    // retrieve and parse the result from GPU
    let msm_result = {
        let raw_limbs = MetalState::retrieve_contents::<u32>(&data.result_buffer);
        P::from_u32_limbs(&raw_limbs)
    };

    Ok(msm_result)
}

/// One-shot GPU-based MSM using the `msm_all_gpu` kernel.
///
/// - `P` is the GPU-friendly point type (e.g., BN254),
/// - `S` is the GPU-friendly scalar type (e.g., BN254 scalar).
///
pub fn metal_msm_all_gpu<P, S>(
    points: &[P],
    scalars: &[S],
    config: &mut MetalMsmConfig,
    batch_size: Option<u32>,
    threads_per_tg: Option<u32>,
) -> Result<P, MetalError>
where
    P: PointGPU<24> + FromLimbs + Add<P, Output = P>,
    S: ScalarGPU<8>,
{
    // -----------------------------------------------------------------------
    // 1. Flatten points and scalars into u32-limb buffers
    // -----------------------------------------------------------------------
    let n = points.len().min(scalars.len());
    let scalars_limbs: Vec<u32> = scalars
        .iter()
        .take(n)
        .flat_map(|s| s.to_u32_limbs())
        .collect();

    let points_limbs: Vec<u32> = points
        .iter()
        .take(n)
        .flat_map(|p| p.to_u32_limbs())
        .collect();

    // Number of threads per threadgroup
    // The partial result for each threadgroup will be 96 bytes => 24 u32
    // This is calculated based on max threadgroup shared memory size: ~ 32 KB / 96 bytes (size of a Point)
    const MAX_MAX_THREADS_PER_TG: u32 = 340;
    let threads_per_tg: u32 = if let Some(threads_per_tg) = threads_per_tg {
        if threads_per_tg <= MAX_MAX_THREADS_PER_TG {
            threads_per_tg
        } else {
            log::warn!(
                "threads_per_tg ({}) exceeds the maximum allowed value ({}). Using {} instead.",
                threads_per_tg,
                MAX_MAX_THREADS_PER_TG,
                MAX_MAX_THREADS_PER_TG
            );
            MAX_MAX_THREADS_PER_TG
        }
    } else {
        340
    };

    let batch_size: u32 = if let Some(batch_size) = batch_size {
        batch_size
    } else {
        ((points.len() as u32) / 16).div_ceil(threads_per_tg)
    };

    // chunk_size = threads_per_tg * batch_size
    let chunk_size = (threads_per_tg as u32) * batch_size;

    // number of threadgroups needed
    let num_tg = (n as u32 + chunk_size - 1) / chunk_size;

    log::debug!("Thread Configuration Information:");
    log::debug!("  - threads_per_tg: {}", threads_per_tg);
    log::debug!("  - batch_size (MSM per thread): {}", batch_size);
    log::debug!("  - chunk_size (MSM per thread group): {}", chunk_size);
    log::debug!("  - num_tg (total thread groups): {}", num_tg);

    // -----------------------------------------------------------------------
    // 2. Allocate GPU buffers
    // -----------------------------------------------------------------------
    let scalar_buffer = config.state.alloc_buffer_data(&scalars_limbs);
    let point_buffer = config.state.alloc_buffer_data(&points_limbs);

    // partial_results_buffer: store one partial sum (24 u32) per threadgroup
    let partial_results_buffer = config
        .state
        .alloc_buffer::<u32>((num_tg as usize) * 24 /* 96 bytes => 24 u32 */);

    // pass the total size (n) to the kernel
    let total_size_buffer = config.state.alloc_buffer_data(&[n as u32]);

    // pass the batch size to the kernel
    let batch_size_buffer = config.state.alloc_buffer_data(&[batch_size as u32]);

    // -----------------------------------------------------------------------
    // 3. Encode the `msm_all_gpu` kernel dispatch
    // -----------------------------------------------------------------------
    let start = Instant::now();
    autoreleasepool(|| {
        let (command_buffer, command_encoder) = config.state.setup_command(
            &config.pipelines.msm_all_gpu,
            Some(&[
                (0, &point_buffer),
                (1, &scalar_buffer),
                (2, &total_size_buffer),
                (3, &batch_size_buffer),
                (4, &partial_results_buffer),
            ]),
        );

        // We dispatch 'num_tg' threadgroups, each with 'threads_per_tg' threads
        let threadgroups = MTLSize::new(num_tg as NSUInteger, 1, 1);
        let threads_per_group = MTLSize::new(threads_per_tg as u64, 1, 1);

        command_encoder.dispatch_thread_groups(threadgroups, threads_per_group);
        command_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    });
    let gpu_time = start.elapsed();
    log::debug!("msm_all_gpu kernel took {:?}", gpu_time);

    // -----------------------------------------------------------------------
    // 4. Read back partial sums & reduce on CPU
    // -----------------------------------------------------------------------
    let partial_sums = MetalState::retrieve_contents::<u32>(&partial_results_buffer);

    // Each partial sum is 24 u32. We'll parse them into points.
    let mut final_result = P::from_u32_limbs(&partial_sums[0..24]); // first partial sum
    for tg_id in 24..(num_tg as usize) {
        // slice [tg_id*24 .. tg_id*24 + 24]
        let offset = tg_id * 24;
        let chunk_limbs = &partial_sums[offset..offset + 24];
        let partial_pt = P::from_u32_limbs(chunk_limbs);
        final_result = final_result + partial_pt;
    }

    Ok(final_result)
}

pub fn metal_msm<P: PointGPU<24> + Sync, S: ScalarGPU<8> + Sync>(
    points: &[P],
    scalars: &[S],
    config: &mut MetalMsmConfig,
) -> Result<P, MetalError> {
    let instance = encode_instances(points, scalars, config, None);
    exec_metal_commands(config, instance)
}


pub fn metal_msm_parallel<P, S>(instance: &MsmInstance<P, S>, target_msm_log_size: Option<usize>) -> P
where
    P: PointGPU<24> + Add<P, Output = P> + Send + Sync + Clone,
    S: ScalarGPU<8> + Send + Sync,
{
    let points = &instance.points;
    let scalars = &instance.scalars;

    // We believe optimal chunk size is 1/3 of the target MSM length
    let chunk_size = if let Some(target_msm_log_size) = target_msm_log_size {
        2usize.pow(target_msm_log_size as u32)
    } else {
        points.len() / 3
    };

    // Shared accumulators
    let accumulator = Arc::new(Mutex::new(P::from_u32_limbs(&[0; 24])));
    let metal_state_time = Arc::new(Mutex::new(Duration::ZERO));
    let encoding_time = Arc::new(Mutex::new(Duration::ZERO));
    let exec_time = Arc::new(Mutex::new(Duration::ZERO));
    let sleep_time = Arc::new(Mutex::new(Duration::ZERO));

    points
        .chunks(chunk_size)
        .zip(scalars.chunks(chunk_size))
        .enumerate() // Add index for delay
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|(i, (pts_chunk, scs_chunk))| {
            // Introduce delay based on chunk index
            let sleep_start = instant::Instant::now();
            // std::thread::sleep(Duration::from_millis(i as u64 * 150));
            let sleep_elapsed = sleep_start.elapsed();
            {
                let mut sleep_time = sleep_time.lock().unwrap();
                *sleep_time += sleep_elapsed;
            }
            // Measure time for setting up metal state
            let metal_config_start = instant::Instant::now();
            let mut metal_config = setup_metal_state();
            let metal_state_elapsed = metal_config_start.elapsed();
            {
                let mut state_time = metal_state_time.lock().unwrap();
                *state_time += metal_state_elapsed;
            }
            log::debug!("Done setting up metal state in {:?}", metal_state_elapsed);

            // Measure time for encoding instances
            let encoding_start = instant::Instant::now();
            let metal_instance = encode_instances(pts_chunk, scs_chunk, &mut metal_config, None);
            let encoding_elapsed = encoding_start.elapsed();
            {
                let mut enc_time = encoding_time.lock().unwrap();
                *enc_time += encoding_elapsed;
            }

            // Measure time for executing commands
            let exec_start = instant::Instant::now();
            let partial_result = exec_metal_commands(&metal_config, metal_instance).unwrap();
            let exec_elapsed = exec_start.elapsed();
            {
                let mut ex_time = exec_time.lock().unwrap();
                *ex_time += exec_elapsed;
            }

            // Add partial result to the shared accumulator
            let mut acc = accumulator.lock().unwrap();
            *acc = acc.clone() + partial_result;

            log::info!("Finished compute thread {i}");
        });

    // Log aggregated times
    log::info!(
        "Total time spent on metal state setup: {:?}",
        *metal_state_time.lock().unwrap()
    );
    log::info!(
        "Total time spent on encoding instances: {:?}",
        *encoding_time.lock().unwrap()
    );
    log::info!(
        "Total time spent on executing commands: {:?}",
        *exec_time.lock().unwrap()
    );
    log::info!(
        "Total time spent on sleeping : {:?}",
        *sleep_time.lock().unwrap()
    );

    // Extract the final accumulated result
    Arc::try_unwrap(accumulator)
        .map_err(|_| "Failed to unwrap accumulator")
        .expect("Failed to unwrap accumulator")
        .into_inner()
        .unwrap()
}

pub fn benchmark_msm<P: PointGPU<24> + Sync, S: ScalarGPU<8> + Sync>(
    instances: Vec<MsmInstance<P, S>>,
    iterations: u32,
) -> Result<Vec<Duration>, preprocess::HarnessError> {
    log::info!("Init metal (GPU) state...");
    let init_start = Instant::now();
    let mut metal_config = setup_metal_state();
    let init_duration = init_start.elapsed();
    log::info!("Done initializing metal (GPU) state in {:?}", init_duration);

    let mut instance_durations = Vec::new();
    for instance in instances {
        let points = &instance.points;
        // map each scalar to a ScalarField
        let scalars = &instance.scalars;

        let mut instance_total_duration = Duration::ZERO;
        for _i in 0..iterations {
            let encoding_data_start = Instant::now();
            log::info!("Encoding instance to GPU memory...");
            let metal_instance = encode_instances(points, &scalars[..], &mut metal_config, None);
            let encoding_data_duration = encoding_data_start.elapsed();
            log::info!("Done encoding data in {:?}", encoding_data_duration);

            let msm_start = Instant::now();
            let _result = exec_metal_commands::<P>(&metal_config, metal_instance).unwrap();
            instance_total_duration += msm_start.elapsed();
        }
        let instance_avg_duration = instance_total_duration / iterations;

        log::info!(
            "Average time to execute MSM with {} points and {} scalars in {} iterations is: {:?}",
            points.len(),
            scalars.len(),
            iterations,
            instance_avg_duration,
        );
        instance_durations.push(instance_avg_duration);
    }
    Ok(instance_durations)
}

pub fn run_benchmark<P: PointGPU<24> + Sync, S: ScalarGPU<8> + FromLimbs + Sync>(
    log_instance_size: u32,
    num_instance: u32,
    utils_dir: Option<&str>,
) -> Result<BenchmarkResult, preprocess::HarnessError> {

    let rng = OsRng::default();

    let benchmark_data: Vec<MsmInstance<P, S>> = get_or_create_msm_instances(log_instance_size, num_instance, rng, utils_dir)?;

    let instance_durations = benchmark_msm(benchmark_data, 1)?;
    // in milliseconds
    let avg_processing_time: f64 = instance_durations
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .sum::<f64>()
        / instance_durations.len() as f64;

    log::info!("Done running benchmark.");
    Ok(BenchmarkResult {
        instance_size: log_instance_size,
        num_instance,
        avg_processing_time,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Once;

    // Static initializer to ensure the logger is initialized only once
    static INIT: Once = Once::new();

    fn init_logger() {
        INIT.call_once(|| {
            env_logger::builder()
                .is_test(true) // Ensures logs go to stdout/stderr in a test-friendly way
                .init();
        });
    }

    const LOG_INSTANCE_SIZE: u32 = 16;
    const NUM_INSTANCE: u32 = 10;
    const BENCHMARKSPATH: &str = "benchmark_results";

    #[cfg(feature = "ark")]
    mod ark {
        use std::env;
        use std::fs::File;
        use std::io::Write;
        use ark_ec::{CurveGroup, VariableBaseMSM};
        use ark_std::cfg_into_iter;
        use rand::rngs::OsRng;
        use crate::msm::metal::abstraction::limbs_conversion::ark::{ArkFr, ArkG};
        use crate::msm::metal::msm::{metal_msm, metal_msm_all_gpu, metal_msm_parallel, run_benchmark, setup_metal_state};
        use crate::msm::metal::msm::tests::{init_logger, BENCHMARKSPATH, LOG_INSTANCE_SIZE, NUM_INSTANCE};
        use crate::msm::utils::preprocess::{get_or_create_msm_instances};

        #[test]
        fn test_msm_correctness_medium_sample_ark() {
            init_logger();

            let mut metal_config = setup_metal_state();
            let rng = OsRng::default();

            let instances = get_or_create_msm_instances::<ArkG, ArkFr>(LOG_INSTANCE_SIZE, NUM_INSTANCE, rng, None).unwrap();

            for (i, instance) in instances.iter().enumerate() {
                let points = &instance.points;
                // map each scalar to a ScalarField
                let scalars = &instance.scalars;

                let affine_points: Vec<_> = cfg_into_iter!(points).map(|p| p.into_affine()).collect();
                let ark_msm = ArkG::msm(&affine_points, &scalars[..]).unwrap();

                let metal_msm = metal_msm::<ArkG, ArkFr>(&points[..], &scalars[..], &mut metal_config).unwrap();
                assert_eq!(metal_msm.into_affine(), ark_msm.into_affine(), "This msm is wrongly computed");
                log::info!(
                    "(pass) {}th instance of size 2^{} is correctly computed",
                    i, LOG_INSTANCE_SIZE
                );
            }
        }

        #[test]
        #[ignore]
        fn test_run_multi_benchmarks() {
            init_logger();

            let output_path = format!(
                "{}/{}/{}_ark_benchmark.txt",
                env::var("CARGO_MANIFEST_DIR").unwrap(),
                &BENCHMARKSPATH,
                "metal_msm"
            );
            let mut output_file = File::create(output_path).expect("output file creation failed");
            writeln!(output_file, "msm_size,num_msm,avg_processing_time(ms)").unwrap();

            let log_instance_sizes: Vec<u32> = vec![8, 12, 16, 18, 20, 22];
            let num_instance: Vec<u32> = vec![10];

            for log_size in log_instance_sizes {
                for num in &num_instance {
                    let result = run_benchmark::<ArkG, ArkFr>(log_size, *num, None).unwrap();
                    log::info!("{}x{} result: {:#?}", log_size, *num, result);
                    writeln!(
                        output_file,
                        "{},{},{}",
                        result.instance_size, result.num_instance, result.avg_processing_time
                    )
                    .unwrap();
                }
            }
        }

        #[test]
        fn test_parallel_gpu_metal_msm_correctness() {
            init_logger();

            let rng = OsRng::default();
            let mut metal_config = setup_metal_state();

            let instances = get_or_create_msm_instances::<ArkG, ArkFr>(LOG_INSTANCE_SIZE, NUM_INSTANCE, rng, None).unwrap();

            for (i, instance) in instances.iter().enumerate() {
                let points = &instance.points;
                // map each scalar to a ScalarField
                let scalars = &instance.scalars;

                let affine_points: Vec<_> = cfg_into_iter!(points).map(|p| p.into_affine()).collect();
                let ark_msm = ArkG::msm(&affine_points, &scalars[..]).unwrap();

                let metal_msm_par = metal_msm_parallel(instance, None);
                let metal_msm = metal_msm::<ArkG, ArkFr>(&points[..], &scalars[..], &mut metal_config).unwrap();
                assert_eq!(metal_msm.into_affine(), ark_msm.into_affine(), "This msm is wrongly computed");
                assert_eq!(metal_msm.into_affine(), metal_msm_par.into_affine(), "This parallel msm is wrongly computed");
                log::info!(
                    "(pass) {}th instance of size 2^{} is correctly computed",
                    i, LOG_INSTANCE_SIZE
                );
            }
        }

        #[test]
        fn test_all_gpu_correctness_medium_sample_ark() {
            init_logger();

            let mut metal_config = setup_metal_state();
            let rng = OsRng::default();

            let instances = get_or_create_msm_instances::<ArkG, ArkFr>(LOG_INSTANCE_SIZE, NUM_INSTANCE, rng, None).unwrap();

            for (i, instance) in instances.iter().enumerate() {
                let points = &instance.points;
                // map each scalar to a ScalarField
                let scalars = &instance.scalars;

                let affine_points: Vec<_> = cfg_into_iter!(points).map(|p| p.into_affine()).collect();
                let ark_msm = ArkG::msm(&affine_points, &scalars[..]).unwrap();

                let metal_msm = metal_msm_all_gpu(&points[..], &scalars[..], &mut metal_config, None, None).unwrap();
                assert_eq!(metal_msm.into_affine(), ark_msm.into_affine(), "This msm is wrongly computed");
                log::info!(
                    "(pass) {}th instance of size 2^{} is correctly computed",
                    i, LOG_INSTANCE_SIZE
                );
            }
        }

    }


    #[cfg(feature = "h2c")]
    mod h2c {
        use std::env;
        use std::fs::File;
        use std::io::Write;
        use ark_std::cfg_into_iter;
        use halo2curves::group::Curve;
        use rand::rngs::OsRng;
        use crate::msm::metal::abstraction::limbs_conversion::h2c::{H2Fr, H2G};
        use crate::msm::metal::msm::{metal_msm, run_benchmark, setup_metal_state};
        use crate::msm::metal::msm::tests::{init_logger, BENCHMARKSPATH, LOG_INSTANCE_SIZE, NUM_INSTANCE};
        use crate::msm::utils::preprocess::get_or_create_msm_instances;

        #[test]
        fn test_msm_correctness_medium_sample_h2c() {
            init_logger();

            let mut metal_config = setup_metal_state();
            let rng = OsRng::default();

            let instances = get_or_create_msm_instances::<H2G, H2Fr>(LOG_INSTANCE_SIZE, NUM_INSTANCE, rng, None).unwrap();

            for (i, instance) in instances.iter().enumerate() {
                let points = &instance.points;
                // map each scalar to a ScalarField
                let scalars = &instance.scalars;

                let affine_points: Vec<_> = cfg_into_iter!(points).map(|p| p.to_affine()).collect();
                // let arkworks_msm = ArkG::msm(&affine_points, &scalars[..]).unwrap();
                let h2c_msm = halo2curves::msm::msm_best(&scalars[..], &affine_points[..]);

                let metal_msm = metal_msm::<H2G, H2Fr>(&points[..], &scalars[..], &mut metal_config).unwrap();
                assert_eq!(metal_msm.to_affine(), h2c_msm.to_affine(), "This msm is wrongly computed");
                log::info!(
                    "(pass) {}th instance of size 2^{} is correctly computed",
                    i, LOG_INSTANCE_SIZE
                );
            }
        }


        #[test]
        #[ignore]
        fn test_run_multi_benchmarks() {
            init_logger();

            let output_path = format!(
                "{}/{}/{}_h2c_benchmark.txt",
                env::var("CARGO_MANIFEST_DIR").unwrap(),
                &BENCHMARKSPATH,
                "metal_msm"
            );
            let mut output_file = File::create(output_path).expect("output file creation failed");
            writeln!(output_file, "msm_size,num_msm,avg_processing_time(ms)").unwrap();

            let log_instance_sizes: Vec<u32> = vec![8, 12, 16, 18, 20, 22];
            let num_instance: Vec<u32> = vec![10];

            for log_size in log_instance_sizes {
                for num in &num_instance {
                    let result = run_benchmark::<H2G, H2Fr>(log_size, *num, None).unwrap();
                    log::info!("{}x{} result: {:#?}", log_size, *num, result);
                    writeln!(
                        output_file,
                        "{},{},{}",
                        result.instance_size, result.num_instance, result.avg_processing_time
                    )
                    .unwrap();
                }
            }
        }
    }

}
