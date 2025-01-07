//! This module dispatches a Metal kernel (`sum_reduction`) that processes each
//! window's buckets to compute a final window sum:
//!
//! res[j] = \sum_{b=0 to buckets_len-1} ( (b+1)*buckets_matrix[j*buckets_len + b] )
//!
//! The kernel logic is parallelized across threads in a group; each group handles
//! exactly one window. Then partial sums are reduced in threadgroup shared memory.

use instant::Instant;
use metal::{MTLSize};
use objc::rc::autoreleasepool;
use crate::metal::msm::{MetalMsmConfig, MetalMsmInstance};

fn sum_reduction_partial(
    config: &MetalMsmConfig,
    instance: &MetalMsmInstance,
    groups_per_window: u32,
) -> Option<metal::Buffer> {
    let data = &instance.data;
    let params = &instance.params;

    let num_windows = params.window_num;
    if num_windows == 0 {
        return None;
    }
    // total threadgroups = num_windows * groups_per_window
    let total_threadgroups = (num_windows as u64) * (groups_per_window as u64);
    if total_threadgroups == 0 {
        return None;
    }

    // Decide how many threads per group (similar logic to your original).
    // We'll clamp by pipeline max, etc.
    let max_threads = config
        .pipelines
        .sum_reduction_partial
        .max_total_threads_per_threadgroup()
        .min(128); // Some heuristic
    // For example:
    let threads_per_group = max_threads;

    // Prepare small buffers for the kernel’s constants
    let buckets_size_buffer  = config.state.alloc_buffer_data(&[params.buckets_size]);
    let window_num_buffer    = config.state.alloc_buffer_data(&[num_windows]);
    let groups_per_window_buffer = config.state.alloc_buffer_data(&[groups_per_window]);

    // Prepare the partial_buffer (device) sized to total_threadgroups
    let partial_result_count = total_threadgroups as usize;
    let partial_result_u32_size = 49; // 49 = 3 * 8 + 3 * 8 + 1 (SerBn254Point, SerBn254Point, u32)
    let partial_results_buffer = config.state.alloc_buffer::<u32>(partial_result_count * partial_result_u32_size);

    // MTL thread dispatch
    let mtl_threadgroups = MTLSize::new(total_threadgroups, 1, 1);
    let mtl_threads_per_group = MTLSize::new(threads_per_group, 1, 1);

    autoreleasepool(|| {
        let (cmd_buffer, cmd_encoder) = config.state.setup_command(
            &config.pipelines.sum_reduction_partial,
            Some(&[
                (0, &buckets_size_buffer),
                (1, &window_num_buffer),
                (2, &groups_per_window_buffer),
                (3, &data.buckets_matrix_buffer),
                (4, &partial_results_buffer),
            ]),
        );

        // We must set threadgroup memory lengths for the arrays: sums, sos, counts.
        // sums and sos each hold `threads_per_group` SerBn254Points:
        let shared_memory_length =  (partial_result_u32_size * size_of::<u32>()) as u64 * (threads_per_group / 2);

        cmd_encoder.set_threadgroup_memory_length(0, shared_memory_length);

        // Dispatch
        cmd_encoder.dispatch_thread_groups(mtl_threadgroups, mtl_threads_per_group);
        cmd_encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    });

    Some(partial_results_buffer)
}

pub fn sum_reduction_final(
    config: &MetalMsmConfig,
    instance: &MetalMsmInstance,
    partial_results_buffer: &metal::Buffer,
    groups_per_window: u32,
) {
    let data = &instance.data;
    let params = &instance.params;
    let num_windows = params.window_num;
    if num_windows == 0 {
        return;
    }

    // We'll have exactly one threadgroup per window => total of num_windows.
    let total_threadgroups = num_windows as u64;

    // Decide how many threads to launch per group. 
    // We need at least enough threads to cover 'groups_per_window' partials.
    let max_threads = config
        .pipelines
        .sum_reduction_final
        .max_total_threads_per_threadgroup()
        .min(32); // Some heuristic
    // Suppose we do:
    let threads_per_group = max_threads.min(groups_per_window as u64).max(1);

    // Prepare small buffers:
    let num_windows_buffer       = config.state.alloc_buffer_data(&[num_windows]);
    let groups_per_window_buffer = config.state.alloc_buffer_data(&[groups_per_window]);

    // MTLSize
    let mtl_threadgroups      = MTLSize::new(total_threadgroups, 1, 1);
    let mtl_threads_per_group = MTLSize::new(threads_per_group, 1, 1);

    autoreleasepool(|| {
        let (cmd_buffer, cmd_encoder) = config.state.setup_command(
            &config.pipelines.sum_reduction_final,
            Some(&[
                (0, &num_windows_buffer),
                (1, &groups_per_window_buffer),
                (2, &partial_results_buffer), // partial results from stage1
                (3, &data.res_buffer),
            ]),
        );

        // As in partial kernel, set threadgroup memory:
        let partial_result_u32_size = 49; // 49 = 3 * 8 + 3 * 8 + 1 (SerBn254Point, SerBn254Point, u32)
        let shared_memory_length =  (partial_result_u32_size * size_of::<u32>()) as u64 * threads_per_group;
        cmd_encoder.set_threadgroup_memory_length(0, shared_memory_length);

        // Dispatch
        cmd_encoder.dispatch_thread_groups(mtl_threadgroups, mtl_threads_per_group);
        cmd_encoder.end_encoding();
        cmd_buffer.commit();
        cmd_buffer.wait_until_completed();
    });
}

/// Dispatches the `sum_reduction` Metal shader kernel.
///
/// # Arguments
///
/// - `config`: GPU configuration/pipelines
/// - `instance`: Must contain:
///    - `window_size_buffer` with `window_size` (c).
///    - `scalar_buffer`: (k_buff) if used
///    - `base_buffer`: (p_buff) if used
///    - `buckets_matrix_buffer`: the matrix of buckets to be reduced
///    - `res_buffer`: final results array, length = `window_size`
///
/// # Flow
/// - We have `params.num_window` windows => #threadgroups = `params.num_window`.
/// - We choose `threads_per_group` so that each group can sum `buckets_len = (1<<window_size)-1` buckets in parallel.
pub fn sum_reduction(
    config: &MetalMsmConfig,
    instance: &MetalMsmInstance,
) {
    let buckets_per_threadgroup = 4096 * 4; // some heuristic
    let groups_per_window = (instance.params.buckets_size + 1).div_ceil(buckets_per_threadgroup);

    // Stage 1: Partial reduction
    let partial_reduction_start = Instant::now();
    let partial_results_buffer = sum_reduction_partial(config, instance, groups_per_window)
        .expect("sum_reduction_partial failed");
    log::debug!("(reduction) Partial reduction took {:?}", partial_reduction_start.elapsed());

    // Stage 2: Final reduction
    let final_reduction_start = Instant::now();
    sum_reduction_final(config, instance, &partial_results_buffer, groups_per_window);
    log::debug!("(reduction) Final reduction took {:?}", final_reduction_start.elapsed());
}

#[cfg(all(test, feature="ark"))] // FIXME - make the tests also work when h2c feature is active
mod tests {
    use std::ops::{Add, Mul};
    use ark_ec::CurveGroup;
    use ark_std::UniformRand;
    use proptest::prelude::any;
    use proptest::{prop_assert_eq, proptest};
    use rand::SeedableRng;
    use super::*;
    use crate::metal::abstraction::limbs_conversion::ark::{ArkFr, ArkG, ArkGAffine};
    use crate::metal::abstraction::limbs_conversion::{FromLimbs, ToLimbs};
    use crate::metal::abstraction::state::MetalState;
    use crate::metal::msm::setup_metal_state;
    use crate::metal::tests::init_logger;

    fn reduce_on_gpu(
        config: &MetalMsmConfig,
        instance: &MetalMsmInstance,
    ) -> Vec<ArkGAffine> {
        sum_reduction(config, instance);

        // Retrieve the results
        let raw_limbs = MetalState::retrieve_contents::<u32>(&instance.data.res_buffer);

        raw_limbs.chunks_exact(24).map(|limbs| ArkG::from_u32_limbs(limbs).into_affine()).collect::<Vec<_>>()
    }

    fn create_test_instance(
        config: &MetalMsmConfig,
        buckets_matrix: &[ArkG],
        window_num: usize,
    ) -> MetalMsmInstance {
        // We'll create a minimal MetalMsmInstance with only the buckets_indices_buffer set
        use crate::metal::msm::{MetalMsmParams, MetalMsmData};

        // Make sure that the buckets_matrix length is a multiple of the number of windows
        assert_eq!(buckets_matrix.len() % window_num, 0,
                   "buckets_matrix length ({}) is not a multiple of num_windows ({})",
                   buckets_matrix.len(), window_num);

        let buckets_size = buckets_matrix.len() / window_num;

        let buckets_matrix_limbs = buckets_matrix.iter().flat_map(|p| p.to_u32_limbs()).collect::<Vec<_>>();

        let instance_data = MetalMsmData {
            window_size_buffer: config.state.alloc_buffer_data(&[0]),
            window_num_buffer: config.state.alloc_buffer_data(&[window_num]),
            instances_size_buffer: config.state.alloc_buffer_data(&[0]),
            window_starts_buffer: config.state.alloc_buffer_data(&[0]),
            scalar_buffer: config.state.alloc_buffer::<u32>(0),
            base_buffer: config.state.alloc_buffer_data(&[0]),
            buckets_matrix_buffer: config.state.alloc_buffer_data::<u32>(&buckets_matrix_limbs),
            buckets_indices_buffer: config.state.alloc_buffer::<u32>(0),
            res_buffer: config.state.alloc_buffer::<u32>(window_num * 24),
            result_buffer: config.state.alloc_buffer::<u32>(0),
        };

        let instance_params = MetalMsmParams {
            instances_size: 0,
            buckets_size: buckets_size as u32,
            window_size: 0,
            window_num: window_num as u32,
        };

        MetalMsmInstance {
            data: instance_data,
            params: instance_params,
        }
    }

    #[test]
    fn test_sum_reduction_small() {
        init_logger();

        // 1) We'll define a small scenario: window_size=3 => buckets_len=7
        let window_num = 1;
        let buckets_size = 3;

        // Setup Metal state
        let config = setup_metal_state();

        let mut rng = rand::thread_rng();

        let mut buckets_matrix = Vec::with_capacity(window_num * buckets_size);
        for _ in 0..window_num {
            for _ in 0..buckets_size {
                buckets_matrix.push(ArkG::rand(&mut rng));
            }
        }

        // buckets_matrix[2] = ArkG::default();

        log::debug!("Buckets matrix: {:?}", buckets_matrix);

        // Create test instance and sorted indices buffer
        let instance =
            create_test_instance(&config, &buckets_matrix, window_num);


        // We'll do the naive approach
        let rust_res = sum_reduction_rust(
            window_num,
            &buckets_matrix
        );

        // Calculate on GPU
        let gpu_res = reduce_on_gpu(&config, &instance);

        // Compare

        log::debug!("GPU: {:?}", gpu_res);
        log::debug!("Rust: {:?}", rust_res);

        // Check that the length of the results is as expected
        assert_eq!(gpu_res.len(), window_num);
        assert_eq!(rust_res.len(), window_num);

        // Compare the results
        for j in 0..window_num as usize {
            assert_eq!(gpu_res[j], rust_res[j],
                "Mismatch in window j={}: GPU={:?}, Rust={:?}", j, gpu_res[j], rust_res[j]);
        }
    }


    proptest! {
        #[test]
        fn test_sum_reduction_large(
            seed in any::<u64>(),
            window_num in 1usize..20,
            log_bucket_size in 1u32..10,
            bucket_size_mul in 1u32..8,
        ) {
            init_logger();

            log::debug!("Log instance size: {}, bucket suze mul: {}, window size: {}", log_bucket_size, bucket_size_mul, window_num);

            // Setup Metal state
            let config = setup_metal_state();

            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

            let buckets_size = (1 << log_bucket_size) * bucket_size_mul as usize;
            let mut buckets_matrix = Vec::with_capacity(window_num * buckets_size);
            for _ in 0..window_num {
                for _ in 0..buckets_size {
                    buckets_matrix.push(ArkG::rand(&mut rng));
                }
            }

            // Create test instance and sorted indices buffer
            let instance =
                create_test_instance(&config, &buckets_matrix, window_num);


            // We'll do the naive approach
            let rust_res = sum_reduction_rust(
                window_num,
                &buckets_matrix
            );

            // Calculate on GPU
            let gpu_res = reduce_on_gpu(&config, &instance);

            // Compare GPU results with Rust implementation
            prop_assert_eq!(rust_res, gpu_res);

            log::debug!("Test passed for window_num={}, bucket_size={}, bucket_size_mul={}", window_num, buckets_size, bucket_size_mul);
        }
    }

    pub fn sum_reduction_rust(
        num_windows: usize,
        buckets_matrix: &[ArkG],
    ) -> Vec<ArkGAffine> {
        let mut res = vec![ArkG::default(); num_windows];
        let buckets_size = buckets_matrix.len() / num_windows;
        for j in 0..num_windows {
            // do a local sum
            let mut local_sum = ArkG::default();
            let base_idx = j * buckets_size;

            for b in 0..buckets_size {
                let val = buckets_matrix[base_idx + b];
                // weight by (b+1)? If your kernel is adding (b+1)*val, do so here:
                let weighted = val.mul(ArkFr::from((b+1) as u64));
                // or simply do `local_sum = local_sum + val;`
                local_sum = local_sum.add(&weighted);
            }
            // add to res[j]
            res[j] = res[j].add(&local_sum);
        }

        res.iter().map(|p| p.into_affine()).collect()
    }

}