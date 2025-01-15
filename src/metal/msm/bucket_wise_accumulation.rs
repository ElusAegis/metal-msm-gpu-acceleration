//! bucket_wise_accumulation.rs
//!
//! This module dispatches a Metal kernel (`bucket_wise_accumulation`) that sums up
//! GPU points in each bucket, based on sorted bucket indices. It also includes
//! a pure Rust implementation of the accumulation step for validation.

use crate::metal::msm::{MetalMsmConfig, MetalMsmInstance};
use metal::MTLSize;
use objc::rc::autoreleasepool;
use crate::config::ConfigManager;

/// Dispatches the `bucket_wise_accumulation` Metal shader kernel.
/// This kernel reads:
/// - `buckets_indices`: array of `(bucket_index, point_index)` sorted ascending by `bucket_index`
/// - `p_buff`: array of GPU points (`Point`) of length = number of input points
/// and accumulates all points for each unique `bucket_index` into the `buckets_matrix` array
/// so that `buckets_matrix[bucket_index]` is the sum of all points mapping to `b`.
///
/// # Arguments
///
/// * `config` - The Metal MSM configuration with pipelines.
/// * `instance` - The MSM instance containing data buffers and parameters.
///
/// # Returns
///
/// * `()` - The function modifies the `buckets_matrix_buffer` in-place, storing the sums.
pub fn bucket_wise_accumulation(config: &MetalMsmConfig, instance: &MetalMsmInstance) {
    let data = &instance.data;
    let params = &instance.params;

    // total buckets
    let total_buckets = (params.buckets_size as u64) * (params.window_num as u64);
    if total_buckets == 0 {
        log::debug!("No buckets to accumulate. Returning.");
        return;
    }

    // For safety, clamp the max GPU threads
    let desired_pairs_per_thread = ConfigManager::default().desired_pairs_per_thread() as u64;
    let actual_threads =
        ((params.instances_size * params.window_num) as u64 + desired_pairs_per_thread - 1)
            / desired_pairs_per_thread;
    // Or choose some other logic if you'd prefer a smaller # of threads

    // Threads per group
    let threads_per_group = config
        .pipelines
        .bucket_wise_accumulation
        .max_total_threads_per_threadgroup()
        .min(64) // The maximum number of threads per group due to shared memory limit
        .min(actual_threads);

    // # of threadgroups
    let num_thread_groups = (actual_threads + threads_per_group - 1) / threads_per_group;

    let mtl_threadgroups = MTLSize::new(num_thread_groups, 1, 1);
    let mtl_threads_per_group = MTLSize::new(threads_per_group, 1, 1);

    // GPU buffers for parameters
    let actual_threads_buffer = config.state.alloc_buffer_data(&[actual_threads as u32]);
    let total_buckets_buffer = config.state.alloc_buffer_data(&[total_buckets as u32]);

    // // Print information about the kernel
    // println!(
    //     "Launching bucket_wise_accumulation kernel with {} threadgroups of {} threads each",
    //     num_thread_groups,
    //     threads_per_group
    // );
    // println!(
    //     "total_buckets={}, actual_threads={}",
    //     total_buckets, actual_threads
    // );

    // Launch kernel
    autoreleasepool(|| {
        let (command_buffer, command_encoder) = config.state.setup_command(
            &config.pipelines.bucket_wise_accumulation,
            Some(&[
                (0, &data.instances_size_buffer),  // _instances_size
                (1, &data.window_num_buffer),      // _num_windows
                (2, &data.base_buffer),            // p_buff
                (3, &data.buckets_indices_buffer), // sorted (x,y)
                (4, &data.buckets_matrix_buffer),  // output sums
                (5, &actual_threads_buffer),       // _actual_threads
                (6, &total_buckets_buffer),        // _total_buckets
            ]),
        );

        // We now need TWO allocations in threadgroup memory:
        //   1) sharedLeftAccum
        //   2) sharedRightAccum
        //
        // Each entry is a struct of size:
        //     sizeof(uint) + sizeof(Point)
        // Scalar is stored in 8 u32s, Point is represented as 3 scalars, which is 24 u32s:
        let pair_accum_size = 32 * size_of::<u32>() as u64;
        let left_accum_size = threads_per_group * pair_accum_size;
        let right_accum_size = threads_per_group * pair_accum_size;

        command_encoder.set_threadgroup_memory_length(0, left_accum_size);
        command_encoder.set_threadgroup_memory_length(1, right_accum_size);

        command_encoder.dispatch_thread_groups(mtl_threadgroups, mtl_threads_per_group);
        command_encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();
    });
}

#[cfg(all(test, feature = "ark"))]
mod tests {
    use super::*;
    use crate::metal::abstraction::limbs_conversion::ark::ArkGAffine;
    use crate::metal::abstraction::limbs_conversion::{ark::ArkG, FromLimbs, ToLimbs};
    use crate::metal::abstraction::state::MetalState;
    use crate::metal::msm::{setup_metal_state, MetalMsmConfig, MetalMsmInstance};
    use crate::metal::tests::init_logger;
    use ark_ec::CurveGroup;
    use ark_std::UniformRand;
    use proptest::prelude::*;
    use rand::{Rng, SeedableRng};
    use std::ops::Add;

    /// Struct to hold individual test case data
    struct BucketAccumTestCase {
        name: &'static str,
        buckets_indices: Vec<(u32, u32)>, // (bucket_idx, point_idx)
    }

    /// We define a small GPU wrapper for testing
    fn accumulate_on_gpu(config: &MetalMsmConfig, instance: &MetalMsmInstance) -> Vec<ArkGAffine> {
        // Sort on the GPU
        bucket_wise_accumulation(config, instance);

        // Read results back from GPU
        let raw_limbs = MetalState::retrieve_contents::<u32>(&instance.data.buckets_matrix_buffer);

        // let scalar_groups = raw_limbs.chunks(8).collect::<Vec<_>>();
        // let point_groups = scalar_groups.chunks(3).collect::<Vec<_>>();
        // for (i, group) in point_groups.iter().enumerate() {
        //     if *group != [[0; 8]; 3] {
        //         println!("Group {}: {:?}", i, group);
        //     }
        //
        // }
        // println!("GPU raw_limbs: {:?}", raw_limbs.chunks(8).collect::<Vec<_>>().chunks(3).collect::<Vec<_>>());

        // parse the raw_limbs into a Vec<ArkG>
        raw_limbs
            .chunks_exact(24)
            .map(|limbs| ArkG::from_u32_limbs(limbs).into_affine())
            .collect::<Vec<_>>()
    }

    fn create_test_instance(
        config: &MetalMsmConfig,
        buckets_indices: &[(u32, u32)],
        points: &[ArkG],
    ) -> MetalMsmInstance {
        // We'll create a minimal MetalMsmInstance with only the buckets_indices_buffer set
        use crate::metal::msm::{MetalMsmData, MetalMsmParams};

        // Make sure that the buckets_indices are sorted by bucket index
        let mut buckets_indices = buckets_indices.to_vec();
        buckets_indices.sort_by_key(|(a, _)| *a);

        // Make sure that the length of buckets_indices is a multiple of points length
        // This multiple is the number of windows
        assert_eq!(
            buckets_indices.len() % points.len(),
            0,
            "buckets_indices length ({}) is not a multiple of points length ({})",
            buckets_indices.len(),
            points.len()
        );

        let num_windows = buckets_indices.len() / points.len();

        // The amount of buckets is the max(bucket_idx) + 1
        let total_bucket_amount = buckets_indices.iter().map(|(a, _)| *a).max().unwrap() + 1;
        let total_bucket_amount = (total_bucket_amount - 1)
            - (total_bucket_amount - 1) % num_windows as u32
            + num_windows as u32;

        assert_eq!(
            total_bucket_amount % num_windows as u32,
            0,
            "total_bucket_amount ({}) is not a multiple of num_windows ({})",
            total_bucket_amount,
            num_windows
        );

        let instance_size = points.len();

        // Fill bases_limbs using write_u32_limbs in parallel
        let bases_limbs: Vec<u32> = points.iter().map(|p| p.to_u32_limbs()).flatten().collect();

        let instance_data = MetalMsmData {
            window_size_buffer: config.state.alloc_buffer_data(&[0]),
            window_num_buffer: config.state.alloc_buffer_data(&[num_windows]),
            instances_size_buffer: config.state.alloc_buffer_data(&[instance_size as u32]),
            window_starts_buffer: config.state.alloc_buffer_data(&[0]),
            scalar_buffer: config.state.alloc_buffer::<u32>(0),
            base_buffer: config.state.alloc_buffer_data(&bases_limbs),
            buckets_matrix_buffer: config
                .state
                .alloc_buffer::<u32>((total_bucket_amount * 8 * 3) as usize),
            buckets_indices_buffer: config.state.alloc_buffer_data(&buckets_indices),
            res_buffer: config.state.alloc_buffer::<u32>(0),
            result_buffer: config.state.alloc_buffer::<u32>(0),
        };

        let instance_params = MetalMsmParams {
            instances_size: instance_size as u32,
            // We do this because the total_bucket_amount is bucket_size * num_window
            buckets_size: total_bucket_amount / num_windows as u32,
            window_size: 0,
            window_num: num_windows as u32,
        };

        MetalMsmInstance {
            data: instance_data,
            params: instance_params,
        }
    }

    #[test]
    fn test_bucket_wise_accumulation_small() {
        init_logger();

        // Define multiple test cases
        // Define multiple test cases
        let test_cases = vec![
            // 1. Simple Indices
            BucketAccumTestCase {
                name: "Simple Indices",
                buckets_indices: vec![
                    (0, 0), // bucket 0 => point0
                    (1, 1), // bucket 1 => point1
                    (2, 2), // bucket 2 => point2
                    (3, 3), // bucket 3 => point3
                ],
            },
            // 2. Double Indices
            BucketAccumTestCase {
                name: "Double Indices",
                buckets_indices: vec![
                    (0, 0), // bucket 0 => point0
                    (0, 0), // bucket 0 => point0
                    (1, 1), // bucket 1 => point1
                    (1, 1), // bucket 1 => point1
                    (2, 2), // bucket 2 => point2
                    (2, 2), // bucket 2 => point2
                    (3, 3), // bucket 3 => point3
                    (3, 3), // bucket 3 => point3
                ],
            },
            // 3. Two Equal Buckets Indices
            BucketAccumTestCase {
                name: "Two Equal Buckets Indices",
                buckets_indices: vec![
                    (1, 0), // bucket 1 => point0
                    (1, 0), // bucket 1 => point0
                    (1, 0), // bucket 1 => point0
                    (1, 1), // bucket 1 => point1
                    (2, 1), // bucket 2 => point1
                    (2, 2), // bucket 2 => point2
                    (2, 1), // bucket 2 => point1
                    (2, 3), // bucket 2 => point1
                ],
            },
            // 4. Single Bucket
            BucketAccumTestCase {
                name: "Single Bucket",
                buckets_indices: vec![
                    (0, 0), // bucket 0 => point0
                    (0, 1), // bucket 0 => point1
                    (0, 2), // bucket 0 => point2
                    (0, 3), // bucket 0 => point3
                ],
            },
            // 5. Buckets with Gaps
            BucketAccumTestCase {
                name: "Buckets with Gaps",
                buckets_indices: vec![
                    (0, 0), // bucket 0 => point0
                    (2, 1), // bucket 2 => point1
                    (4, 2), // bucket 4 => point2
                    (6, 3), // bucket 6 => point3
                ],
            },
            // 6. Non-Contiguous Indices
            BucketAccumTestCase {
                name: "Non-Contiguous Indices",
                buckets_indices: vec![
                    (1, 0), // bucket 1 => point0
                    (1, 1), // bucket 1 => point1
                    (2, 2), // bucket 2 => point2
                    (3, 3), // bucket 3 => point3
                    (5, 4), // bucket 5 => point4
                ],
            },
            // 7. Varying Frequencies
            BucketAccumTestCase {
                name: "Varying Frequencies",
                buckets_indices: vec![
                    (0, 0), // bucket 0 => point0
                    (0, 1), // bucket 0 => point1
                    (0, 2), // bucket 0 => point2
                    (1, 3), // bucket 1 => point3
                    (2, 4), // bucket 2 => point4
                    (2, 5), // bucket 2 => point5
                    (2, 6), // bucket 2 => point6
                    (2, 7), // bucket 2 => point7
                ],
            },
            // 8. Maximum Bucket Index Large
            BucketAccumTestCase {
                name: "Max Bucket Index Large",
                buckets_indices: vec![
                    (0, 0),  // bucket 0 => point0
                    (10, 1), // bucket 10 => point1
                    (10, 2), // bucket 10 => point2
                    (20, 3), // bucket 20 => point3
                ],
            },
            // 9. Minimal Bucket Indices Starting High
            BucketAccumTestCase {
                name: "Minimal Bucket Indices Starting High",
                buckets_indices: vec![
                    (5, 0), // bucket 5 => point0
                    (5, 1), // bucket 5 => point1
                    (6, 2), // bucket 6 => point2
                    (7, 3), // bucket 7 => point3
                ],
            },
            // // 10. Empty Bucket Indices
            // BucketAccumTestCase {
            //     name: "Empty Bucket Indices",
            //     buckets_indices: vec![],
            // },
            // 11. Single Point
            BucketAccumTestCase {
                name: "Single Point",
                buckets_indices: vec![
                    (0, 0), // bucket 0 => point0
                ],
            },
            // 12. Interleaved Frequencies
            BucketAccumTestCase {
                name: "Interleaved Frequencies",
                buckets_indices: vec![
                    (0, 0), // bucket 0 => point0
                    (1, 1), // bucket 1 => point1
                    (1, 2), // bucket 1 => point2
                    (2, 3), // bucket 2 => point3
                    (2, 4), // bucket 2 => point4
                    (3, 5), // bucket 3 => point5
                    (3, 6), // bucket 3 => point6
                    (3, 7), // bucket 3 => point7
                ],
            },
            // 13. Non-Consecutive Buckets Varying Frequencies
            BucketAccumTestCase {
                name: "Non-Consecutive Buckets Varying Frequencies",
                buckets_indices: vec![
                    (0, 0), // bucket 0 => point0
                    (2, 1), // bucket 2 => point1
                    (2, 2), // bucket 2 => point2
                    (4, 3), // bucket 4 => point3
                    (4, 4), // bucket 4 => point4
                    (4, 5), // bucket 4 => point5
                    (6, 6), // bucket 6 => point6
                ],
            },
            // 14. Buckets Missing Except Some
            BucketAccumTestCase {
                name: "Buckets Missing Except Some",
                buckets_indices: vec![
                    (1, 0), // bucket 1 => point0
                    (1, 1), // bucket 1 => point1
                    (3, 2), // bucket 3 => point2
                    (5, 3), // bucket 5 => point3
                    (5, 4), // bucket 5 => point4
                ],
            },
            // 15. Failing Instance
            BucketAccumTestCase {
                name: "Failing Instance from Large Test #1",
                buckets_indices: vec![
                    (0, 7),
                    (1, 4),
                    (2, 0),
                    (2, 6),
                    (3, 5),
                    (5, 1),
                    (7, 2),
                    (8, 3),
                ],
            },
            // 16. Failing Instance from Large Test
            BucketAccumTestCase {
                name: "Failing Instance from Large Test #2",
                buckets_indices: vec![
                    (6, 6),
                    (11, 1),
                    (12, 6),
                    (14, 4),
                    (20, 3),
                    (21, 0),
                    (21, 0),
                    (27, 4),
                    (27, 2),
                    (38, 4),
                    (50, 5),
                    (51, 4),
                    (54, 1),
                    (55, 2),
                    (55, 2),
                    (67, 6),
                    (73, 1),
                    (80, 2),
                    (90, 4),
                    (90, 8),
                    (92, 6),
                    (97, 2),
                    (101, 8),
                    (103, 2),
                    (110, 2),
                    (113, 1),
                    (115, 0),
                ],
            },
            // 17. Failing Instance from Large Test
            BucketAccumTestCase {
                name: "Failing Instance from Large Test #3",
                buckets_indices: vec![
                    (6, 6),
                    (11, 1),
                    (12, 6),
                    (14, 4),
                    (20, 3),
                    (21, 0),
                    (21, 0),
                    (27, 4),
                    (27, 2),
                    (38, 4),
                    (50, 5),
                    (51, 4),
                    (54, 1),
                    (55, 2),
                    (55, 2),
                    (67, 6),
                    (73, 1),
                    (80, 2),
                    (90, 4),
                    (90, 7),
                    (92, 6),
                    (97, 2),
                    (101, 7),
                    (103, 2),
                    (110, 2),
                    (113, 1),
                    (115, 0),
                    (151, 2),
                    (152, 7),
                    (156, 3),
                    (162, 7),
                    (171, 2),
                    (172, 6),
                    (178, 2),
                    (190, 5),
                    (190, 5),
                    (190, 4),
                    (190, 5),
                    (190, 6),
                    (190, 7),
                    (254, 0),
                    (254, 1),
                    (254, 6),
                    (254, 2),
                    (254, 3),
                    (254, 4),
                    (254, 5),
                    (255, 6),
                ],
            },
        ];

        // Iterate over each test case
        for test_case in test_cases {
            // Setup Metal state
            let config = setup_metal_state();

            let mut rand = rand::thread_rng();
            let point_amount = test_case
                .buckets_indices
                .iter()
                .map(|(_, idx)| *idx)
                .max()
                .unwrap_or(0) as usize
                + 1;
            let points: Vec<ArkG> = (0..point_amount).map(|_| ArkG::rand(&mut rand)).collect();

            // Create test instance and sorted indices buffer
            let instance = create_test_instance(&config, &test_case.buckets_indices, &points);

            // Read the results from the GPU buffer (buckets_matrix)
            let gpu_buckets_matrix = accumulate_on_gpu(&config, &instance);

            // Compute expected results using Rust
            let mut rust_acc = bucket_wise_accumulation_rust(&test_case.buckets_indices, &points);
            rust_acc.resize(
                (instance.params.buckets_size * instance.params.window_num) as usize,
                ArkG::default(),
            );

            // Log the test case name
            log::debug!("\n\nRunning test case: {}", test_case.name);
            log::debug!("Points: {:?}", points);
            log::debug!("GPU buckets_matrix: {:?}", gpu_buckets_matrix);
            log::debug!("Expected (Rust) buckets_matrix: {:?}", rust_acc);

            // Compare the lengths
            assert_eq!(
                gpu_buckets_matrix.len(),
                rust_acc.len(),
                "Test case '{}' - Mismatch in lengths: GPU={}, Rust={}",
                test_case.name,
                gpu_buckets_matrix.len(),
                rust_acc.len()
            );

            // Compare each bucket's result
            for i in 0..rust_acc.len() {
                assert_eq!(
                    gpu_buckets_matrix[i],
                    rust_acc[i].into_affine(),
                    "Test case '{}' - Mismatch in bucket {}: GPU={:?}, Rust={:?}",
                    test_case.name,
                    i,
                    gpu_buckets_matrix[i],
                    rust_acc[i].into_affine()
                );
            }
        }
    }

    // A property-based test for "large" scenarios
    proptest! {
        #[test]
        fn test_bucket_wise_accumulation_large_instance(
            seed in any::<u64>(),
            log_size in 3u32..10,
            num_buckets in 2u32..32,
        ) {
            init_logger();

            // Prepare random data
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let size = 1 << log_size;
            let points: Vec<ArkG> = (0..size).map(|_| ArkG::rand(&mut rng)).collect();

            // We'll create a random sorted list of (bucket_index, point_index)
            // bucket_index in [0..num_buckets)
            // point_index in [0..size)
            let mut buckets_indices = Vec::with_capacity(size);
            for point_idx in 0..size {
                let b_idx = rng.gen_range(0..num_buckets);
                buckets_indices.push((b_idx, point_idx as u32));
            }
            // Sort them by bucket_index
            buckets_indices.sort_by_key(|(b, _)| *b);

            let config = setup_metal_state();
            let instance = create_test_instance(&config, &buckets_indices, &points);

            // GPU
            let gpu_buckets_matrix = accumulate_on_gpu(&config, &instance);

            // Rust
            let mut rust_acc = bucket_wise_accumulation_rust(&buckets_indices, &points);
            rust_acc.resize((instance.params.buckets_size * instance.params.window_num) as usize, ArkG::default());

            // Compare the lengths
            prop_assert_eq!(gpu_buckets_matrix.len(), rust_acc.len(),
                "Mismatch in lengths: GPU={}, Rust={}", gpu_buckets_matrix.len(), rust_acc.len());

            // log::debug!("Comparison started with seed={}, log_size={}, num_buckets={}", seed, log_size, num_buckets);

            // Compare
            for i in 0..gpu_buckets_matrix.len() {
                prop_assert_eq!(gpu_buckets_matrix[i], rust_acc[i],
                    "Mismatch in bucket {}: GPU={:?}, Rust={:?} (seed={}, log_size={}, num_buckets={})",
                    i, gpu_buckets_matrix[i], rust_acc[i], seed, log_size, num_buckets
                );
            }

            log::debug!("Test passed with seed={}, log_size={}, num_buckets={}", seed, log_size, num_buckets);


        }

        #[test]
        #[ignore]
        fn test_bucket_wise_accumulation_large_buckets(
            seed in any::<u64>(),
            log_size in 3u32..9,
            log_num_buckets in 2u32..9,
            num_buckets_mul in 1u32..8,
            num_indices_mul in 1u32..18,
        ) {
            init_logger();

            // Prepare random data
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let size = 1 << log_size;
            let points: Vec<ArkG> = (0..size).map(|_| ArkG::rand(&mut rng)).collect();

            // We'll create a random sorted list of (bucket_index, point_index)
            // bucket_index in [0..num_buckets)
            // point_index in [0..size)
            let mut buckets_indices = Vec::with_capacity(size);
            let num_buckets = 1 << log_num_buckets * num_buckets_mul;
            for _ in 0..(size as u32 * num_indices_mul) {
                let b_idx = rng.gen_range(0..num_buckets);
                let p_idx = rng.gen_range(0..size as u32);
                buckets_indices.push((b_idx, p_idx));
            }
            // Sort them by bucket_index
            buckets_indices.sort_by_key(|(b, _)| *b);

            let config = setup_metal_state();
            let instance = create_test_instance(&config, &buckets_indices, &points);

            // GPU
            let gpu_buckets_matrix = accumulate_on_gpu(&config, &instance);

            // Rust
            let mut rust_acc = bucket_wise_accumulation_rust(&buckets_indices, &points);
            rust_acc.resize((instance.params.buckets_size * instance.params.window_num) as usize, ArkG::default());

            // Compare the lengths
            prop_assert_eq!(gpu_buckets_matrix.len(), rust_acc.len(),
                "Mismatch in lengths: GPU={}, Rust={}", gpu_buckets_matrix.len(), rust_acc.len());

            // log::debug!("Comparison started with seed={}, log_size={}, num_buckets={}", seed, log_size, num_buckets);

            // Compare
            for i in 0..gpu_buckets_matrix.len() {
                prop_assert_eq!(gpu_buckets_matrix[i], rust_acc[i],
                    "Mismatch in bucket {}: GPU={:?}, Rust={:?} (seed={}, log_size={}, num_buckets={})",
                    i, gpu_buckets_matrix[i], rust_acc[i], seed, log_size, num_buckets
                );
            }

            log::debug!("Test passed with seed={}, log_size={}, num_buckets={}", seed, log_size, num_buckets);


        }
    }

    pub fn bucket_wise_accumulation_rust(
        buckets_indices: &[(u32, u32)],
        points: &[ArkG],
    ) -> Vec<ArkG> {
        let total_buckets = buckets_indices.iter().map(|(a, _)| *a).max().unwrap_or(0) as usize + 1;
        let mut result = vec![ArkG::default(); total_buckets];

        // Because `buckets_indices` is sorted by bucket_index, we can do a single pass:
        // or we can do a grouping approach. We'll do a single pass:
        for &(bucket_idx, point_idx) in buckets_indices {
            if bucket_idx == 0xFFFFFFFF {
                continue; // sentinel or empty
            }
            if (bucket_idx as usize) < total_buckets {
                result[bucket_idx as usize] =
                    result[bucket_idx as usize].add(&points[point_idx as usize]);
            }
        }
        result
    }
}
