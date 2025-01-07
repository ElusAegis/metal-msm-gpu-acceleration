use std::sync::{Arc, Condvar, Mutex};
use rayon::prelude::ParallelSliceMut;
use crate::metal::msm::{MetalMsmConfig, MetalMsmInstance};

lazy_static::lazy_static! {
    /// Used to notify other processes that the Sort stage has finished in GPU MSM computation.
    /// Sorting is CPU intensive and we want to avoid CPU contention with other processes.
    pub static ref CPU_SORT_FINISHED: Arc<(Mutex<bool>, Condvar)> = Arc::new((Mutex::new(false), Condvar::new()));
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
pub fn sort_buckets_indices(
    _config: &MetalMsmConfig,
    instance: &MetalMsmInstance,
) {
    // Retrieve the raw metal::Buffer
    let buffer = &instance.data.buckets_indices_buffer;

    // Cast its contents to *mut u32
    let ptr = buffer.contents() as *mut u32;
    // Calculate how many u32s we have in total
    let total_u32s = buffer.length() as usize / size_of::<u32>();

    // Interpret the slice as pairs of (u32, u32) or [u32; 2]
    let pair_count = total_u32s / 2;
    let pair_ptr = ptr as *mut [u32; 2];
    let pair_slice = unsafe { std::slice::from_raw_parts_mut(pair_ptr, pair_count) };

    // Sort in-place by the first element of each pair
    pair_slice.par_sort_by_key(|pair| pair[0]);

    // At this point, 'pair_slice' is sorted in place in GPU-shared memory.
    // Hence, we can stop here.


    // Unlock the static value and notify the CPU thread
    let (lock, cvar) = &*CPU_SORT_FINISHED.clone();
    let mut started = lock.lock().unwrap();
    *started = true; // Unlock the value
    cvar.notify_one(); // Notify the waiting CPU thread
}


/// Creates a MetalMsmInstance with the given (u32) data in the buckets_indices_buffer.
/// This is a mock/truncated approach. Adjust as needed to fit your actual code.
pub fn create_test_instance(
    config: &mut MetalMsmConfig,
    data: Vec<u32>,
) -> MetalMsmInstance {
    // We'll create a minimal MetalMsmInstance with only the buckets_indices_buffer set
    use crate::metal::msm::{MetalMsmParams, MetalMsmData};

    let length = data.len() / 2;
    let buckets_indices_buffer = config.state.alloc_buffer_data(&data);

    let instance_data = MetalMsmData {
        window_size_buffer: config.state.alloc_buffer_data(&[0]),
        instances_size_buffer: config.state.alloc_buffer_data(&[length as u32]),
        window_starts_buffer: config.state.alloc_buffer_data(&[0]),
        scalar_buffer: config.state.alloc_buffer::<u32>(0),
        base_buffer: config.state.alloc_buffer::<u32>(0),
        window_num_buffer: config.state.alloc_buffer_data(&[1u32]),
        buckets_matrix_buffer: config.state.alloc_buffer::<u32>(0),
        buckets_indices_buffer,
        res_buffer: config.state.alloc_buffer::<u32>(0),
        result_buffer: config.state.alloc_buffer::<u32>(0),
    };

    let instance_params = MetalMsmParams {
        instances_size: length as u32,
        buckets_size: 0,
        window_size: 0,
        window_num: 1,
    };

    MetalMsmInstance {
        data: instance_data,
        params: instance_params,
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::msm::setup_metal_state;
    use std::collections::HashSet;
    use proptest::prelude::*;
    use rand::SeedableRng;
    use crate::metal::abstraction::state::MetalState;
    use crate::metal::tests::init_logger;

    /// We define a small GPU wrapper for testing
    fn sort_on_gpu(config: &MetalMsmConfig, instance: &MetalMsmInstance) -> Vec<u32> {
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
        let data = vec![
            15, 2,
            10, 1,
            500, 3,
            0, 0,
            10, 2,
            10, 3,
        ];

        // Build a mock instance
        let instance = create_test_instance(&mut config, data.clone());

        // GPU sort
        let gpu_sorted_flat = sort_on_gpu(&config, &instance);

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
            let mut data = Vec::with_capacity(length * 2);
            (0..length).for_each(|_| {
                data.push(rng.gen::<u32>() % (length as u32 * 16));
                data.push(rng.gen::<u32>() % length as u32) ;
            });

            // Build a mock instance
            let instance = create_test_instance(&mut config, data.clone());

            // GPU sort
            let gpu_sorted_flat = sort_on_gpu(&config, &instance);

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
}