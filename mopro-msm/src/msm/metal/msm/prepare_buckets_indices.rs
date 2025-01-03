use metal::{MTLSize};
use objc::rc::autoreleasepool;
use super::{MetalMsmConfig, MetalMsmInstance};

/// Executes the `prepare_buckets_indices` Metal shader kernel.
///
/// # Arguments
///
/// * `config` - The Metal MSM configuration containing pipelines and state.
/// * `instance` - The MSM instance containing data and parameters.
///
/// # Returns
///
/// * `()` - The function modifies the `buckets_indices_buffer` within `instance`.
pub(crate) fn prepare_buckets_indices(
    config: &MetalMsmConfig,
    instance: &MetalMsmInstance,
) {
    let data = &instance.data;
    let params = &instance.params;

    autoreleasepool(|| {
        let (command_buffer, command_encoder) = config.state.setup_command(
            &config.pipelines.prepare_buckets_indices,
            Some(&[
                (0, &data.window_size_buffer),
                (1, &data.window_starts_buffer),
                (2, &data.num_windows_buffer),
                (3, &data.scalar_buffer),
                (4, &data.buckets_indices_buffer),
            ]),
        );
        command_encoder.dispatch_thread_groups(
            MTLSize::new(params.instances_size as u64, 1, 1),
            MTLSize::new(1, 1, 1),
        );
        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    });
}

#[cfg(test)]
mod test {
    use std::collections::{HashMap, HashSet};
    use std::sync::Once;
    use ark_ff::{BigInt, BigInteger, PrimeField};
    use ark_std::UniformRand;
    use itertools::Itertools;
    use proptest::prelude::{any};
    use proptest::{prop_assert_eq, prop_assume, proptest};
    use rand::prelude::StdRng;
    use rand::SeedableRng;
    use crate::msm::metal::abstraction::limbs_conversion::ark::{ArkFr, ArkG};
    use crate::msm::metal::abstraction::limbs_conversion::ScalarGPU;
    use crate::msm::metal::abstraction::state::MetalState;
    use crate::msm::metal::msm::{encode_instances, setup_metal_state};
    use crate::msm::metal::msm::prepare_buckets_indices::prepare_buckets_indices;


    // Static initializer to ensure the logger is initialized only once
    static INIT: Once = Once::new();

    fn init_logger() {
        INIT.call_once(|| {
            env_logger::builder()
                .is_test(true) // Ensures logs go to stdout/stderr in a test-friendly way
                .init();
        });
    }

    // Helper function to get a scalar fragment
    // This function copies the logic from the Metal shader UnsignedInteger right shift
    fn get_scalar_fragment<S: ScalarGPU<8>>(scalar: &S, window_start: u32) -> u32 {
        let a = (window_start / 32) as usize; // Number of whole limbs to shift
        let b = window_start % 32;            // Number of bits to shift within a limb

        #[allow(non_snake_case)]
        let NUM_LIMBS: usize = S::N;

        if a >= NUM_LIMBS {
            // Shift exceeds the total number of bits; result is 0
            return 0;
        }

        let scalar_limbs = scalar.to_u32_limbs();

        if b == 0 {
            // Shift by whole limbs only
            return scalar_limbs[NUM_LIMBS - 1 - a];
        } else {
            // Shift by limbs and bits
            // Ensure we don't underflow when accessing limbs
            let high_limb = if NUM_LIMBS > 1 && (NUM_LIMBS - 1 >= a + 1) {
                scalar_limbs[NUM_LIMBS - 1 - a - 1]
            } else {
                0
            };

            let low_limb = scalar_limbs[NUM_LIMBS - 1 - a] >> b;

            // Combine the shifted limbs
            low_limb | (high_limb << (32 - b))
        }
    }

    fn prepare_buckets_indices_rust(
        window_size: u32,
        num_windows: u32,
        scalars: &[ArkFr],
    ) -> HashSet<(u32, u32)> {
        let buckets_len = (1 << window_size) - 1;
        let mut buckets_indices = HashSet::new();

        for (point_idx, scalar) in scalars.iter().enumerate() {

            let window_starts = (0..num_windows).map(|i| i * window_size);

            for (i, window_start) in window_starts.enumerate() {

                let mut scalar_fragment = get_scalar_fragment(scalar, window_start);
                scalar_fragment &= buckets_len;


                if scalar_fragment != 0 {
                    let bucket_idx = i as u32 * buckets_len + scalar_fragment - 1;
                    buckets_indices.insert((bucket_idx, point_idx as u32));
                } else {
                    // Impossible case, as we do not support 32 log size MSM
                    buckets_indices.insert((0xFFFFFFFF, 0xFFFFFFFF));
                }
            }
        }

        buckets_indices
    }

    #[test]
    fn test_prepare_buckets_indices_smaller() {

        init_logger();

        // Setup MetalMsmConfig with mock pipelines
        let mut config = setup_metal_state();

        // Create mock points and scalars
        let size = 2;
        let rng = &mut ark_std::test_rng();
        let points: Vec<ArkG> = (0..size).map(|_| ArkG::rand(rng)).collect();
        let scalars: Vec<ArkFr> = (0..size).map(|_| ArkFr::rand(rng)).collect();
        let mut breaking_scalar = BigInt::one();
        breaking_scalar.muln(14);
        breaking_scalar.add_with_carry(&BigInt::one());
        let scalars: Vec<ArkFr> = vec![ArkFr::from_bigint(breaking_scalar).unwrap()];

        let instance = encode_instances(&points, &scalars, &mut config, Some(14));

        // Execute the GPU kernel
        prepare_buckets_indices(&config, &instance);

        // Retrieve the results from the GPU buffer
        let gpu_buckets_indices_flat = MetalState::retrieve_contents::<u32>(&instance.data.buckets_indices_buffer);

        // Convert flat GPU output to a HashSet of tuples
        let gpu_buckets_indices: HashSet<(u32, u32)> = gpu_buckets_indices_flat
            .chunks(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect();

        // Execute the pure Rust implementation
        let rust_buckets_indices = prepare_buckets_indices_rust(
            instance.params.window_size,
            instance.params.num_window,
            &scalars,
        );

        // TODO - remove this block once the implementation is stable
        {

           // Helper function to create a frequency map
            fn create_frequency_map(buckets: &HashSet<(u32, u32)>) -> HashMap<u32, usize> {
                let mut freq_map = HashMap::new();
                for (bucket, _) in buckets {
                    *freq_map.entry(*bucket).or_insert(0) += 1;
                }
                freq_map
            }

            // Create frequency maps for Rust and GPU results
            let rust_freq_map = create_frequency_map(&rust_buckets_indices);
            let gpu_freq_map = create_frequency_map(&gpu_buckets_indices);

            // Helper function to count the distribution of frequencies
            fn count_frequencies(freq_map: &HashMap<u32, usize>) -> HashMap<usize, usize> {
                let mut count_map = HashMap::new();
                for &count in freq_map.values() {
                    *count_map.entry(count).or_insert(0) += 1;
                }
                count_map
            }

            // Count the distribution of frequencies for Rust and GPU results
            let rust_freq_distribution = count_frequencies(&rust_freq_map);
            let gpu_freq_distribution = count_frequencies(&gpu_freq_map);

            // Print the distribution
            log::debug!("Frequency | Rust Count | GPU Count");
            log::debug!("------------------------------------");

            // Collect all unique frequencies
            let all_frequencies: HashSet<usize> = rust_freq_distribution.keys().cloned().collect::<HashSet<_>>()
                .union(&gpu_freq_distribution.keys().cloned().collect::<HashSet<_>>())
                .cloned()
                .collect();

            // Sort the frequencies for orderly printing
            let mut sorted_frequencies: Vec<usize> = all_frequencies.into_iter().collect();
            sorted_frequencies.sort_unstable();

            // Print the counts for each frequency
            for freq in sorted_frequencies {
                let rust_count = rust_freq_distribution.get(&freq).cloned().unwrap_or(0);
                let gpu_count = gpu_freq_distribution.get(&freq).cloned().unwrap_or(0);
                println!("{:9} | {:10} | {:9}", freq, rust_count, gpu_count);
            }
        }

        log::debug!("Rust Result: {:?}", rust_buckets_indices.iter().sorted());
        log::debug!("GPU Result: {:?}", gpu_buckets_indices.iter().sorted());

        let expected_size = instance.params.num_window * instance.params.instances_size;

        // Check that the number of buckets is as expected
        assert!(gpu_buckets_indices.len() <= expected_size as usize);
        assert_eq!(rust_buckets_indices.len(), gpu_buckets_indices.len());

        // Compare GPU results with Rust implementation
        assert_eq!(gpu_buckets_indices, rust_buckets_indices);
    }


    proptest! {
        #[test]
        fn test_prepare_buckets_indices_large(
            seed in any::<u64>(),
            window_size in 2u32..25,
            log_instance_size in 3u32..16,
        ) {
            init_logger();

            log::debug!("Log instance size: {}, window size: {}", log_instance_size, window_size);

            // Ensure that `window_starts` length matches `num_windows`
            prop_assume!(window_size <= (ArkFr::N * 8) as u32);

            // Setup MetalMsmConfig with mock pipelines
            let mut config = setup_metal_state();

            // Create mock points and scalars
            let mut rng = StdRng::seed_from_u64(seed);
            let size = 1 << log_instance_size;
            let points: Vec<ArkG> = (0..size).map(|_| ArkG::rand(&mut rng)).collect();
            let scalars: Vec<ArkFr> = (0..size).map(|_| ArkFr::rand(&mut rng)).collect();

            let instance = encode_instances(&points, &scalars, &mut config, Some(window_size));

            // Execute the GPU kernel
            prepare_buckets_indices(&config, &instance);

            // Retrieve the results from the GPU buffer
            let gpu_buckets_indices_flat = MetalState::retrieve_contents::<u32>(&instance.data.buckets_indices_buffer);

            // Convert flat GPU output to a HashSet of tuples
            let gpu_buckets_indices: HashSet<(u32, u32)> = gpu_buckets_indices_flat
                .chunks(2)
                .map(|chunk| (chunk[0], chunk[1]))
                .collect();

            // Execute the pure Rust implementation
            let rust_buckets_indices = prepare_buckets_indices_rust(
                instance.params.window_size,
                instance.params.num_window,
                &scalars,
            );

            // Compare GPU results with Rust implementation
            prop_assert_eq!(gpu_buckets_indices, rust_buckets_indices);
        }
    }
}