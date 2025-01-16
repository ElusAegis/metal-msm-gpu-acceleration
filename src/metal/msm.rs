pub mod bucket_wise_accumulation;
mod final_accumulation;
pub mod prepare_buckets_indices;
pub mod sort_buckets;
pub mod sum_reduction;

use crate::metal::abstraction::{errors::MetalError, state::*};
use std::sync::{Arc, Condvar, Mutex};
// For benchmarking
use crate::metal::abstraction::limbs_conversion::{PointGPU, ScalarGPU, ToLimbs};
use crate::metal::msm::bucket_wise_accumulation::bucket_wise_accumulation;
use crate::metal::msm::final_accumulation::final_accumulation;
use crate::metal::msm::prepare_buckets_indices::prepare_buckets_indices;
use crate::metal::msm::sort_buckets::sort_buckets_indices;
use crate::metal::msm::sum_reduction::sum_reduction;
use crate::utils::preprocess::MsmInstance;
use metal::*;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::time::Instant;
use once_cell::sync::Lazy;

// Global optional config to be reused
static GLOBAL_METAL_CONFIG: Lazy<Mutex<Option<MetalMsmConfig>>> = Lazy::new(|| Mutex::new(None));

pub struct MetalMsmData {
    pub window_size_buffer: Buffer,
    pub window_num_buffer: Buffer,
    pub instances_size_buffer: Buffer,
    pub window_starts_buffer: Buffer,
    pub scalar_buffer: Buffer,
    pub base_buffer: Buffer,
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
    pub window_num: u32,
}

 #[derive(Clone)]
pub struct MetalMsmPipeline {
    pub prepare_buckets_indices: ComputePipelineState,
    pub bucket_wise_accumulation: ComputePipelineState,
    pub sum_reduction_partial: ComputePipelineState,
    pub sum_reduction_final: ComputePipelineState,
}

#[derive(Clone)]
pub struct MetalMsmConfig {
    pub state: MetalState,
    pub pipelines: MetalMsmPipeline,
}

pub struct MetalMsmInstance {
    pub data: MetalMsmData,
    pub params: MetalMsmParams,
}

// // Helper function for getting the windows size
// // TODO - find out the heuristic
// fn ln_without_floats(a: usize) -> usize {
//     // log2(a) * ln(2)
//     (ark_std::log2(a) * 69 / 100) as usize
// }

pub fn setup_metal_state() -> MetalMsmConfig {
    let state = MetalState::new(None).unwrap();

    let prepare_buckets_indices = state.setup_pipeline("prepare_buckets_indices").unwrap();
    let bucket_wise_accumulation = state.setup_pipeline("bucket_wise_accumulation").unwrap();
    let sum_reduction_partial = state.setup_pipeline("sum_reduction_partial").unwrap();
    let sum_reduction_final = state.setup_pipeline("sum_reduction_final").unwrap();

    MetalMsmConfig {
        state,
        pipelines: MetalMsmPipeline {
            prepare_buckets_indices,
            bucket_wise_accumulation,
            sum_reduction_partial,
            sum_reduction_final,
        },
    }
}

pub fn setup_metal_state_reusable() -> MetalMsmConfig {
    let mut config_guard = GLOBAL_METAL_CONFIG.lock().unwrap();
    if let Some(config) = config_guard.clone() {
        log::debug!("MetalMsmConfig already initialized; reusing existing config.");
        config
    } else {
        let config = setup_metal_state(); // Your original function
        *config_guard = Some(config);
        log::debug!("MetalMsmConfig initialized globally!");
        config_guard.clone().expect("Failed to initialize MetalMsmConfig")
    }
}

/// Retrieve a (cloned) `MetalMsmConfig` from the global static.
/// If you need an actual mutable reference to the *same* config,
/// you must be careful about concurrency.
pub fn get_global_metal_config() -> MetalMsmConfig {
    let config_guard = GLOBAL_METAL_CONFIG.lock().unwrap();
    config_guard
        .clone()
        .expect("MetalMsmConfig must be initialized before use.")
}

pub fn encode_instances<
    P: ToLimbs<NP> + Sync,
    S: ScalarGPU<NS> + Sync,
    const NP: usize,
    const NS: usize,
>(
    bases: &[P],
    scalars: &[S],
    config: &mut MetalMsmConfig,
    window_size: Option<u32>,
) -> MetalMsmInstance {
    let modulus_bit_size = S::MODULUS_BIT_SIZE;

    let instances_size = bases.len().min(scalars.len());
    let window_size = if let Some(window_size) = window_size {
        window_size
    } else if instances_size < 32 {
        3
    } else {
        15 // TODO - learn how to calculate this
    };
    let buckets_size = (1 << window_size) - 1;
    let window_starts: Vec<u32> = (0..modulus_bit_size as u32)
        .step_by(window_size as usize)
        .collect();
    let window_num = window_starts.len();

    // store params to GPU shared memory
    let window_size_buffer = config.state.alloc_buffer_data(&[window_size]);
    let window_num_buffer = config.state.alloc_buffer_data(&[window_num as u32]);
    let instances_size_buffer = config.state.alloc_buffer_data(&[instances_size as u32]);
    let scalar_buffer = config.state.alloc_buffer_data_direct(scalars);
    let base_buffer = config.state.alloc_buffer_data_direct(bases);
    let buckets_matrix_buffer = config
        .state
        .alloc_buffer::<u32>(buckets_size * window_num * 8 * 3);
    let res_buffer = config.state.alloc_buffer::<u32>(window_num * 8 * 3);
    let result_buffer = config.state.alloc_buffer::<u32>(8 * 3);
    // convert window_starts to u32 to give the exact storage need for GPU
    let window_starts_buffer = config.state.alloc_buffer_data(&window_starts);
    // prepare bucket_size * num_windows * 2
    let buckets_indices_buffer = config
        .state
        .alloc_buffer::<u32>(instances_size * window_num * 2);

    MetalMsmInstance {
        data: MetalMsmData {
            window_size_buffer,
            window_num_buffer,
            instances_size_buffer,
            window_starts_buffer,
            scalar_buffer,
            base_buffer,
            buckets_matrix_buffer,
            buckets_indices_buffer,
            res_buffer,
            result_buffer,
            // debug_buffer,
        },
        params: MetalMsmParams {
            instances_size: instances_size as u32,
            buckets_size: buckets_size as u32,
            window_size: window_size,
            window_num: window_num as u32,
        },
    }
}

pub fn exec_metal_commands<P: PointGPU<24>>(
    config: &MetalMsmConfig,
    instance: MetalMsmInstance,
) -> Result<P, MetalError> {
    let prepare_time = Instant::now();
    prepare_buckets_indices(config, &instance);
    log::debug!("Prepare buckets indices time: {:?}", prepare_time.elapsed());

    let sort_time = Instant::now();
    sort_buckets_indices(config, &instance);
    log::debug!("Sort buckets indices time: {:?}", sort_time.elapsed());

    let accumulation_time = Instant::now();
    bucket_wise_accumulation(config, &instance);
    log::debug!(
        "Bucket wise accumulation time: {:?}",
        accumulation_time.elapsed()
    );

    let reduction_time = Instant::now();
    sum_reduction(config, &instance);
    log::debug!("Sum reduction time: {:?}", reduction_time.elapsed());

    let final_time = Instant::now();
    let msm_result = final_accumulation::<P>(config, &instance);
    log::debug!("Final accumulation time: {:?}", final_time.elapsed());

    Ok(msm_result)
}

pub fn metal_msm<P: PointGPU<24> + Sync, S: ScalarGPU<8> + Sync>(
    points: &[P],
    scalars: &[S],
    config: &mut MetalMsmConfig,
) -> Result<P, MetalError> {
    let encoding_time = Instant::now();
    let instance = encode_instances(points, scalars, config, None);
    log::debug!("Encoding Instance Time: {:?}", encoding_time.elapsed());

    let exec_time = Instant::now();
    let res = exec_metal_commands(config, instance);
    log::debug!("> Pure Compute Time: {:?}", exec_time.elapsed());

    res
}

pub fn metal_msm_parallel<P, S>(
    instance: &MsmInstance<P, S>,
    target_msm_log_size: Option<usize>,
) -> P
where
    P: PointGPU<24> + Send + Sync + Clone,
    S: ScalarGPU<8> + Send + Sync,
{
    let points = &instance.points;
    let scalars = &instance.scalars;

    // Shared accumulators
    let accumulator = Arc::new(Mutex::new(P::from_u32_limbs(&[0; 24])));

    // We believe optimal chunk size is 1/3 of the target MSM length
    let chunk_size = if let Some(target_msm_log_size) = target_msm_log_size {
        2usize.pow(target_msm_log_size as u32)
    } else {
        points.len() / 2
    };
    let amount_of_sub_instances = (points.len() + chunk_size - 1) / chunk_size;

    (0..amount_of_sub_instances).into_par_iter().for_each(|i| {
        let start = i * chunk_size;
        let end = ((i + 1) * chunk_size).min(points.len());

        let sub_points = &points[start..end];
        let sub_scalars = &scalars[start..end];

        let mut config = setup_metal_state();
        let sub_instance = encode_instances(sub_points, sub_scalars, &mut config, None);

        let partial_result = exec_metal_commands(&config, sub_instance).unwrap();

        let mut accumulator = accumulator.lock().unwrap();
        // Add partial result to the shared accumulator
        *accumulator = accumulator.clone().add(partial_result);
    });

    // Extract the final accumulated result
    Arc::try_unwrap(accumulator)
        .map_err(|_| "Failed to unwrap accumulator")
        .expect("Failed to unwrap accumulator")
        .into_inner()
        .unwrap()
}

#[cfg(feature = "h2c")]
pub fn gpu_msm_h2c<C, PIN, POUT, S>(scalars: &[C::Scalar], points: &[C]) -> C::Curve
where
    C: halo2curves::CurveAffine, // Your curve type
    PIN: ToLimbs<24> + Sync,     // GPU-compatible point type
    POUT: PointGPU<24> + Sync,   // GPU-compatible point type
    S: ScalarGPU<8> + Sync,      // GPU-compatible scalar type
{
    // Step 1: Convert scalars to GPU representation
    let gpu_scalars: &[S] = unsafe {
        assert_eq!(
            std::mem::size_of::<C::Scalar>(),
            std::mem::size_of::<S>(),
            "C::Scalar and S must have the same size for reinterpret casting"
        );
        assert_eq!(
            std::mem::align_of::<C::Scalar>(),
            std::mem::align_of::<S>(),
            "C::Scalar and S must have the same alignment for reinterpret casting"
        );
        std::slice::from_raw_parts(scalars.as_ptr() as *const S, scalars.len())
    };

    // Step 2: Convert points to GPU representation
    let gpu_points: &[PIN] = unsafe {
        assert_eq!(
            std::mem::size_of::<C>(),
            std::mem::size_of::<PIN>(),
            "C and P must have the same size for reinterpret casting"
        );
        assert_eq!(
            std::mem::align_of::<C>(),
            std::mem::align_of::<PIN>(),
            "C and P must have the same alignment for reinterpret casting"
        );
        std::slice::from_raw_parts(points.as_ptr() as *const PIN, points.len())
    };

    // Step 3: Setup GPU
    let setup_time = Instant::now();
    let mut config = setup_metal_state_reusable();
    log::debug!("Config Setup Time: {:?}", setup_time.elapsed());

    // Step 4: Perform the GPU computation
    let encode_time = Instant::now();
    let instance = encode_instances(gpu_points, gpu_scalars, &mut config, None);
    log::debug!("Encode Instances Time: {:?}", encode_time.elapsed());
    let compute_time = Instant::now();
    let gpu_result: POUT = exec_metal_commands(&config, instance).unwrap();
    log::debug!("Pure Compute Time: {:?}", compute_time.elapsed());

    // Step 5: Convert GPU result to CPU representation
    let cpu_result: C::Curve = unsafe {
        assert_eq!(
            std::mem::size_of::<C::Curve>(),
            std::mem::size_of::<POUT>(),
            "C and P must have the same size for reinterpret casting"
        );
        assert_eq!(
            std::mem::align_of::<C::Curve>(),
            std::mem::align_of::<POUT>(),
            "C and P must have the same alignment for reinterpret casting"
        );
        std::ptr::read(&gpu_result as *const POUT as *const C::Curve)
    };

    cpu_result
}

#[cfg(feature = "h2c")]
pub fn gpu_msm_h2c_sync<C, PIN, POUT, S>(scalars: &[C::Scalar], points: &[C], sync_pair: Arc<(Mutex<bool>, Condvar)>) -> C::Curve
where
    C: halo2curves::CurveAffine, // Your curve type
    PIN: ToLimbs<24> + Sync,     // GPU-compatible point type
    POUT: PointGPU<24> + Sync,   // GPU-compatible point type
    S: ScalarGPU<8> + Sync,      // GPU-compatible scalar type
{
    // Step 1: Convert scalars to GPU representation
    let gpu_scalars: &[S] = unsafe {
        assert_eq!(
            size_of::<C::Scalar>(),
            size_of::<S>(),
            "C::Scalar and S must have the same size for reinterpret casting"
        );
        assert_eq!(
            align_of::<C::Scalar>(),
            align_of::<S>(),
            "C::Scalar and S must have the same alignment for reinterpret casting"
        );
        std::slice::from_raw_parts(scalars.as_ptr() as *const S, scalars.len())
    };

    // Step 2: Convert points to GPU representation
    let gpu_points: &[PIN] = unsafe {
        assert_eq!(
            size_of::<C>(),
            size_of::<PIN>(),
            "C and P must have the same size for reinterpret casting"
        );
        assert_eq!(
            align_of::<C>(),
            align_of::<PIN>(),
            "C and P must have the same alignment for reinterpret casting"
        );
        std::slice::from_raw_parts(points.as_ptr() as *const PIN, points.len())
    };

    // Step 3: Setup GPU
    let setup_time = Instant::now();
    let mut config = setup_metal_state_reusable();
    log::debug!("Config Setup Time: {:?}", setup_time.elapsed());

    // Step 4: Perform the GPU computation
    let encode_time = Instant::now();
    let instance = encode_instances(gpu_points, gpu_scalars, &mut config, None);
    log::debug!("Encode Instances Time: {:?}", encode_time.elapsed());
    let compute_time = Instant::now();
    let prepare_time = Instant::now();
    prepare_buckets_indices(&config, &instance);
    log::debug!("Prepare buckets indices time: {:?}", prepare_time.elapsed());

    let sort_time = Instant::now();
    sort_buckets_indices(&config, &instance);

    // Notify the CPU thread that GPU sorting is complete
    {
        let (lock, cvar) = &*sync_pair;
        let mut gpu_done = lock.lock().unwrap();
        *gpu_done = true;
        cvar.notify_one();
    }

    log::debug!("Sort buckets indices time: {:?}", sort_time.elapsed());

    let accumulation_time = Instant::now();
    bucket_wise_accumulation(&config, &instance);
    log::debug!(
        "Bucket wise accumulation time: {:?}",
        accumulation_time.elapsed()
    );

    let reduction_time = Instant::now();
    sum_reduction(&config, &instance);
    log::debug!("Sum reduction time: {:?}", reduction_time.elapsed());

    let final_time = Instant::now();
    let msm_result = final_accumulation::<POUT>(&config, &instance);
    log::debug!("Final accumulation time: {:?}", final_time.elapsed());
    log::debug!("Pure Compute Time: {:?}", compute_time.elapsed());

    // Step 5: Convert GPU result to CPU representation
    let cpu_result: C::Curve = unsafe {
        assert_eq!(
            std::mem::size_of::<C::Curve>(),
            std::mem::size_of::<POUT>(),
            "C and P must have the same size for reinterpret casting"
        );
        assert_eq!(
            std::mem::align_of::<C::Curve>(),
            std::mem::align_of::<POUT>(),
            "C and P must have the same alignment for reinterpret casting"
        );
        std::ptr::read(&msm_result as *const POUT as *const C::Curve)
    };

    cpu_result
}


#[cfg(feature = "h2c")]
pub fn gpu_with_cpu<C, PIN, POUT, S>(scalar: &[C::Scalar], points: &[C]) -> C::Curve
where
    C: halo2curves::CurveAffine, // Your curve type
    PIN: ToLimbs<24> + Sync,     // GPU-compatible point type
    POUT: PointGPU<24> + Sync,   // GPU-compatible point type
    S: ScalarGPU<8> + Sync,      // GPU-compatible scalar type
{
    // Split the scalar and points into two halves
    // TODO - learn how to select the best split ratio - for lower values of n, CPU is faster
    let split_at = if scalar.len() < 2usize.pow(18) {
        scalar.len() * 1 / 3
    } else if scalar.len() < 2usize.pow(20) {
        scalar.len() * 1 / 2
    } else {
        scalar.len() * 2 / 5
    };

    let (scalar_1, scalar_2) = scalar.split_at(split_at);
    let (point_1, point_2) = points.split_at(split_at);

    // Create a shared Arc<Condvar> to synchronize GPU and CPU
    let sync_pair = Arc::new((Mutex::new(false), Condvar::new()));
    let sync_pair_for_cpu = Arc::clone(&sync_pair);

    // Use Rayon to run GPU and CPU tasks in parallel
    let (gpu_result, cpu_result) = rayon::join(
        || {
            let gpu_start = Instant::now();

            // We release the lock after sorting is done
            let res = gpu_msm_h2c_sync::<C, PIN, POUT, S>(scalar_1, point_1, sync_pair);

            log::debug!("> GPU Time: {:?}", gpu_start.elapsed());

            res
        },
        || {

            // Wait for the GPU thread to send the completion signal
            let (lock, cvar) = &*sync_pair_for_cpu;
            let mut gpu_done = lock.lock().unwrap();
            while !*gpu_done {
                gpu_done = cvar.wait(gpu_done).unwrap();
            }

            let cpu_start = Instant::now();

            let res = halo2curves::msm::msm_best(scalar_2, point_2);

            log::debug!("> CPU Time: {:?}", cpu_start.elapsed());

            res
        },
    );

    // Combine GPU and CPU results
    gpu_result + cpu_result
}

#[cfg(feature = "h2c")]
pub fn msm_best<C, PIN, POUT, S>(scalars: &[C::Scalar], points: &[C]) -> C::Curve
where
    C: halo2curves::CurveAffine, // Your curve type
    PIN: ToLimbs<24> + Sync,     // GPU-compatible point type
    POUT: PointGPU<24> + Sync,   // GPU-compatible point type
    S: ScalarGPU<8> + Sync,      // GPU-compatible scalar type
{
    if scalars.len() >= 2_usize.pow(17) {
        gpu_with_cpu::<C, PIN, POUT, S>(scalars, points)
    } else {
        halo2curves::msm::msm_best(scalars, points)
    }
}

#[cfg(test)]
mod tests {
    const LOG_INSTANCE_SIZE: u32 = 20;
    const NUM_INSTANCE: u32 = 5;

    #[cfg(feature = "ark")]
    mod ark {
        use crate::metal::abstraction::limbs_conversion::ark::{ArkFr, ArkG};
        use crate::metal::msm::tests::{LOG_INSTANCE_SIZE, NUM_INSTANCE};
        use crate::metal::msm::{metal_msm, metal_msm_parallel, setup_metal_state};
        use crate::metal::tests::init_logger;
        use crate::utils::preprocess::get_or_create_msm_instances;
        use ark_ec::{CurveGroup, VariableBaseMSM};
        use ark_std::cfg_into_iter;
        use rand::rngs::OsRng;

        #[test]
        fn test_msm_correctness_medium_sample_ark() {
            init_logger();

            let mut metal_config = setup_metal_state();
            let rng = OsRng::default();

            let instances = get_or_create_msm_instances::<ArkG, ArkFr>(
                LOG_INSTANCE_SIZE,
                NUM_INSTANCE,
                rng,
                None,
            )
            .unwrap();

            for (i, instance) in instances.iter().enumerate() {
                let points = &instance.points;
                // map each scalar to a ScalarField
                let scalars = &instance.scalars;

                let affine_points: Vec<_> =
                    cfg_into_iter!(points).map(|p| p.into_affine()).collect();
                let ark_msm = ArkG::msm(&affine_points, &scalars[..]).unwrap();

                let metal_msm =
                    metal_msm::<ArkG, ArkFr>(&points[..], &scalars[..], &mut metal_config).unwrap();
                assert_eq!(
                    metal_msm.into_affine(),
                    ark_msm.into_affine(),
                    "This msm is wrongly computed"
                );
                log::info!(
                    "(pass) {}th instance of size 2^{} is correctly computed",
                    i,
                    LOG_INSTANCE_SIZE
                );
            }
        }

        #[test]
        fn test_parallel_gpu_metal_msm_correctness() {
            init_logger();

            let rng = OsRng::default();
            let mut metal_config = setup_metal_state();

            let instances = get_or_create_msm_instances::<ArkG, ArkFr>(
                LOG_INSTANCE_SIZE,
                NUM_INSTANCE,
                rng,
                None,
            )
            .unwrap();

            for (i, instance) in instances.iter().enumerate() {
                let points = &instance.points;
                // map each scalar to a ScalarField
                let scalars = &instance.scalars;

                let affine_points: Vec<_> =
                    cfg_into_iter!(points).map(|p| p.into_affine()).collect();
                let ark_msm = ArkG::msm(&affine_points, &scalars[..]).unwrap();

                let metal_msm_par = metal_msm_parallel(instance, None);
                let metal_msm =
                    metal_msm::<ArkG, ArkFr>(&points[..], &scalars[..], &mut metal_config).unwrap();
                assert_eq!(
                    metal_msm.into_affine(),
                    ark_msm.into_affine(),
                    "This msm is wrongly computed"
                );
                assert_eq!(
                    metal_msm.into_affine(),
                    metal_msm_par.into_affine(),
                    "This parallel msm is wrongly computed"
                );
                log::info!(
                    "(pass) {}th instance of size 2^{} is correctly computed",
                    i,
                    LOG_INSTANCE_SIZE
                );
            }
        }
    }

    #[cfg(feature = "h2c")]
    mod h2c {
        use crate::metal::abstraction::limbs_conversion::h2c::{H2Fr, H2GAffine, H2G};
        use crate::metal::msm::tests::{LOG_INSTANCE_SIZE, NUM_INSTANCE};
        use crate::metal::msm::{gpu_msm_h2c, gpu_with_cpu, metal_msm, setup_metal_state};
        use crate::metal::tests::init_logger;
        use crate::utils::preprocess::get_or_create_msm_instances;
        use halo2curves::group::Curve;
        use rand::rngs::OsRng;
        use rayon::prelude::{IntoParallelIterator, ParallelIterator};

        #[test]
        fn test_msm_correctness_medium_sample_h2c() {
            init_logger();

            let mut metal_config = setup_metal_state();
            let rng = OsRng::default();

            let instances = get_or_create_msm_instances::<H2G, H2Fr>(
                LOG_INSTANCE_SIZE,
                NUM_INSTANCE,
                rng,
                None,
            )
            .unwrap();

            for (i, instance) in instances.iter().enumerate() {
                let points = &instance.points;
                // map each scalar to a ScalarField
                let scalars = &instance.scalars;

                let affine_points: Vec<_> = points.into_par_iter().map(|p| p.to_affine()).collect();
                let h2c_msm = halo2curves::msm::msm_best(&scalars[..], &affine_points[..]);

                let metal_msm =
                    metal_msm::<H2G, H2Fr>(&points[..], &scalars[..], &mut metal_config).unwrap();
                assert_eq!(
                    metal_msm.to_affine(),
                    h2c_msm.to_affine(),
                    "This msm is wrongly computed"
                );
                log::info!(
                    "(pass) {}th instance of size 2^{} is correctly computed",
                    i,
                    LOG_INSTANCE_SIZE
                );
            }
        }

        #[test]
        fn test_best_msm_medium_sample_h2c() {
            init_logger();

            let rng = OsRng::default();

            let instances = get_or_create_msm_instances::<H2G, H2Fr>(
                LOG_INSTANCE_SIZE,
                NUM_INSTANCE,
                rng,
                None,
            )
            .unwrap();

            for (i, instance) in instances.iter().enumerate() {
                let points = &instance.points;
                // map each scalar to a ScalarField
                let scalars = &instance.scalars;

                let affine_points: Vec<_> = points.into_par_iter().map(|p| p.to_affine()).collect();
                let h2c_msm = halo2curves::msm::msm_best(&scalars[..], &affine_points[..]);

                let metal_msm =
                    gpu_msm_h2c::<H2GAffine, H2GAffine, H2G, H2Fr>(scalars, &affine_points);
                assert_eq!(
                    metal_msm.to_affine(),
                    h2c_msm.to_affine(),
                    "This msm is wrongly computed"
                );
                log::info!(
                    "(pass) {}th instance of size 2^{} is correctly computed",
                    i,
                    LOG_INSTANCE_SIZE
                );
            }
        }

        #[test]
        fn test_gpu_cpu_correctness_medium_sample_h2c() {
            init_logger();

            let rng = OsRng::default();

            let instances = get_or_create_msm_instances::<H2G, H2Fr>(
                LOG_INSTANCE_SIZE,
                NUM_INSTANCE,
                rng,
                None,
            )
            .unwrap();

            for (i, instance) in instances.iter().enumerate() {
                let points = &instance.points;
                // map each scalar to a ScalarField
                let scalars = &instance.scalars;

                let affine_points: Vec<_> = points.into_par_iter().map(|p| p.to_affine()).collect();
                let h2c_msm = halo2curves::msm::msm_best(&scalars[..], &affine_points[..]);

                let metal_msm =
                    gpu_with_cpu::<H2GAffine, H2GAffine, H2G, H2Fr>(scalars, &affine_points);
                assert_eq!(
                    metal_msm.to_affine(),
                    h2c_msm.to_affine(),
                    "This msm is wrongly computed"
                );
                log::info!(
                    "(pass) {}th instance of size 2^{} is correctly computed",
                    i,
                    LOG_INSTANCE_SIZE
                );
            }
        }
    }
}
