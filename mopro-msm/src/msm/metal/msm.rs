pub mod prepare_buckets_indices;
pub mod sort_buckets;
pub mod bucket_wise_accumulation;
pub mod sum_reduction;

use std::ops::Add;
use std::sync::{Arc, Mutex};
use crate::msm::metal::abstraction::{
    errors::MetalError,
    limbs_conversion::{FromLimbs},
    state::*,
};
use ark_std::{vec::Vec};
// For benchmarking
use std::time::Instant;
use crate::msm::metal::abstraction::limbs_conversion::{PointGPU, ScalarGPU};
use crate::msm::utils::preprocess::MsmInstance;
use metal::*;
use objc::rc::autoreleasepool;
use rayon::prelude::{ParallelSliceMut, ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator, IntoParallelIterator};
use crate::msm::metal::msm::bucket_wise_accumulation::bucket_wise_accumulation;
use crate::msm::metal::msm::prepare_buckets_indices::prepare_buckets_indices;
use crate::msm::metal::msm::sort_buckets::sort_buckets_indices;
use crate::msm::metal::msm::sum_reduction::sum_reduction;

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

pub struct MetalMsmPipeline {
    pub prepare_buckets_indices: ComputePipelineState,
    pub bucket_wise_accumulation: ComputePipelineState,
    pub sum_reduction: ComputePipelineState,
    pub final_accumulation: ComputePipelineState,
}

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
    let final_accumulation = state.setup_pipeline("final_accumulation").unwrap();

    // TODO:
    let prepare_buckets_indices = state.setup_pipeline("prepare_buckets_indices").unwrap();
    let bucket_wise_accumulation = state.setup_pipeline("bucket_wise_accumulation").unwrap();
    let sum_reduction = state.setup_pipeline("sum_reduction").unwrap();

    MetalMsmConfig {
        state,
        pipelines: MetalMsmPipeline {
            prepare_buckets_indices,
            bucket_wise_accumulation,
            sum_reduction,
            final_accumulation,
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
            15 // TODO - learn how to calculate this
        }
    };
    let buckets_size = (1 << window_size) - 1;
    let window_starts: Vec<u32> = (0..modulus_bit_size as u32).step_by(window_size as usize).collect();
    let window_num = window_starts.len();

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
    let window_num_buffer = config.state.alloc_buffer_data(&[window_num as u32]);
    let instances_size_buffer = config.state.alloc_buffer_data(&[instances_size as u32]);
    let scalar_buffer = config.state.alloc_buffer_data(&scalars_limbs);
    let base_buffer = config.state.alloc_buffer_data(&bases_limbs);
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
    log::debug!("Store params time: {:?}", store_params_start.elapsed());

    // // debug
    // let debug_buffer = config.state.alloc_buffer::<u32>(2048);

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
            window_size: window_size as u32,
            window_num: window_num as u32,
        },
    }
}

pub fn exec_metal_commands<P: FromLimbs>(
    config: &MetalMsmConfig,
    instance: MetalMsmInstance,
) -> Result<P, MetalError> {
    let data = &instance.data;

    let prepare_time = Instant::now();
    prepare_buckets_indices(&config, &instance);
    log::debug!("Prepare buckets indices time: {:?}", prepare_time.elapsed());


    let sort_time = Instant::now();
    sort_buckets_indices(&config, &instance);
    log::debug!("Sort buckets indices time: {:?}", sort_time.elapsed());

    let accumulation_time = Instant::now();
    bucket_wise_accumulation(&config, &instance);
    log::debug!("Bucket wise accumulation time: {:?}", accumulation_time.elapsed());

    let reduction_time = Instant::now();
    sum_reduction(&config, &instance);
    log::debug!("Sum reduction time: {:?}", reduction_time.elapsed());

    {
        // Sequentially accumulate the msm results on GPU
        let final_time = Instant::now();
        autoreleasepool(|| {
            let (command_buffer, command_encoder) = config.state.setup_command(
                &config.pipelines.final_accumulation,
                Some(&[
                    (0, &data.window_size_buffer),
                    (1, &data.window_starts_buffer),
                    (2, &data.window_num_buffer),
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
        let end = ark_std::cmp::min((i + 1) * chunk_size, points.len());

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

#[cfg(test)]
mod tests {
    const LOG_INSTANCE_SIZE: u32 = 18;
    const NUM_INSTANCE: u32 = 10;

    #[cfg(feature = "ark")]
    mod ark {
        use ark_ec::{CurveGroup, VariableBaseMSM};
        use ark_std::cfg_into_iter;
        use rand::rngs::OsRng;
        use crate::msm::metal::abstraction::limbs_conversion::ark::{ArkFr, ArkG};
        use crate::msm::metal::msm::{metal_msm, metal_msm_parallel, setup_metal_state};
        use crate::msm::metal::msm::tests::{LOG_INSTANCE_SIZE, NUM_INSTANCE};
        use crate::msm::metal::tests::init_logger;
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

    }


    #[cfg(feature = "h2c")]
    mod h2c {
        use ark_std::cfg_into_iter;
        use halo2curves::group::Curve;
        use rand::rngs::OsRng;
        use crate::msm::metal::abstraction::limbs_conversion::h2c::{H2Fr, H2G};
        use crate::msm::metal::msm::{metal_msm, setup_metal_state};
        use crate::msm::metal::msm::tests::{LOG_INSTANCE_SIZE, NUM_INSTANCE};
        use crate::msm::metal::tests::init_logger;
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
    }

}
