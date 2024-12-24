use crate::msm::metal::abstraction::{
    errors::MetalError,
    limbs_conversion::{FromLimbs},
    state::*,
};
use crate::msm::utils::{benchmark::BenchmarkResult, preprocess};
use ark_std::{cfg_into_iter, vec::Vec};
// For benchmarking
use std::time::{Duration, Instant};

use crate::msm::metal::abstraction::limbs_conversion::{PointGPU, ScalarGPU};
use crate::msm::utils::preprocess::{get_or_create_msm_instances, MsmInstance};
use metal::*;
use objc::rc::autoreleasepool;
use rand::rngs::OsRng;
use rayon::prelude::ParallelSliceMut;

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
    pub num_window: u64,
}

pub struct MetalMsmPipeline {
    pub init_buckets: ComputePipelineState,
    pub accumulation_and_reduction: ComputePipelineState,
    pub final_accumulation: ComputePipelineState,
    pub prepare_buckets_indices: ComputePipelineState,
    pub bucket_wise_accumulation: ComputePipelineState,
    pub sum_reduction: ComputePipelineState,
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
fn ln_without_floats(a: usize) -> usize {
    // log2(a) * ln(2)
    (ark_std::log2(a) * 69 / 100) as usize
}

fn sort_buckets_indices(buckets_indices: &mut Vec<u32>) -> () {
    // parse the buckets_indices to a Vec<(u32, u32)>
    let mut buckets_indices_pairs: Vec<(u32, u32)> = Vec::new();
    for i in 0..buckets_indices.len() / 2 {
        // skip empty indices (0, 0)
        if buckets_indices[2 * i] == 0 && buckets_indices[2 * i + 1] == 0 {
            continue;
        }
        buckets_indices_pairs.push((buckets_indices[2 * i], buckets_indices[2 * i + 1]));
    }
    // parallel sort the buckets_indices_pairs by the first element
    buckets_indices_pairs.par_sort_by(|a, b| a.0.cmp(&b.0));

    // flatten the sorted pairs to a Vec<u32>
    buckets_indices.clear();
    for (start, end) in buckets_indices_pairs {
        buckets_indices.push(start);
        buckets_indices.push(end);
    }
}

pub fn setup_metal_state() -> MetalMsmConfig {
    let state = MetalState::new(None).unwrap();
    let init_buckets = state.setup_pipeline("initialize_buckets").unwrap();
    let accumulation_and_reduction = state
        .setup_pipeline("accumulation_and_reduction_phase")
        .unwrap();
    let final_accumulation = state.setup_pipeline("final_accumulation").unwrap();

    // TODO:
    let prepare_buckets_indices = state.setup_pipeline("prepare_buckets_indices").unwrap();
    let bucket_wise_accumulation = state.setup_pipeline("bucket_wise_accumulation").unwrap();
    let sum_reduction = state.setup_pipeline("sum_reduction").unwrap();
    // let make_histogram_uint32 = state.setup_pipeline("make_histogram_uint32").unwrap();
    // let reorder_uint32 = state.setup_pipeline("reorder_uint32").unwrap();

    // let make_histogram_uint32_raw = state.library.get_function("reorder_uint32", None).unwrap();
    // let tmp = state.setup_pipeline("reorder_uint32").unwrap();
    // println!("tmp: {:?}", tmp);
    // state.library.function_names().iter().for_each(|name| {
    //     println!("Function name: {:?}", name);
    // });
    // let compute_descriptor = ComputePipelineDescriptor::new();
    // compute_descriptor.set_compute_function(Some(&make_histogram_uint32_raw));
    // println!("make_histogram_uint32: {:?}", compute_descriptor.compute_function().unwrap());
    // println!("make_histogram_uint32: {:?}", result);

    MetalMsmConfig {
        state,
        pipelines: MetalMsmPipeline {
            init_buckets,
            accumulation_and_reduction,
            final_accumulation,
            prepare_buckets_indices,
            bucket_wise_accumulation,
            sum_reduction,
        },
    }
}

pub fn encode_instances<P: PointGPU, S: ScalarGPU>(
    points: &[P],
    scalars: &[S],
    config: &mut MetalMsmConfig,
) -> MetalMsmInstance {
    let modulus_bit_size = S::MODULUS_BIT_SIZE;

    let instances_size = ark_std::cmp::min(points.len(), scalars.len());
    let window_size = if instances_size < 32 {
        3
    } else {
        ln_without_floats(instances_size) + 2
    };
    let buckets_size = (1 << window_size) - 1;
    let window_starts: Vec<usize> = (0..modulus_bit_size).step_by(window_size).collect();
    let num_windows = window_starts.len();

    // flatten scalar and base to Vec<u32> for GPU usage
    let scalars_limbs = cfg_into_iter!(scalars)
        .map(S::to_u32_limbs)
        .flatten()
        .collect::<Vec<u32>>();
    let bases_limbs = cfg_into_iter!(points)
        .map(P::to_u32_limbs)
        .flatten()
        .collect::<Vec<u32>>();

    // store params to GPU shared memory
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
    let window_starts_buffer = config.state.alloc_buffer_data(
        &(window_starts
            .iter()
            .map(|x| *x as u32)
            .collect::<Vec<u32>>()),
    );
    // prepare bucket_size * num_windows * 2
    let buckets_indices_buffer = config
        .state
        .alloc_buffer::<u32>(instances_size * num_windows * 2);

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
            num_window: num_windows as u64,
        },
    }
}

pub fn exec_metal_commands<P: FromLimbs>(
    config: &MetalMsmConfig,
    instance: MetalMsmInstance,
) -> Result<P, MetalError> {
    let data = instance.data;
    let params = instance.params;

    // Init the pipleline for MSM
    let init_time = Instant::now();
    autoreleasepool(|| {
        let (command_buffer, command_encoder) = config.state.setup_command(
            &config.pipelines.init_buckets,
            Some(&[
                (0, &data.window_size_buffer),
                (1, &data.window_starts_buffer),
                (2, &data.buckets_matrix_buffer),
            ]),
        );
        command_encoder
            .dispatch_thread_groups(MTLSize::new(params.num_window, 1, 1), MTLSize::new(1, 1, 1));
        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    });
    log::debug!("Init buckets time: {:?}", init_time.elapsed());

    let prepare_time = Instant::now();
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
    log::debug!("Prepare buckets indices time: {:?}", prepare_time.elapsed());

    // sort the buckets_indices in CPU parallelly
    let sort_start = Instant::now();
    let mut buckets_indices = MetalState::retrieve_contents::<u32>(&data.buckets_indices_buffer);
    sort_buckets_indices(&mut buckets_indices);

    // send the sorted buckets back to GPU
    let sorted_buckets_indices_buffer = config.state.alloc_buffer_data(&buckets_indices);
    log::debug!("Sort buckets indices time: {:?}", sort_start.elapsed());

    // accumulate the buckets_matrix using sorted bucket indices on GPU
    let max_threads_per_group = MTLSize::new(
        config
            .pipelines
            .bucket_wise_accumulation
            .thread_execution_width(),
        config
            .pipelines
            .bucket_wise_accumulation
            .max_total_threads_per_threadgroup()
            / config
                .pipelines
                .bucket_wise_accumulation
                .thread_execution_width(),
        1,
    );
    let max_thread_size = params.buckets_size as u64 * params.num_window;
    let opt_threadgroups_amount = max_thread_size
        / config
            .pipelines
            .bucket_wise_accumulation
            .max_total_threads_per_threadgroup()
        + 1;
    let opt_threadgroups = MTLSize::new(opt_threadgroups_amount, 1, 1);
    log::debug!(
        "(accumulation) max thread per threadgroup: {:?}",
        max_threads_per_group
    );
    log::debug!("(accumulation) opt threadgroups: {:?}", opt_threadgroups);

    let max_thread_size_accu_buffer = config.state.alloc_buffer_data(&[max_thread_size as u32]);
    let bucket_wise_time = Instant::now();
    autoreleasepool(|| {
        let (command_buffer, command_encoder) = config.state.setup_command(
            &config.pipelines.bucket_wise_accumulation,
            Some(&[
                (0, &data.instances_size_buffer),
                (1, &data.num_windows_buffer),
                (2, &data.base_buffer),
                (3, &sorted_buckets_indices_buffer),
                (4, &data.buckets_matrix_buffer),
                (5, &max_thread_size_accu_buffer),
                // (6, &data.debug_buffer),
            ]),
        );
        // command_encoder.dispatch_thread_groups(
        //     MTLSize::new(params.buckets_size as u64 * params.num_window, 1, 1),
        //     MTLSize::new(1, 1, 1),
        // );
        command_encoder.dispatch_thread_groups(opt_threadgroups, max_threads_per_group);
        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    });
    let bucket_wise_elapsed = bucket_wise_time.elapsed();
    log::debug!(
        "Bucket wise accumulation time (using {:?} threads): {:?}",
        params.buckets_size as u64 * params.num_window,
        bucket_wise_elapsed
    );

    // // debug
    // let debug_data = MetalState::retrieve_contents::<u32>(&data.debug_buffer);
    // log::debug!("Debug data: {:?}", debug_data);

    // Reduce the buckets_matrix on GPU
    let max_thread_size = params.num_window;
    let opt_threadgroups_amount = max_thread_size
        / config
            .pipelines
            .bucket_wise_accumulation
            .max_total_threads_per_threadgroup()
        + 1;
    let opt_threadgroups = MTLSize::new(opt_threadgroups_amount, 1, 1);
    let max_thread_size_reduc_buffer = config.state.alloc_buffer_data(&[max_thread_size as u32]);
    let reduction_time = Instant::now();
    autoreleasepool(|| {
        let (command_buffer, command_encoder) = config.state.setup_command(
            &config.pipelines.sum_reduction,
            Some(&[
                (0, &data.window_size_buffer),
                (1, &data.scalar_buffer),
                (2, &data.base_buffer),
                (3, &data.buckets_matrix_buffer),
                (4, &data.res_buffer),
                (5, &max_thread_size_reduc_buffer),
            ]),
        );
        // command_encoder
        //     .dispatch_thread_groups(MTLSize::new(params.num_window, 1, 1), MTLSize::new(1, 1, 1));
        command_encoder.dispatch_thread_groups(opt_threadgroups, max_threads_per_group);
        command_encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    });
    log::debug!("Reduction time: {:?}", reduction_time.elapsed());

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

    // retrieve and parse the result from GPU
    let msm_result = {
        let raw_limbs = MetalState::retrieve_contents::<u32>(&data.result_buffer);
        P::from_u32_limbs(&raw_limbs)
    };

    Ok(msm_result)
}

pub fn metal_msm<P: PointGPU, S: ScalarGPU>(
    points: &[P],
    scalars: &[S],
    config: &mut MetalMsmConfig,
) -> Result<P, MetalError> {
    let instance = encode_instances(points, scalars, config);
    exec_metal_commands(config, instance)
}

pub fn benchmark_msm<P: PointGPU, S: ScalarGPU>(
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
            let metal_instance = encode_instances(points, &scalars[..], &mut metal_config);
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

pub fn run_benchmark<P: PointGPU, S: ScalarGPU + FromLimbs>(
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
        use crate::msm::metal::msm::{metal_msm, run_benchmark, setup_metal_state};
        use crate::msm::metal::msm::tests::{init_logger, BENCHMARKSPATH, LOG_INSTANCE_SIZE, NUM_INSTANCE};
        use crate::msm::utils::preprocess::get_or_create_msm_instances;

        #[test]
        fn test_msm_correctness_medium_sample_ark() {
            init_logger();

            let mut metal_config = setup_metal_state();
            let rng = OsRng::default();

            let instances = get_or_create_msm_instances::<ArkG, ArkFr>(2 << LOG_INSTANCE_SIZE, NUM_INSTANCE, rng, None).unwrap();

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
