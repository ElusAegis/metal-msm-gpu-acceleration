#![cfg(all(feature = "ark", feature = "h2c"))]

use std::time::Duration;
use ark_ec::{CurveGroup, VariableBaseMSM};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use halo2curves::group::{Curve};
use halo2curves::msm::msm_best;
use instant::Instant;
use rand::rngs::OsRng;
use mopro_msm::msm::metal::abstraction::limbs_conversion::ark::{ArkFr, ArkG, ArkGAffine};
use mopro_msm::msm::metal::abstraction::limbs_conversion::h2c::{H2Fr, H2GAffine, H2G};
use mopro_msm::msm::metal::abstraction::limbs_conversion::{PointGPU, ScalarGPU};
use mopro_msm::msm::metal::msm::{encode_instances, exec_metal_commands, metal_msm_all_gpu, metal_msm_parallel, setup_metal_state};
use mopro_msm::msm::utils::preprocess::{get_or_create_msm_instances, MsmInstance};

pub fn msm_h2c_cpu(instances: &Vec<(Vec<H2GAffine>, Vec<H2Fr>)>) {
    for instance in instances {
        let _ = msm_best(&instance.1, &instance.0);
    }
}

pub fn msm_ark_cpu(instances: &Vec<(Vec<ArkGAffine>, Vec<ArkFr>)>) {
    for instance in instances {
        let _ = ArkG::msm(&instance.0, &instance.1).unwrap();
    }
}

fn msm_gpu<P: PointGPU<24> + Sync, S: ScalarGPU<8> + Sync>(instances: &Vec<MsmInstance<P, S>>) {

    let metal_config_start = Instant::now();
    let mut metal_config = setup_metal_state();
    log::debug!("Done setting up metal state in {:?}", metal_config_start.elapsed());
    for instance in instances {
        let encoding_data_start = Instant::now();
        let metal_instance = encode_instances(&instance.points, &instance.scalars, &mut metal_config, None);
        log::debug!("Done encoding data in {:?}", encoding_data_start.elapsed());


        let msm_start = Instant::now();
        let _result: P = exec_metal_commands(&metal_config, metal_instance).unwrap();
        log::debug!("Done msm in {:?}", msm_start.elapsed());
    }
}

fn msm_gpu_par(instances: &Vec<MsmInstance<ArkG, ArkFr>>, target_msm_log_size: Option<usize>) {

    for instance in instances {
        let _ = metal_msm_parallel(instance, target_msm_log_size);
    }
}

fn msm_all_gpu(instances: &Vec<MsmInstance<ArkG, ArkFr>>, batch_size: Option<u32>, threads_per_tg: Option<u32>) {
    let mut metal_config = setup_metal_state();

    for instance in instances {
        let points = &instance.points;
        let scalars = &instance.scalars;

        let _ = metal_msm_all_gpu(points, scalars, &mut metal_config, batch_size, threads_per_tg);
    }
}


fn benchmark_msm(criterion: &mut Criterion) {
    init_logger();

    let mut bench_group = criterion.benchmark_group("benchmark_msm");

    // Set target time and sample size
    bench_group.sample_size(20); // Number of iterations to run
    bench_group.measurement_time(Duration::from_secs(15)); // Total time per benchmark

    let rng = OsRng::default();

    const LOG_INSTANCE_SIZE: u32 = 18;
    const NUM_INSTANCES: u32 = 1;

    let instances = get_or_create_msm_instances::<ArkG, ArkFr>(LOG_INSTANCE_SIZE, NUM_INSTANCES, rng, None).unwrap();
    let instances_h2c = instances
        .iter()
        .map(|instance| {
            (
                instance
                    .points
                    .iter()
                    .map(|p| PointGPU::into::<H2G>(p))
                    .map(|p| p.to_affine())
                    .collect::<Vec<_>>(),
                instance.scalars.iter().map(ScalarGPU::into::<H2Fr>).collect(),
            )
        })
        .collect::<Vec<_>>();
    let instances_ark = instances
        .iter()
        .map(|instance| {
            (
                instance
                    .points
                    .iter()
                    .map(|p| p.into_affine())
                    .collect::<Vec<_>>(),
                instance.scalars.clone(),
            )
        })
        .collect::<Vec<_>>();

    // Benchmark Halo2Curves CPU implementation
    bench_group.bench_function("msm_h2c_cpu", |b| {
        b.iter(|| msm_h2c_cpu(&instances_h2c))
    });

    // Benchmark Arkworks CPU implementation
    bench_group.bench_function("msm_ark_cpu", |b| {
        b.iter(|| msm_ark_cpu(&instances_ark))
    });

    // Benchmark GPU implementation
    bench_group.bench_function("msm_gpu", |b| b.iter(|| msm_gpu(&instances)));

    // Benchmark parallel GPU implementation
    bench_group.bench_function("msm_gpu_par", |b| b.iter(|| msm_gpu_par(&instances, None)));

    // Benchmark all GPU implementation
    bench_group.bench_function("msm_all_gpu", |b| b.iter(|| msm_all_gpu(&instances, None, None)));

    bench_group.finish();
}

fn benchmark_find_optimal_par_par_gpu(criterion: &mut Criterion) {
    init_logger();

    let mut bench_group = criterion.benchmark_group("parameter_benches_par_gpu");

    // Set target time and sample size
    bench_group.sample_size(20); // Number of iterations to run
    bench_group.measurement_time(Duration::from_secs(15)); // Total time per benchmark

    let rng = OsRng::default();

    const LOG_INSTANCE_SIZE: u32 = 16;
    const NUM_INSTANCES: u32 = 5;

    let instances = get_or_create_msm_instances::<ArkG, ArkFr>(LOG_INSTANCE_SIZE, NUM_INSTANCES, rng, None).unwrap();

    let target_msm_sizes = 11..16usize;

    for target_msm_size in target_msm_sizes {
        bench_group.bench_with_input(BenchmarkId::new("msm_gpu_optimal_log_chunk_size", target_msm_size), &target_msm_size, |b, &v| {
            b.iter(|| msm_gpu_par(&instances, Some(v)));
        });
    }

    bench_group.finish();
}

fn benchmark_find_optimal_par_all_gpu(criterion: &mut Criterion) {
    init_logger();

    // Create a benchmark group
    let mut bench_group = criterion.benchmark_group("parameter_benches_all_gpu");

    // Adjust these as you like
    bench_group.sample_size(10); // Number of samples for Criterion
    bench_group.measurement_time(Duration::from_secs(10)); // Time spent collecting data

    let mut rng = OsRng::default();

    // Example range of MSM sizes (number of points/scalars).
    // Adjust as needed: e.g., 2^14, 2^15, 2^16, ...
    let msm_sizes = (18..21u32).step_by(2);

    // Example batch sizes (powers of 2, from 2 to 64)
    let batch_sizes = (0..7).map(|x| 2u32.pow(x)).collect::<Vec<u32>>();

    // Example threads per threadgroup candidates
    // (4 “distant” values, as you suggested)
    let threads_per_tg_values  = [128, 256];

    // For each MSM size, generate or load data
    for msm_size in msm_sizes {
        // Generate or retrieve points and scalars
        let instances = get_or_create_msm_instances::<ArkG, ArkFr>(msm_size, 1, &mut rng, None).unwrap();

        // We’ll reuse the same config for each run, or recreate if needed


        for &batch_size in &batch_sizes {
            for &threads_per_tg in &threads_per_tg_values {

                // If there will be less then 16 threadgroups, the GPU is underutilized
                if 2u32.pow(msm_size) / batch_size / threads_per_tg < 32 {
                    continue;
                }

                bench_group.bench_with_input(BenchmarkId::new(format!("msm_all_gpu/batch_{}__thread_group_{}__msm_{}", batch_size, threads_per_tg, msm_size), 0), &(batch_size, threads_per_tg), |b, &v| {
                    b.iter(|| msm_all_gpu(&instances, Some(v.0), Some(v.1)));
                });
            }
        }
    }

    bench_group.finish();
}


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

// Criterion groups
criterion_group!(general_benches, benchmark_msm);
criterion_group!(parameter_benches_par_gpu, benchmark_find_optimal_par_par_gpu);
criterion_group!(parameter_benches_all_gpu, benchmark_find_optimal_par_all_gpu);

// Criterion main
criterion_main!(general_benches, parameter_benches_par_gpu, parameter_benches_all_gpu);