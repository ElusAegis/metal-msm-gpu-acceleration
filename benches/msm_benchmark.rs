#[cfg(feature = "ark")]
use ark_ec::{CurveGroup, VariableBaseMSM};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
#[cfg(feature = "h2c")]
use halo2curves::group::Curve;
#[cfg(feature = "ark")]
use mopro_msm::metal::abstraction::limbs_conversion::ark::{ArkFr as Fr, ArkG as G};
#[cfg(feature = "ark")]
use mopro_msm::metal::abstraction::limbs_conversion::ark::{ArkFr, ArkG, ArkGAffine};
#[cfg(feature = "h2c")]
use mopro_msm::metal::abstraction::limbs_conversion::h2c::{H2Fr, H2GAffine, H2G};
#[cfg(all(feature = "h2c", not(feature = "ark")))]
use mopro_msm::metal::abstraction::limbs_conversion::h2c::{H2Fr as Fr, H2G as G};
use mopro_msm::metal::abstraction::limbs_conversion::{PointGPU, ScalarGPU};
use mopro_msm::metal::msm::{metal_msm_parallel, setup_metal_state};
use mopro_msm::utils::preprocess::{get_or_create_msm_instances, MsmInstance};
use rand::rngs::OsRng;
use std::ops::Add;
use std::time::Duration;

#[cfg(feature = "h2c")]
pub fn msm_h2c_cpu(instances: &Vec<(Vec<H2GAffine>, Vec<H2Fr>)>) {
    for instance in instances {
        let _ = halo2curves::msm::msm_best(&instance.1, &instance.0);
    }
}

#[cfg(feature = "h2c")]
pub fn msm_h2c_gpu_best(instances: &Vec<(Vec<H2GAffine>, Vec<H2Fr>)>) {
    for instance in instances {
        let _ =
            mopro_msm::metal::msm_best::<H2GAffine, H2GAffine, H2G, H2Fr>(&instance.1, &instance.0);
    }
}

#[cfg(feature = "ark")]
pub fn msm_ark_cpu(instances: &Vec<(Vec<ArkGAffine>, Vec<ArkFr>)>) {
    for instance in instances {
        let _ = ArkG::msm(&instance.0, &instance.1).unwrap();
    }
}

#[cfg(feature = "ark")]
fn msm_gpu<P: PointGPU<24> + Sync, S: ScalarGPU<8> + Sync>(instances: &Vec<MsmInstance<P, S>>) {
    let mut metal_config = setup_metal_state();
    for instance in instances {
        let _result: P = mopro_msm::metal::msm::metal_msm(
            &instance.points,
            &instance.scalars,
            &mut metal_config,
        )
        .unwrap();
    }
}

fn msm_gpu_par<P, S>(instances: &Vec<MsmInstance<P, S>>, target_msm_log_size: Option<usize>)
where
    P: PointGPU<24> + Add<P, Output = P> + Send + Sync + Clone,
    S: ScalarGPU<8> + Send + Sync,
{
    for instance in instances {
        let _ = metal_msm_parallel(instance, target_msm_log_size);
    }
}

fn benchmark_msm(criterion: &mut Criterion) {
    init_logger();

    let mut bench_group = criterion.benchmark_group("benchmark_msm");

    // Set target time and sample size
    bench_group.sample_size(20); // Number of iterations to run
    bench_group.measurement_time(Duration::from_secs(15)); // Total time per benchmark

    let rng = OsRng;

    const LOG_INSTANCE_SIZE: u32 = 20;
    const NUM_INSTANCES: u32 = 5;

    let instances =
        get_or_create_msm_instances::<G, Fr>(LOG_INSTANCE_SIZE, NUM_INSTANCES, rng, None).unwrap();
    #[cfg(feature = "h2c")]
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
                instance
                    .scalars
                    .iter()
                    .map(ScalarGPU::into::<H2Fr>)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    #[cfg(feature = "ark")]
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
    #[cfg(feature = "h2c")]
    bench_group.bench_function("msm_h2c_cpu", |b| b.iter(|| msm_h2c_cpu(&instances_h2c)));

    // Benchmark Arkworks CPU implementation
    #[cfg(feature = "ark")]
    bench_group.bench_function("msm_ark_cpu", |b| b.iter(|| msm_ark_cpu(&instances_ark)));

    // Benchmark Best Halo2Curves GPU implementation
    #[cfg(feature = "h2c")]
    bench_group.bench_function("msm_h2c_gpu_best", |b| {
        b.iter(|| msm_h2c_gpu_best(&instances_h2c))
    });

    // Benchmark GPU implementation
    #[cfg(feature = "ark")]
    bench_group.bench_function("msm_gpu", |b| b.iter(|| msm_gpu(&instances)));

    // Benchmark parallel GPU implementation
    #[cfg(feature = "ark")]
    bench_group.bench_function("msm_gpu_par", |b| b.iter(|| msm_gpu_par(&instances, None)));

    bench_group.finish();
}

fn benchmark_find_optimal_par_par_gpu(criterion: &mut Criterion) {
    init_logger();

    let mut bench_group = criterion.benchmark_group("parameter_benches_par_gpu");

    // Set target time and sample size
    bench_group.sample_size(20); // Number of iterations to run
    bench_group.measurement_time(Duration::from_secs(15)); // Total time per benchmark

    let rng = OsRng;

    const LOG_INSTANCE_SIZE: u32 = 16;
    const NUM_INSTANCES: u32 = 5;

    let instances =
        get_or_create_msm_instances::<G, Fr>(LOG_INSTANCE_SIZE, NUM_INSTANCES, rng, None).unwrap();

    let target_msm_sizes = 11..16usize;

    for target_msm_size in target_msm_sizes {
        bench_group.bench_with_input(
            BenchmarkId::new("msm_gpu_optimal_log_chunk_size", target_msm_size),
            &target_msm_size,
            |b, &v| {
                b.iter(|| msm_gpu_par(&instances, Some(v)));
            },
        );
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
criterion_group!(
    parameter_benches_par_gpu,
    benchmark_find_optimal_par_par_gpu
);

// Criterion main
criterion_main!(general_benches, parameter_benches_par_gpu);
