#![cfg(all(feature = "ark", feature = "h2c"))]

use std::time::Duration;
use ark_ec::{CurveGroup, VariableBaseMSM};
use criterion::{criterion_group, criterion_main, Criterion};
use halo2curves::group::Curve;
use halo2curves::msm::msm_best;
use instant::Instant;
use rand::rngs::OsRng;
use mopro_msm::msm::metal::abstraction::limbs_conversion::ark::{ArkFr, ArkG, ArkGAffine};
use mopro_msm::msm::metal::abstraction::limbs_conversion::h2c::{H2Fr, H2GAffine, H2G};
use mopro_msm::msm::metal::abstraction::limbs_conversion::{PointGPU, ScalarGPU};
use mopro_msm::msm::metal::msm::{encode_instances, exec_metal_commands, setup_metal_state};
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

fn msm_gpu<P: PointGPU, S: ScalarGPU>(instances: &Vec<MsmInstance<P, S>>) {

    let metal_config_start = Instant::now();
    let mut metal_config = setup_metal_state();
    log::debug!("Done setting up metal state in {:?}", metal_config_start.elapsed());
    for instance in instances {
        let encoding_data_start = Instant::now();
        let metal_instance = encode_instances(&instance.points, &instance.scalars, &mut metal_config);
        log::debug!("Done encoding data in {:?}", encoding_data_start.elapsed());


        let msm_start = Instant::now();
        let _result: P = exec_metal_commands(&metal_config, metal_instance).unwrap();
        log::debug!("Done msm in {:?}", msm_start.elapsed());
    }
}

fn benchmark_msm(criterion: &mut Criterion) {
    env_logger::builder().is_test(true).init();

    let mut bench_group = criterion.benchmark_group("msm");

    // Set target time and sample size
    bench_group.sample_size(25); // Number of iterations to run
    bench_group.measurement_time(Duration::from_secs(30)); // Total time per benchmark

    let rng = OsRng::default();

    let instances = get_or_create_msm_instances::<ArkG, ArkFr>(16, 5, rng, None).unwrap();
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

    bench_group.finish();
}
// Criterion group and main function
criterion_group!(benches, benchmark_msm);
criterion_main!(benches);