use halo2curves::bn256::{Fr, G1Affine};
use crate::metal::abstraction::limbs_conversion::h2c::{H2Fr, H2GAffine, H2G};
use halo2curves::ff::Field;
use instant::Instant;

#[uniffi::export]
fn benchmark_h2c_metal_and_cpu_msm_best(
    log_size: u32,
) -> u64 {
    let (points, scalars) = prepare_instance(log_size);

    let start = Instant::now();
    crate::metal::msm_best::<H2GAffine, H2GAffine, H2G, H2Fr>(&scalars, &points);
    let duration = start.elapsed();

    return duration.as_millis() as u64;
}

#[uniffi::export]
fn benchmark_h2c_cpu_msm_best(
    log_size: u32,
) -> u64 {
    let (points, scalars) = prepare_instance(log_size);

    let start = Instant::now();
    halo2curves::msm::msm_best(&scalars, &points);
    let duration = start.elapsed();

    return duration.as_millis() as u64;
}

fn prepare_instance(log_size: u32) -> (Vec<G1Affine>, Vec<Fr>) {
    let mut rng = rand::thread_rng();
    let instance_size = 1 << log_size;

    (0..instance_size)
        .into_iter()
        .map(|_| {
            let point = H2GAffine::random(&mut rng);
            let scalar = H2Fr::random(&mut rng);
            (point, scalar)
        })
        .unzip()
}