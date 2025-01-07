use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::SeedableRng;
use rand::rngs::StdRng;
use mopro_msm::metal::abstraction::limbs_conversion::{PointGPU, ScalarGPU, ToLimbs};

// Size of the benchmark dataset (2^20)
const LOG_SIZE: usize = 20;
const SIZE: usize = 1 << LOG_SIZE;

fn benchmark_to_u32_limbs(c: &mut Criterion) {
    // Initialize the random number generator with a fixed seed for reproducibility
    let seed: u64 = 42;
    let mut rng = StdRng::seed_from_u64(seed);


    // Create a benchmark group
    let mut group = c.benchmark_group("to_u32_limbs");

    #[cfg(feature = "ark")]
    {
        use mopro_msm::metal::abstraction::limbs_conversion::ark::{ArkFr, ArkG, ArkGAffine};
        use ark_ec::CurveGroup;

        const ARK_FR_LIMBS: usize = 8;
        const ARK_G_LIMBS: usize = 24;

        // Generate random instances for each type
        let ark_fr_instances: Vec<ArkFr> = (0..SIZE).map(|_| ArkFr::random(&mut rng)).collect();
        let ark_g_instances: Vec<ArkG> = (0..SIZE).map(|_| ArkG::random(&mut rng)).collect();
        let ark_g_affine_instances: Vec<ArkGAffine> = ark_g_instances.iter().map(|g| g.into_affine()).collect();


        // Benchmark ArkFr
        {
            let mut ark_fr_limbs = vec![0u32; ark_fr_instances.len() * ARK_FR_LIMBS];
            group.throughput(Throughput::Elements(SIZE as u64));

            group.bench_function(BenchmarkId::new("ArkFr", SIZE), |b| {
                b.iter(|| {
                    ark_fr_instances.as_slice().write_u32_limbs(ark_fr_limbs.as_mut_slice());
                });
            });
        }

        // Benchmark ArkG
        {
            let mut ark_g_limbs = vec![0u32; ark_g_instances.len() * ARK_G_LIMBS];
            group.throughput(Throughput::Elements(SIZE as u64));

            group.bench_function(BenchmarkId::new("ArkG", SIZE), |b| {
                b.iter(|| {
                    ark_g_instances.as_slice().write_u32_limbs(ark_g_limbs.as_mut_slice());
                });
            });
        }

        // Benchmark ArkGAffine
        {
            let mut ark_g_affine_limbs = vec![0u32; ark_g_affine_instances.len() * ARK_G_LIMBS];
            group.throughput(Throughput::Elements(SIZE as u64));

            group.bench_function(BenchmarkId::new("ArkGAffine", SIZE), |b| {
                b.iter(|| {
                    ark_g_affine_instances.as_slice().write_u32_limbs(ark_g_affine_limbs.as_mut_slice());
                });
            });
        }
    }

    #[cfg(feature = "h2c")]
    {
        use mopro_msm::metal::abstraction::limbs_conversion::h2c::{H2Fr, H2GAffine, H2G};
        use halo2curves::group::Curve;

        const H2_FR_LIMBS: usize = 8;
        const H2_G_LIMBS: usize = 24;

        let h2_fr_instances: Vec<H2Fr> = (0..SIZE).map(|_| H2Fr::random(&mut rng)).collect();
        let h2_g_instances: Vec<H2G> = (0..SIZE).map(|_| H2G::random(&mut rng)).collect();
        let h2_g_affine_instances: Vec<H2GAffine> = h2_g_instances.iter().map(|g| g.to_affine()).collect();

        // Benchmark H2Fr
        {
            let mut h2_fr_limbs = vec![0u32; h2_fr_instances.len() * H2_FR_LIMBS];
            group.throughput(Throughput::Elements(SIZE as u64));

            group.bench_function(BenchmarkId::new("H2Fr", SIZE), |b| {
                b.iter(|| {
                    h2_fr_instances.as_slice().write_u32_limbs(h2_fr_limbs.as_mut_slice());
                });
            });
        }

        // Benchmark H2G
        {
            let mut h2_g_limbs = vec![0u32; h2_g_instances.len() * H2_G_LIMBS];
            group.throughput(Throughput::Elements(SIZE as u64));

            group.bench_function(BenchmarkId::new("H2G", SIZE), |b| {
                b.iter(|| {
                    h2_g_instances.as_slice().write_u32_limbs(h2_g_limbs.as_mut_slice());
                });
            });
        }

        // Benchmark H2GAffine
        {
            let mut h2_g_affine_limbs = vec![0u32; h2_g_affine_instances.len() * H2_G_LIMBS];
            group.throughput(Throughput::Elements(SIZE as u64));

            group.bench_function(BenchmarkId::new("H2GAffine", SIZE), |b| {
                b.iter(|| {
                    h2_g_affine_instances.as_slice().write_u32_limbs(h2_g_affine_limbs.as_mut_slice());
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_to_u32_limbs);
criterion_main!(benches);