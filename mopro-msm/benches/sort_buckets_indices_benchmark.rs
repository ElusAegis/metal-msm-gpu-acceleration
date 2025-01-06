use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use mopro_msm::metal::msm::{setup_metal_state, sort_buckets};
use mopro_msm::metal::msm::sort_buckets::sort_buckets_indices;

fn benchmark_sort_buckets_indices(c: &mut Criterion) {
    // Define the different lengths based on the modes
    let modes = [
        16,
        18,
        20,
        22,
    ];

    // Initialize the Criterion benchmark group
    let mut group = c.benchmark_group("sort_buckets_indices");

    for &log_length in &modes {
        // Setup for each benchmark iteration
        let length = 2usize.pow(log_length) * 17;
        group.throughput(Throughput::Elements(length as u64));

        group.bench_with_input(BenchmarkId::from_parameter(length), &length, |b, &len| {
            // Setup configuration
            let mut config = setup_metal_state();

            // Initialize random number generator with a fixed seed for consistency
            let seed = 42u64; // You can vary this if needed
            let mut rng = StdRng::seed_from_u64(seed);

            // Generate random data
            let mut data = Vec::with_capacity(len * 2);
            for _ in 0..len {
                data.push(rng.gen::<u32>() % (len as u32 * 16));
                data.push(rng.gen::<u32>() % len as u32);
            }

            // Create test instance
            let instance = sort_buckets::create_test_instance(&mut config, data.clone());

            // Precompute any necessary data outside the benchmarked closure

            // Benchmark the sort_buckets_indices function
            b.iter(|| {
                // It's assumed that sort_buckets_indices does not mutate config or instance.
                // If it does, you may need to clone them here.
                sort_buckets_indices(&config, &instance)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, benchmark_sort_buckets_indices);
criterion_main!(benches);