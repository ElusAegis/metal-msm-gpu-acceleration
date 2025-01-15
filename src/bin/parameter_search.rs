use std::env;
use std::time::Instant;
use halo2curves::group::Curve;
use rand::Rng;
use rand::rngs::OsRng;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use mopro_msm::config::{Config, ConfigManager};
use mopro_msm::metal::abstraction::limbs_conversion::h2c::{
    H2Fr as Fr, H2GAffine as GAffine, H2G as G,
};
use mopro_msm::metal::msm::{gpu_msm_h2c, gpu_with_cpu};
use mopro_msm::utils::preprocess::get_or_create_msm_instances;

/// Generate a coarse exponential-ish sequence by multiplying by ~1.5
/// Starting from `start` until `max_value`.
fn generate_exp_sequence(start: u32, max_value: u32) -> Vec<u32> {
    let mut seq = Vec::new();
    let mut current = start;
    while current <= max_value {
        seq.push(current);
        // Multiply by ~1.5 (coarse approximation).
        // For example: 32 -> 48 -> 72 -> 108 -> 162 -> 243 -> ...
        current = ((current as f64) * 1.5).ceil() as u32;
    }
    seq
}

/// Benchmark function that uses the loaded `instances` and `affine_points`.
/// Returns the average single-instance time in milliseconds.
/// In a real scenario, you'd use `config.cpu_gpu_split_ratio`, `config.bucket_size_16`, etc.
fn measure_performance(
    config: &Config,
    instances: &[mopro_msm::utils::preprocess::MsmInstance<G, Fr>],
    affine_points: &[Vec<GAffine>],
    mode: &str,         // "cpu", "gpu", or "cpu_gpu"
    retries: u32,       // how many times to repeat
) -> f64 {
    let num_instances = instances.len();
    let mut rng = OsRng::default();

    // We store the average times from each retry
    let mut retry_times = Vec::with_capacity(retries as usize);

    // Example usage of config parameters (logging them here for illustration).
    // In practice, you'd pass them deeper into your CPU/GPU code.
    ConfigManager::default().save_config(config);
    log::info!("[CONFIG] Using config: {:?}", config);

    for retry in 0..retries {
        log::debug!("Starting retry {}/{}", retry + 1, retries);

        // Example: you might configure CPU/GPU split logic or bucket sizing here
        // e.g., set global or thread-local parameters, etc.

        let total_start = Instant::now();

        // Generate random groups for parallel execution
        let mut groups = Vec::new();
        let mut start = 0;
        while start < num_instances {
            // For demonstration, group_size is 1..=7
            let group_size = rng.gen_range(1..=7);
            let end = (start + group_size).min(num_instances);
            groups.push(start..end);
            start = end;
        }

        // In parallel, run MSM on each group
        groups.into_par_iter().for_each(|group_range| {
            for i in group_range {
                let instance = &instances[i];
                let scalars = &instance.scalars;
                let points = &affine_points[i];

                let start_execution = Instant::now();

                match mode {
                    "gpu" => {
                        // GPU-only
                        let _ = gpu_msm_h2c::<GAffine, GAffine, G, Fr>(scalars, points);
                    }
                    "cpu_gpu" => {
                        // Hybrid CPU/GPU
                        // (in practice you'd incorporate `config.cpu_gpu_split_ratio`).
                        let _ = gpu_with_cpu::<GAffine, GAffine, G, Fr>(scalars, points);
                    }
                    "cpu" => {
                        // CPU-only
                        let _ = halo2curves::msm::msm_best(scalars, points);
                    }
                    _ => {
                        log::error!("Invalid mode: {}. Falling back to cpu...", mode);
                        let _ = halo2curves::msm::msm_best(scalars, points);
                    }
                }

                let elapsed_ms = start_execution.elapsed().as_millis();
                // Printing each group time for debugging
                log::debug!("Group {i}: Execution Time: {} ms", elapsed_ms);
            }
        });

        let elapsed_time = total_start.elapsed();
        // Convert total time to seconds
        retry_times.push(elapsed_time.as_secs_f64());

        log::debug!(
            "Retry {}/{} processed. Execution Time: {:?}",
            retry + 1,
            retries,
            elapsed_time
        );
    }

    // Average total time over retries
    let total_avg_sec = retry_times.iter().sum::<f64>() / retries as f64;
    // Convert to ms
    let total_avg_ms = total_avg_sec * 1000.0;
    // Single-instance average (ms)
    let avg_instance_ms = total_avg_ms / (instances.len() as f64);

    log::info!(
        "  [RESULT] Average Total Execution Time: {:.0} ms",
        total_avg_ms
    );
    log::info!(
        "  [RESULT] Average Single Instance Execution Time: {:.0} ms",
        avg_instance_ms
    );

    avg_instance_ms
}

fn main() {
    // Setup logger
    env_logger::builder().init();

    // Parse command-line arguments for instance setup
    let args: Vec<String> = env::args().collect();

    // Default values
    let default_log_instance_size: u32 = 20;
    let default_num_instances: u32 = 50;
    let default_retries: u32 = 3; // For each test run

    // Parse log_instance_size, num_instances, and optional 'retries'
    let log_instance_size = args
        .get(1)
        .and_then(|arg| arg.parse::<u32>().ok())
        .unwrap_or(default_log_instance_size);
    let num_instances = args
        .get(2)
        .and_then(|arg| arg.parse::<u32>().ok())
        .unwrap_or(default_num_instances);
    let mode = args.get(3).unwrap_or(&"cpu_gpu".to_string()).to_lowercase();
    let retries = args
        .get(4)
        .and_then(|arg| arg.parse::<u32>().ok())
        .unwrap_or(default_retries);

    log::info!("Log instance size: {}", log_instance_size);
    log::info!("Number of instances: {}", num_instances);
    log::info!("Mode: {}", mode);
    log::info!("Retries: {}", retries);

    // ---- Load Instances Once ----
    let rng = OsRng::default();
    let instances =
        get_or_create_msm_instances::<G, Fr>(log_instance_size, num_instances, rng, None)
            .expect("Failed to get MSM instances");

    let affine_points: Vec<Vec<GAffine>> = instances
        .iter()
        .map(|instance| {
            instance
                .points
                .iter()
                .map(|p| p.to_affine())
                .collect::<Vec<GAffine>>()
        })
        .collect();

    log::info!("Instances loaded. Starting approximate parameter search...");

    // ---- Define Our Parameter Search Space ----
    let cpu_gpu_split_candidates: Vec<f64> = (5..100u32)
        .step_by(5)
        .map(|x| x as f64 / 100.0)
        .collect::<Vec<_>>();

    let bucket_size_16_candidates: Vec<u32> = (5..=15u32).collect();

    // Generate an exponential-like sequence for desired_pairs_per_thread
    // e.g. 32, 48, 72, 108, 162, ...
    let desired_pairs_candidates = generate_exp_sequence(32, 32768);

    // Similarly for buckets_per_threadgroup
    let buckets_candidates = generate_exp_sequence(32, 32768);

    // For min_max_threads, just a small range
    let min_max_threads_candidates = vec![32, 48, 64, 80];

    let mut best_config = Config {
        cpu_gpu_split_ratio: 0.5,
        bucket_size_16: 10,
        desired_pairs_per_thread: 512,
        buckets_per_threadgroup: 1024,
        min_max_threads: 32,
    };
    let mut best_time = f64::MAX;

    // ---- Approximate Search ----
    // We'll do a nested search but break early when we find < 60 ms.
    // Because the search can become large, we keep it short-circuited.

    'outer: for &cpu_gpu_split_ratio in &cpu_gpu_split_candidates {
        for &bucket_size_16 in &bucket_size_16_candidates {
            for &desired_pairs_per_thread in &desired_pairs_candidates {
                for &buckets_per_threadgroup in &buckets_candidates {
                    for &min_max_threads in &min_max_threads_candidates {
                        let config = Config {
                            cpu_gpu_split_ratio,
                            bucket_size_16,
                            desired_pairs_per_thread,
                            buckets_per_threadgroup,
                            min_max_threads,
                        };

                        // ---- Benchmark with this config ----
                        let avg_instance_time_ms = measure_performance(
                            &config,
                            &instances,
                            &affine_points,
                            &mode,
                            retries,
                        );

                        // Update the best config if improved
                        if avg_instance_time_ms < best_time {
                            best_time = avg_instance_time_ms;
                            best_config = config.clone();
                            log::warn!(
                                "\n[UPDATE] Found better config! Avg time = {:.0} ms\nConfig = {:?}",
                                best_time, best_config
                            );
                        }

                        // If below 60 ms, stop searching
                        if best_time < 60.0 {
                            log::warn!("\n[STOP] Achieved <60 ms average. Stopping early.");
                            break 'outer;
                        }
                    }
                }
            }
        }
    }

    println!("\n--- SEARCH COMPLETE ---");
    println!("Best average single-instance time: {:.0} ms", best_time);
    println!("Best config: {:?}", best_config);
}