#[cfg(all(feature = "ark", not(feature = "h2c")))]
use ark_ec::{CurveGroup, VariableBaseMSM};
#[cfg(feature = "h2c")]
use halo2curves::group::Curve;

#[cfg(all(feature = "ark", not(feature = "h2c")))]
use mopro_msm::metal::abstraction::limbs_conversion::ark::{ArkFr as Fr, ArkG as G};
#[cfg(feature = "h2c")]
use mopro_msm::metal::abstraction::limbs_conversion::h2c::{
    H2Fr as Fr, H2GAffine as GAffine, H2G as G,
};

#[cfg(feature = "h2c")]
use mopro_msm::metal::msm::{gpu_msm_h2c, gpu_with_cpu};
#[cfg(all(feature = "ark", not(feature = "h2c")))]
use mopro_msm::metal::msm::{metal_msm, setup_metal_state};
use mopro_msm::utils::preprocess::{get_or_create_msm_instances, MsmInstance};

use rand::rngs::OsRng;
use std::env;
use std::time::Duration;
use halo2curves::bn256::{G1Affine, G1};
use rand::{thread_rng, Rng};
// --- Add this (or equivalent) to run parallel chunks ---
use rayon::prelude::*;

fn main() {
    // Setup logger
    env_logger::builder().init();

    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();

    // Default value for `log_instance_size`
    let default_log_instance_size: u32 = 16;

    // Parse `log_instance_size` argument (if provided)
    let log_instance_size = args
        .get(1)
        .and_then(|arg| arg.parse::<u32>().ok())
        .unwrap_or(default_log_instance_size);
    log::info!("Log instance size: {}", log_instance_size);

    // Default value for `num_instances`
    let default_num_instances: u32 = 1;

    // Parse `num_instances` argument (if provided)
    let num_instances = args
        .get(2)
        .and_then(|arg| arg.parse::<u32>().ok())
        .unwrap_or(default_num_instances);
    log::info!("Number of instances: {}", num_instances);

    // Parse `RUN_MODE` argument
    let default_run_mode = "gpu".to_string();
    let run_mode = args.get(3).unwrap_or(&default_run_mode).to_lowercase();
    log::info!("Run mode: {}", run_mode);

    // Default value for `retries`
    let default_retries: u32 = 3;
    let retries = args
        .get(4)
        .and_then(|arg| arg.parse::<u32>().ok())
        .unwrap_or(default_retries);
    log::info!("Retries: {}", retries);

    // Default value for parallel runs
    let default_parallel_runs: bool = false;
    let parallel_runs = args
        .get(5)
        .and_then(|arg| arg.parse::<bool>().ok())
        .unwrap_or(default_parallel_runs);
    log::info!("Parallel runs: {}", parallel_runs);

    // RNG initialization
    let rng = OsRng::default();

    // Load MSM instances
    let instances = get_or_create_msm_instances::<G, Fr>(log_instance_size, num_instances, rng, None)
        .expect("Failed to get MSM instances");

    // Precompute affine points for CPU-based runs
    let affine_points = {
        #[cfg(all(feature = "ark", not(feature = "h2c")))]
        {
            instances[0]
                .points
                .iter()
                .map(|p| p.into_affine())
                .collect::<Vec<_>>()
        }
        #[cfg(feature = "h2c")]
        {
            instances[0]
                .points
                .iter()
                .map(|p| p.to_affine())
                .collect::<Vec<_>>()
        }
    };

    let start_execution = instant::Instant::now();

    // Main benchmarking loop
    for _ in 0..retries {
        if !parallel_runs {
            // ---- SEQUENTIAL RUNS ----
            for instance in &instances {
                run_selected_msm(&run_mode, &affine_points, &instance);
            }
        } else {
            // ---- PARALLEL RUNS: RANDOM CHUNK SIZES (up to 10), each MSM with random delay up to 5s ----
            let mut rng = thread_rng();

            // We'll iterate over `instances` in "random-sized" chunks of up to 10
            let mut i = 0;
            while i < instances.len() {
                // Random chunk size between 1 and 10
                let chunk_size = rng.gen_range(1..=10);
                let end = (i + chunk_size).min(instances.len());
                let chunk = &instances[i..end];
                i = end;

                // Process each chunk in parallel
                chunk.par_iter().enumerate().for_each(|(i, instance)| {
                    // Each MSM in this chunk sleeps for a random offset (0..=5s) before starting
                    let mut rng = thread_rng();
                    let offset_ms = rng.gen_range(0..=(chunk_size as u64 * 500_u64));
                    if i > 0 {
                        std::thread::sleep(Duration::from_millis(offset_ms));
                    }

                    run_selected_msm(&run_mode, &affine_points, &instance);
                });
            }
        }
    }

    let total_time = start_execution.elapsed();
    log::info!("Total Execution Time: {:?}", total_time);
    log::info!(
        "Average Instance Execution Time: {:?}",
        total_time / num_instances / retries
    );
}

fn run_selected_msm(run_mode: &String, affine_points: &Vec<G1Affine>, instance: &MsmInstance<G1, Fr>) {
    // Run the MSM
    match run_mode.as_str() {
        #[cfg(all(feature = "ark", not(feature = "h2c")))]
        "gpu" => {
            let config_time = instant::Instant::now();
            let mut metal_config = setup_metal_state();
            log::debug!("Config Setup Time: {:?}", config_time.elapsed());

            let _ = metal_msm(&instance.points, &instance.scalars, &mut metal_config)
                .expect("Metal MSM failed");
        }
        #[cfg(feature = "h2c")]
        "gpu" => {
            let _ = gpu_msm_h2c::<GAffine, GAffine, G, Fr>(
                &instance.scalars,
                &affine_points,
            );
        }
        #[cfg(feature = "h2c")]
        "gpu_cpu" => {
            let _ = gpu_with_cpu::<GAffine, GAffine, G, Fr>(
                &instance.scalars,
                &affine_points,
            );
        }
        #[cfg(all(feature = "ark", not(feature = "h2c")))]
        "cpu" => {
            let _ = G::msm(&affine_points, &instance.scalars)
                .expect("CPU MSM failed");
        }
        #[cfg(feature = "h2c")]
        "cpu" => {
            let _ = halo2curves::msm::msm_best(&instance.scalars, &affine_points);
        }
        #[cfg(feature = "h2c")]
        "check" => {
            let res1 = gpu_with_cpu::<GAffine, GAffine, G, Fr>(
                &instance.scalars,
                &affine_points,
            )
                .to_affine();
            let res2 =
                halo2curves::msm::msm_best(&instance.scalars, &affine_points)
                    .to_affine();
            assert_eq!(res1, res2);
        }
        _ => {
            log::error!("Invalid RUN_MODE: {}", run_mode);
            std::process::exit(1);
        }
    }
}