use halo2curves::group::Curve;
use mopro_msm::metal::msm::{gpu_msm_h2c, gpu_with_cpu};
use mopro_msm::metal::abstraction::limbs_conversion::h2c::{
    H2Fr as Fr, H2GAffine as GAffine, H2G as G,
};
use mopro_msm::utils::preprocess::get_or_create_msm_instances;
use rand::{rngs::OsRng, Rng};
use rayon::prelude::*;
use std::io::{self, Write};
use std::sync::Mutex;

fn main() {
    // Setup logger
    env_logger::builder().init();

    // Parse command-line arguments for initial setup
    let args: Vec<String> = std::env::args().collect();

    // Default values
    let default_log_instance_size: u32 = 20;
    let default_num_instances: u32 = 50;

    // Parse `log_instance_size` and `num_instances`
    let log_instance_size = args
        .get(1)
        .and_then(|arg| arg.parse::<u32>().ok())
        .unwrap_or(default_log_instance_size);
    let num_instances = args
        .get(2)
        .and_then(|arg| arg.parse::<u32>().ok())
        .unwrap_or(default_num_instances);

    log::info!("Log instance size: {}", log_instance_size);
    log::info!("Number of instances: {}", num_instances);

    // RNG initialization
    let mut rng = OsRng::default();

    // Load instances and preprocess them
    let instances =
        get_or_create_msm_instances::<G, Fr>(log_instance_size, num_instances, rng, None).unwrap();
    let affine_points: Vec<Vec<_>> = instances
        .iter()
        .map(|instance| {
            instance
                .points
                .iter()
                .map(|p| p.to_affine())
                .collect::<Vec<_>>()
        })
        .collect();

    log::info!("Instances loaded. Ready for testing.");

    // Interactive testing loop
    loop {
        print!("Enter testing parameters (e.g., 'gpu 2' or 'exit' to quit): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        // Exit condition
        if input.eq_ignore_ascii_case("exit") {
            log::info!("Exiting program.");
            break;
        }

        // Parse input for mode and retries
        let mut parts = input.split_whitespace();
        let mode = parts.next().unwrap_or("gpu").to_lowercase();
        let retries: u32 = parts
            .next()
            .and_then(|arg| arg.parse::<u32>().ok())
            .unwrap_or(1);

        log::info!("Mode: {}", mode);
        log::info!("Number of retries: {}", retries);

        let mut retry_times = vec![];

        for retry in 0..retries {
            log::info!("Starting retry {}/{}", retry + 1, retries);

            let total_start = instant::Instant::now();

            // Generate random groups for parallel execution
            let mut groups = vec![];
            let mut start = 0;
            while start < num_instances as usize {
                let group_size = rng.gen_range(1..=2);
                let end = (start + group_size).min(num_instances as usize);
                groups.push(start..end);
                start = end;
            }

            groups.into_par_iter().for_each(|group_range| {
                for i in group_range.clone() {
                    let instance = &instances[i];
                    let scalars = &instance.scalars;
                    let points = &affine_points[i];

                    let start_execution = instant::Instant::now();

                    match mode.as_str() {
                        "gpu" => {
                            let _ = gpu_msm_h2c::<GAffine, GAffine, G, Fr>(scalars, points);
                        }
                        "cpu_gpu" => {
                            let _ = gpu_with_cpu::<GAffine, GAffine, G, Fr>(scalars, points);
                        }
                        "cpu" => {
                            let _ = halo2curves::msm::msm_best(&scalars, &points);
                        }
                        _ => {
                            log::error!("Invalid mode: {}", mode);
                            return;
                        }
                    }

                    let elapsed_ms =
                        start_execution.elapsed().as_millis();
                    log::info!("Group {i}: Execution Time: {} ms", elapsed_ms);
                }
            });

            let elapsed_time = total_start.elapsed();
            retry_times.push(elapsed_time.as_secs_f64());
            log::info!(
                "Retry {}/{} processed. Execution Time: {:?}",
                retry + 1,
                retries,
                elapsed_time
            );
        }

        let total_avg_time: f64 = retry_times.iter().sum::<f64>() / retries as f64;
        let avg_instance_time = (total_avg_time * 1000.0) / num_instances as f64;

        log::info!("Average Total Execution Time: {:.0} ms", total_avg_time * 1000.0);
        log::info!(
            "Average Single Instance Execution Time: {:.0} ms",
            avg_instance_time
        );
    }
}