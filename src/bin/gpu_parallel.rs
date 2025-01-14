use halo2curves::group::Curve;
use mopro_msm::metal::msm::{gpu_with_cpu};
use mopro_msm::metal::abstraction::limbs_conversion::h2c::{
    H2Fr as Fr, H2GAffine as GAffine, H2G as G,
};
use mopro_msm::utils::preprocess::get_or_create_msm_instances;
use rand::{rngs::OsRng, Rng};
use rayon::prelude::*;
use std::env;
use mopro_msm::metal::msm_best;

fn main() {
    // Setup logger
    env_logger::builder().init();

    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();

    // Default value for `log_instance_size`
    let default_log_instance_size: u32 = 20;

    // Parse `log_instance_size` argument (if provided)
    let log_instance_size = args
        .get(1)
        .and_then(|arg| arg.parse::<u32>().ok())
        .unwrap_or(default_log_instance_size);
    log::info!("Log instance size: {}", log_instance_size);

    // Default value for `num_instances`
    let default_num_instances: u32 = 50;

    // Parse the num_instances argument
    let num_instances = args
        .get(2)
        .and_then(|arg| arg.parse::<u32>().ok())
        .unwrap_or(default_num_instances);
    log::info!("Number of instances: {}", num_instances);

    // Parse the mode argument
    let mode = args.get(3).unwrap_or(&"gpu".to_string()).to_lowercase();
    log::info!("Mode: {}", mode);

    // RNG initialization
    let mut rng = OsRng::default();

    let instances =
        get_or_create_msm_instances::<G, Fr>(log_instance_size, num_instances, rng, None).unwrap();

    // Convert points to affine form for MSM processing
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

    // Generate random groups of 1 to 5 instances
    let mut groups = vec![];
    let mut start = 0;
    while start < num_instances as usize {
        let group_size = rng.gen_range(1..=5); // Random size between 1 and 5
        let end = (start + group_size).min(num_instances as usize);
        groups.push(start..end);
        start = end;
    }

    log::info!("Generated {} groups for parallel execution.", groups.len());
    let total_start = instant::Instant::now();

    // Execute each group in parallel
    groups.into_par_iter().for_each(|group_range| {
        for i in group_range {
            let instance = &instances[i];
            let scalars = &instance.scalars;
            let points = &affine_points[i];

            let start_execution = instant::Instant::now();

            // Use gpu_with_cpu to process the group
            match mode.as_str() {
                "gpu" => {
                    let _ = msm_best::<GAffine, GAffine, G, Fr>(scalars, points);
                }
                "cpu" => {
                    let _ = halo2curves::msm::msm_best(&scalars, &points);
                }
                _ => {
                    log::error!("Invalid mode: {}", mode);
                    return;
                }
            }

            log::info!(
                "Group {i}: Total Execution Time: {:?}",
                start_execution.elapsed()
            );
        }
    });

    log::info!("All groups processed.");
    log::info!("Total Execution Time: {:?}", total_start.elapsed());
}