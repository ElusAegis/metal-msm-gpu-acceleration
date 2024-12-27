use std::{env};
use rand::rngs::OsRng;
use mopro_msm::msm::metal::abstraction::limbs_conversion::ark::{ArkFr, ArkG};
use mopro_msm::msm::metal::msm::{encode_instances, exec_metal_commands, metal_msm_parallel, setup_metal_state};
use mopro_msm::msm::utils::preprocess::get_or_create_msm_instances;

fn main() {
    // Setup logger
    env_logger::builder().init();
    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();

    // Default values
    let default_log_instance_size: u32 = 16;
    let default_target_msm_log_size: usize = 13;

    // Parse `log_instance_size` argument (if provided)
    let log_instance_size = args.get(1)
        .and_then(|arg| arg.parse::<u32>().ok())
        .unwrap_or(default_log_instance_size);
    log::info!("Log instance size: {}", log_instance_size);

    // Parse `target_msm_log_size` argument (if provided)
    let target_msm_log_size = args.get(2)
        .and_then(|arg| arg.parse::<usize>().ok())
        .unwrap_or(default_target_msm_log_size);
    log::info!("Target MSM log size: {}", target_msm_log_size);

    // Check for `RUN_PARALLEL` environment variable
    let run_parallel = env::var("RUN_PARALLEL")
        .map(|val| val == "true" || val == "1")
        .unwrap_or(false);
    log::info!("Running in parallel mode: {}", run_parallel);

    // RNG initialization
    let rng = OsRng::default();

    // Generate or retrieve MSM instances
    const NUM_INSTANCES: u32 = 1;
    let instances = get_or_create_msm_instances::<ArkG, ArkFr>(log_instance_size, NUM_INSTANCES, rng, None).unwrap();

    // Initialize Metal configuration
    let mut metal_config = setup_metal_state();

    let start_execution = instant::Instant::now();

    // Process MSM instances
    for instance in instances {
        if run_parallel {
            let _ = metal_msm_parallel(&instance, target_msm_log_size);
        } else {
            let metal_instance = encode_instances(&instance.points, &instance.scalars, &mut metal_config);
            let _res: ArkG = exec_metal_commands(&metal_config, metal_instance).unwrap();
        }
    }

    log::info!("Total Execution Time: {:?}", start_execution.elapsed());
}