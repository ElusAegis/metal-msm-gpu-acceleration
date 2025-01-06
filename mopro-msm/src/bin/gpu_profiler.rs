use std::env;
use ark_ec::{CurveGroup, VariableBaseMSM};
use rand::rngs::OsRng;
use mopro_msm::msm::metal::abstraction::limbs_conversion::ark::{ArkFr, ArkG};
use mopro_msm::msm::metal::msm::{metal_msm_parallel, metal_msm_all_gpu, setup_metal_state, metal_msm};
use mopro_msm::msm::utils::preprocess::get_or_create_msm_instances;

fn main() {
    // Setup logger
    env_logger::builder().init();

    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();

    // Default value for `log_instance_size`
    let default_log_instance_size: u32 = 16;

    // Parse `log_instance_size` argument (if provided)
    let log_instance_size = args.get(1)
        .and_then(|arg| arg.parse::<u32>().ok())
        .unwrap_or(default_log_instance_size);
    log::info!("Log instance size: {}", log_instance_size);

    // Parse `RUN_MODE` argument
    let run_mode = args.get(2).unwrap_or(&"gpu".to_string()).to_lowercase();
    log::info!("Run mode: {}", run_mode);


    // RNG initialization
    let rng = OsRng::default();

    // Generate or retrieve MSM instances
    const NUM_INSTANCES: u32 = 1;
    let instances = get_or_create_msm_instances::<ArkG, ArkFr>(log_instance_size, NUM_INSTANCES, rng, None).unwrap();
    let affine_points = instances[0].points.iter().map(|p| p.into_affine()).collect::<Vec<_>>();

    // Initialize Metal configuration
    let mut metal_config = setup_metal_state();

    let start_execution = instant::Instant::now();

    // Process MSM instances based on the run mode
    for instance in instances {
        match run_mode.as_str() {
            "gpu" => {
                let _ = metal_msm(&instance.points, &instance.scalars, &mut metal_config).unwrap();
            }
            "par_gpu" => {
                let target_msm_log_size = args.get(3).and_then(|arg| arg.parse::<usize>().ok());

                let _ = metal_msm_parallel(&instance, target_msm_log_size);
            }
            "all_gpu" => {
                let batch_size = args.get(3).and_then(|arg| arg.parse::<u32>().ok());
                let threads_per_tg = args.get(4).and_then(|arg| arg.parse::<u32>().ok());

                let _ = metal_msm_all_gpu(
                    &instance.points,
                    &instance.scalars,
                    &mut metal_config,
                    batch_size,
                    threads_per_tg
                ).unwrap();
            }
            "cpu" => {
                let _ = ArkG::msm(&affine_points, &instance.scalars).unwrap();
            }
            _ => {
                log::error!("Invalid RUN_MODE: {}", run_mode);
                std::process::exit(1);
            }
        }
    }

    log::info!("Total Execution Time: {:?}", start_execution.elapsed());
}