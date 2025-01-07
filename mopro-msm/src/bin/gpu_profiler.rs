use std::env;
#[cfg(all(feature = "ark", not(feature = "h2c")))]
use ark_ec::{CurveGroup, VariableBaseMSM};
#[cfg(feature = "h2c")]
use halo2curves::group::Curve;
use rand::rngs::OsRng;
#[cfg(all(feature = "ark", not(feature = "h2c")))]
use mopro_msm::metal::abstraction::limbs_conversion::ark::{ArkFr as Fr , ArkG as G};
#[cfg(feature = "h2c")]
use mopro_msm::metal::abstraction::limbs_conversion::h2c::{H2Fr as Fr , H2G as G, H2GAffine as GAffine};
#[cfg(feature = "h2c")]
use mopro_msm::metal::best_msm;
use mopro_msm::metal::msm::{metal_msm_parallel};
#[cfg(all(feature = "ark", not(feature = "h2c")))]
use mopro_msm::metal::msm::{setup_metal_state, metal_msm};
use mopro_msm::utils::preprocess::get_or_create_msm_instances;

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
    let instances = get_or_create_msm_instances::<G, Fr>(log_instance_size, NUM_INSTANCES, rng, None).unwrap();
    #[cfg(all(feature = "ark", not(feature = "h2c")))]
    let affine_points = instances[0].points.iter().map(|p| p.into_affine()).collect::<Vec<_>>();
    #[cfg(feature = "h2c")]
    let affine_points = instances[0].points.iter().map(|p| p.to_affine()).collect::<Vec<_>>();

    let start_execution = instant::Instant::now();

    // Process MSM instances based on the run mode
    for instance in instances {
        match run_mode.as_str() {
            #[cfg(all(feature = "ark", not(feature = "h2c")))]
            "gpu" => {
                // Initialize Metal configuration

                let config_time = instant::Instant::now();
                let mut metal_config = setup_metal_state();
                log::debug!("Config Setup Time: {:?}", config_time.elapsed());

                let _ = metal_msm(&instance.points, &instance.scalars, &mut metal_config).unwrap();
            }
            #[cfg(feature = "h2c")]
            "gpu" => {
                let _ = best_msm::<GAffine, GAffine, G, Fr>(&instance.scalars, &affine_points);
            }
            "par_gpu" => {
                let target_msm_log_size = args.get(3).and_then(|arg| arg.parse::<usize>().ok());

                let _ = metal_msm_parallel(&instance, target_msm_log_size);
            }
            #[cfg(all(feature = "ark", not(feature = "h2c")))]
            "cpu" => {
                let _ = G::msm(&affine_points, &instance.scalars).unwrap();
            }
            #[cfg(feature = "h2c")]
            "cpu" => {
                let _ = halo2curves::msm::msm_best(&instance.scalars, &affine_points);
            }
            _ => {
                log::error!("Invalid RUN_MODE: {}", run_mode);
                std::process::exit(1);
            }
        }
    }

    log::info!("Total Execution Time: {:?}", start_execution.elapsed());
}