use std::{env};
use rand::rngs::OsRng;
use mopro_msm::msm::metal::abstraction::limbs_conversion::ark::{ArkFr, ArkG};
use mopro_msm::msm::metal::msm::{encode_instances, exec_metal_commands, metal_msm_parallel, setup_metal_state};
use mopro_msm::msm::utils::preprocess::get_or_create_msm_instances;

fn main() {
    let run_parallel = env::var("RUN_PARALLEL")
        .map(|val| val == "true" || val == "1")
        .unwrap_or(false);
    let rng = OsRng::default();

    const LOG_INSTANCE_SIZE: u32 = 16;
    const NUM_INSTANCES: u32 = 5;

    let instances = get_or_create_msm_instances::<ArkG, ArkFr>(LOG_INSTANCE_SIZE, NUM_INSTANCES, rng, None).unwrap();

    let mut metal_config = setup_metal_state();

    for instance in instances {

        if run_parallel {
             let _ = metal_msm_parallel(&instance, 10);
        } else {
            let metal_instance = encode_instances(&instance.points, &instance.scalars, &mut metal_config);
            let _res: ArkG = exec_metal_commands(&metal_config, metal_instance).unwrap();
        }
    }
}