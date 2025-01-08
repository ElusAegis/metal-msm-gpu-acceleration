use crate::metal::abstraction::limbs_conversion::PointGPU;
use crate::metal::abstraction::state::MetalState;
use crate::metal::msm::{MetalMsmConfig, MetalMsmInstance};

pub fn final_accumulation<P: PointGPU<24>>(
    _config: &MetalMsmConfig,
    instance: &MetalMsmInstance,
) -> P {
    // Retrieve the results from the `res_buffer` where partial reductions are stored
    let raw_limbs = MetalState::retrieve_contents::<u32>(&instance.data.res_buffer);

    // Convert the flat array of u32 into points of type `P`
    let res_points: Vec<P> = raw_limbs
        .chunks(24) // Assuming 24 u32 values represent a single point (e.g., 3 Fp elements)
        .map(P::from_u32_limbs)
        .collect();

    // Perform the accumulation logic as in the Metal kernel
    let mut total_sum = res_points.last().unwrap().clone();
    let lowest_window_sum = res_points.first().unwrap().clone();
    // Perform `window_size` doublings on `total_sum`
    for _ in 0..instance.params.window_size {
        total_sum = total_sum.clone() + total_sum;
    }
    let last_res_idx = (instance.params.window_num - 1) as usize;

    // Iterate over the windows from the highest index (num_windows - 1) down to 1
    for i in 1..instance.params.window_num - 1 {
        let current_sum = res_points[last_res_idx - (i as usize)].clone();
        total_sum = total_sum + current_sum; // Add the current window's sum

        // Perform `window_size` doublings on `total_sum`
        for _ in 0..instance.params.window_size {
            total_sum = total_sum.clone() + total_sum;
        }
    }

    // Add the lowest window's sum to the accumulated total and return the final MSM result
    total_sum + lowest_window_sum
}
