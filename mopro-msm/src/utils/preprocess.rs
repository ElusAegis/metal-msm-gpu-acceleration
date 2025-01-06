use std::env;
use rand::RngCore;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::metal::abstraction::limbs_conversion::{FromLimbs, PointGPU, ScalarGPU, ToLimbs};

#[derive(Debug, Error)]
pub enum HarnessError {
    #[error("I/O error: {0}")]
    FileOpenError(#[from] std::io::Error),

    #[error("Data in file is invalid or incomplete")]
    DeserializationError,

    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// A single MSM instance in *runtime* form.
#[derive(Debug)]
pub struct MsmInstance<P, S> {
    pub points: Vec<P>,
    pub scalars: Vec<S>,
}

impl<P, S> Serialize for MsmInstance<P, S>
where
    P: ToLimbs<24>,
    S: ToLimbs<8>,
{
    fn serialize<T>(&self, serializer: T) -> Result<T::Ok, T::Error>
    where
        T: Serializer,
    {
        // Convert points to Vec<Vec<u32>> limbs
        let points_limb_data: Vec<Vec<u32>> = self
            .points
            .iter()
            .map(|p| p.to_u32_limbs())
            .collect();

        // Convert scalars to Vec<Vec<u32>> limbs
        let scalars_limb_data: Vec<Vec<u32>> = self
            .scalars
            .iter()
            .map(|s| s.to_u32_limbs())
            .collect();

        // Serde can automatically serialize the structure { points, scalars } if we wrap it
        // in a temporary struct or tuple. We'll do a tuple for brevity: (Vec<Vec<u32>>, Vec<Vec<u32>>).
        (points_limb_data, scalars_limb_data).serialize(serializer)
    }
}

impl<'de, P, S> Deserialize<'de> for MsmInstance<P, S>
where
    P: FromLimbs,
    S: FromLimbs,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // We'll be reading a tuple: (Vec<Vec<u32>>, Vec<Vec<u32>>)
        let (points_limb_data, scalars_limb_data): (Vec<Vec<u32>>, Vec<Vec<u32>>)
            = Deserialize::deserialize(deserializer)?;

        // Convert each Vec<u32> → P or S using FromLimbs.
        let points: Vec<P> = points_limb_data
            .into_iter()
            .map(|limbs| P::from_u32_limbs(&limbs))
            .collect();
        let scalars: Vec<S> = scalars_limb_data
            .into_iter()
            .map(|limbs| S::from_u32_limbs(&limbs))
            .collect();

        // Add a check to ensure that the number of points and scalars are equal
        assert_eq!(points.len(), scalars.len());

        Ok(Self { points, scalars })
    }
}


pub fn save_msm_instances<P, S, PathT>(
    path: PathT,
    data: &Vec<MsmInstance<P, S>>,
) -> Result<(), HarnessError>
where
    PathT: AsRef<Path>,
    P: ToLimbs<24>,
    S: ToLimbs<8>,
{
    // We rely on the custom Serialize of MsmInstance<P,S> here
    let mut file = File::create(path)?;
    let encoded = bincode::serialize(data)
        .map_err(|_| HarnessError::DeserializationError)?;
    file.write_all(&encoded)?;
    Ok(())
}

pub fn load_msm_instances<P, S, PathT>(
    path: PathT,
) -> Result<Vec<MsmInstance<P, S>>, HarnessError>
where
    PathT: AsRef<Path>,
    P: FromLimbs,
    S: FromLimbs,
{
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let data: Vec<MsmInstance<P, S>> = bincode::deserialize(&buffer)
        .map_err(|_| HarnessError::DeserializationError)?;
    Ok(data)
}

fn generate_msm_instances<P, S>(
    instance_size: u32,
    num_instances: u32,
    rng: &mut impl RngCore,
) -> Vec<MsmInstance<P, S>>
where
    P: PointGPU<24>,
    S: ScalarGPU<8> + FromLimbs,
{
    let mut out = Vec::with_capacity(num_instances as usize);
    for _ in 0..num_instances {
        let mut points = Vec::with_capacity(instance_size as usize);
        let mut scalars = Vec::with_capacity(instance_size as usize);

        for _ in 0..instance_size {

            let point = P::random(rng);
            let scalar = S::random(rng);

            points.push(point);
            scalars.push(scalar);
        }

        out.push(MsmInstance { points, scalars });
    }
    out
}


/// Tries to load precomputed MSM instances from a single file in `dir`.
/// The file name is derived from `msm_{instance_size}x{num_instances}.bin`.
/// If not found, generate new ones, save, and return them.
pub fn get_or_create_msm_instances<P, S>(
    log_instance_size: u32,
    num_instances: u32,
    mut rng: impl RngCore,
    dir: Option<&str>,
) -> Result<Vec<MsmInstance<P, S>>, HarnessError>
where
    P: PointGPU<24>,
    S: ScalarGPU<8> + FromLimbs,
{
    // Resolve the directory path
    let dir_path = match dir {
        Some(d) => PathBuf::from(d),
        None => default_msm_vec_repo(),
    };

    // Ensure the directory exists, creating it if necessary
    if !dir_path.exists() {
        std::fs::create_dir_all(&dir_path)?;
    }

    // Construct the full path to the file
    let filename = format!("msm_{}x{}.bin", log_instance_size, num_instances);
    let full_path = dir_path.join(filename);

    if Path::new(&full_path).exists() {
        log::debug!("Loading MSM instances from file: {:?}", full_path);
        // Try loading the MSM instances
        let msm_list = load_msm_instances::<P, S, _>(&full_path)?;

        // Validate the loaded instances
        let is_valid = !msm_list.is_empty()
            && msm_list.len() == num_instances as usize
            && msm_list[0].points.len() == 2u32.pow(log_instance_size) as usize;

        if is_valid {
            return Ok(msm_list);
        }

        // Return an error if validation fails
        let loaded_instance_size = msm_list.get(0).map_or(0, |msm| msm.points.len());
        return Err(HarnessError::InvalidData(format!(
            "File mismatch: has instance_size={} and num_instances={}, need {} & {}",
            loaded_instance_size,
            msm_list.len(),
            log_instance_size,
            num_instances
        )));
    }

    // 2. Not found => generate new data
    log::debug!("Generating new MSM instances");
    let msm_list = generate_msm_instances::<P, S>(2u32.pow(log_instance_size), num_instances, &mut rng);

    // 3. Save
    log::debug!("Saving MSM instances to file: {:?}", full_path);
    save_msm_instances(&full_path, &msm_list)?;
    Ok(msm_list)
}

fn default_msm_vec_repo() -> PathBuf {
    let home_dir = env::var("HOME")
        .or_else(|_| env::var("USERPROFILE")) // Windows fallback
        .unwrap_or_else(|_| "/tmp".to_string()); // System-wide fallback for Unix or POSIX-like systems

    Path::new(&home_dir)
        .join(".msm_gpu_acceleration")
        .join("msm_vecs")
}

#[cfg(all(test, feature = "ark"))]
mod tests {
    use super::*;
    use crate::metal::abstraction::limbs_conversion::ark::{ArkFr, ArkG};
    use ark_std::rand::thread_rng;
    use std::fs;

    /// Helper to clean up test directories after tests run
    fn cleanup_test_dir(dir: &str) {
        if Path::new(dir).exists() {
            let _ = fs::remove_dir_all(dir);
        }
    }

    /// Helper to clean up test files after tests run
    fn cleanup_test_file(file: &str) {
        if Path::new(file).exists() {
            let _ = fs::remove_file(file);
        }
    }

    /// Test generating MSM instances and verifying their counts
    #[test]
    fn test_generate_msm_instances() {
        let instance_size = 8;
        let num_instances = 3;
        let mut rng = thread_rng();

        let data = generate_msm_instances::<ArkG, ArkFr>(instance_size, num_instances, &mut rng);
        assert_eq!(data.len(), num_instances as usize);
        for inst in &data {
            assert_eq!(inst.points.len(), instance_size as usize);
            assert_eq!(inst.scalars.len(), instance_size as usize);
        }
    }

    /// Test serialization and deserialization of a single MSM instance
    #[test]
    fn test_serialize_deserialize_single_instance() {
        let mut rng = thread_rng();
        let instance = generate_msm_instances::<ArkG, ArkFr>(4, 1, &mut rng).remove(0);

        // Serialize with bincode
        let encoded = bincode::serialize(&instance).expect("Serialization failed");

        // Deserialize back
        let decoded: MsmInstance<ArkG, ArkFr> =
            bincode::deserialize(&encoded).expect("Deserialization failed");

        // Verify lengths
        assert_eq!(instance.points.len(), decoded.points.len());
        assert_eq!(instance.scalars.len(), decoded.scalars.len());

        // Verify limb consistency for the first point and scalar
        let original_points_limbs: Vec<Vec<u32>> = instance.points.iter().map(|p| p.to_u32_limbs()).collect();
        let decoded_points_limbs: Vec<Vec<u32>> = decoded.points.iter().map(|p| p.to_u32_limbs()).collect();
        assert_eq!(original_points_limbs, decoded_points_limbs);

        let original_scalars_limbs: Vec<Vec<u32>> = instance.scalars.iter().map(|s| s.to_u32_limbs()).collect();
        let decoded_scalars_limbs: Vec<Vec<u32>> = decoded.scalars.iter().map(|s| s.to_u32_limbs()).collect();
        assert_eq!(original_scalars_limbs, decoded_scalars_limbs);
    }

    /// Test saving and loading MSM instances with file naming convention
    #[test]
    fn test_save_and_load_msm_instances() -> Result<(), HarnessError> {
        let test_file = default_msm_vec_repo().join("test_data_msm.bin");
        let test_file = test_file.to_str().unwrap();
        cleanup_test_file(test_file); // Ensure a clean state

        let instance_size = 4;
        let num_instances = 2;
        let mut rng = thread_rng();

        // Generate instances
        let generated_instances = generate_msm_instances::<ArkG, ArkFr>(
            instance_size,
            num_instances,
            &mut rng,
        );

        // Save instances
        save_msm_instances(test_file, &generated_instances)?;

        // Load instances
        let loaded_instances = load_msm_instances::<ArkG, ArkFr, _>(
            test_file,
        )?;

        // Verify counts
        assert_eq!(loaded_instances.len(), num_instances as usize);

        for (i, (gen, load)) in generated_instances.iter().zip(loaded_instances.iter()).enumerate() {
            assert_eq!(gen.points.len(), load.points.len());
            assert_eq!(gen.scalars.len(), load.scalars.len());

            // Verify limb consistency for the first point and scalar
            let gen_points_limbs: Vec<Vec<u32>> = gen.points.iter().map(|p| p.to_u32_limbs()).collect();
            let load_points_limbs: Vec<Vec<u32>> = load.points.iter().map(|p| p.to_u32_limbs()).collect();
            assert_eq!(gen_points_limbs, load_points_limbs, "Mismatch in points for instance {}", i);

            let gen_scalars_limbs: Vec<Vec<u32>> = gen.scalars.iter().map(|s| s.to_u32_limbs()).collect();
            let load_scalars_limbs: Vec<Vec<u32>> = load.scalars.iter().map(|s| s.to_u32_limbs()).collect();
            assert_eq!(gen_scalars_limbs, load_scalars_limbs, "Mismatch in scalars for instance {}", i);
        }

        cleanup_test_file(test_file); // Clean up after test
        Ok(())
    }

    /// Test `get_or_create_msm_instances` functionality
    #[test]
    fn test_get_or_create_msm_instances() -> Result<(), HarnessError> {
        let test_dir = default_msm_vec_repo().join("test_data_get_or_create");
        let test_dir = test_dir.to_str().unwrap();
        cleanup_test_dir(test_dir); // Ensure a clean state

        let log_instance_size = 8;
        let num_instances = 5;
        let mut rng = thread_rng();

        // First call: should generate and save
        let generated_instances = get_or_create_msm_instances::<ArkG, ArkFr>(
            log_instance_size,
            num_instances,
            &mut rng,
            Some(test_dir),
        )?;
        assert_eq!(generated_instances.len(), num_instances as usize);
        for inst in &generated_instances {
            assert_eq!(inst.points.len(), 2u32.pow(log_instance_size) as usize);
            assert_eq!(inst.scalars.len(), 2u32.pow(log_instance_size) as usize);
        }

        // Second call: should load the same data
        let loaded_instances = get_or_create_msm_instances::<ArkG, ArkFr>(
            log_instance_size,
            num_instances,
            &mut rng,
            Some(test_dir),
        )?;
        assert_eq!(loaded_instances.len(), num_instances as usize);

        // Verify that loaded data matches the generated data
        for (i, (gen, load)) in generated_instances.iter().zip(loaded_instances.iter()).enumerate() {
            // Compare limbs for points and scalars
            let gen_points_limbs: Vec<Vec<u32>> = gen.points.iter().map(|p| p.to_u32_limbs()).collect();
            let load_points_limbs: Vec<Vec<u32>> = load.points.iter().map(|p| p.to_u32_limbs()).collect();
            assert_eq!(gen_points_limbs, load_points_limbs, "Mismatch in points for instance {}", i);

            let gen_scalars_limbs: Vec<Vec<u32>> = gen.scalars.iter().map(|s| s.to_u32_limbs()).collect();
            let load_scalars_limbs: Vec<Vec<u32>> = load.scalars.iter().map(|s| s.to_u32_limbs()).collect();
            assert_eq!(gen_scalars_limbs, load_scalars_limbs, "Mismatch in scalars for instance {}", i);
        }

        cleanup_test_dir(test_dir); // Clean up after test
        Ok(())
    }
}