use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct Config {
    /// CPU/GPU split ratio, from 0.0 to 1.0
    pub cpu_gpu_split_ratio: f64,
    /// Bucket size for 16-bit instance length
    pub bucket_size_16: u32,
    /// Accumulation: Desired pairs per thread
    pub desired_pairs_per_thread: u32,
    /// Reduction: Buckets per threadgroup
    pub buckets_per_threadgroup: u32,
    /// Minimum or maximum threads in some stage (32..=64)
    pub min_max_threads: u32,
}


pub struct ConfigManager {
    config_path: String,
}

impl ConfigManager {
    /// Creates a new ConfigManager with the specified path to the config file.
    fn new(config_path: &str) -> Self {
        ConfigManager {
            config_path: config_path.to_string(),
        }
    }

    /// Loads the config file from disk.
    fn load_config(&self) -> Config {
        let config_str = fs::read_to_string(&self.config_path)
            .expect("Failed to read config file");
        toml::from_str(&config_str).expect("Failed to parse config file")
    }

    pub fn save_config(&self, config: &Config) {
        let config_str = toml::to_string(config).expect("Failed to serialize config");
        fs::write(&self.config_path, config_str).expect("Failed to write config file");
    }

    /// Getter for `desired_pairs_per_thread` that dynamically loads the config.
    pub(crate) fn desired_pairs_per_thread(&self) -> u32 {
        self.load_config().desired_pairs_per_thread
    }

    /// Getter for `buckets_per_threadgroup` that dynamically loads the config.
    pub(crate) fn buckets_per_threadgroup(&self) -> u32 {
        self.load_config().buckets_per_threadgroup
    }

    /// Getter for `min_max_threads` that dynamically loads the config.
    pub(crate) fn min_max_threads(&self) -> u32 {
        self.load_config().min_max_threads
    }

    /// Getter for `bucket_size_16` that dynamically loads the config.
    pub(crate) fn bucket_size_16(&self) -> u32 {
        self.load_config().bucket_size_16
    }

    /// Getter for `cpu_gpu_proportion` that dynamically loads the config.
    pub(crate) fn cpu_gpu_split_ratio(&self) -> f64 {
        self.load_config().cpu_gpu_split_ratio
    }
}

/// Default implementation for ConfigManager
impl Default for ConfigManager {
    fn default() -> Self {
        ConfigManager::new("config.toml")
    }
}