pub mod metal;
pub mod utils;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum MoproError {
    #[error("CircomError: {0}")]
    CircomError(String),
    #[error("Halo2Error: {0}")]
    Halo2Error(String),
}


#[cfg(all(feature = "ios-bindings", feature = "h2c"))]
mod ios_bindings;

#[cfg(feature = "ios-bindings")]
uniffi::setup_scaffolding!();
