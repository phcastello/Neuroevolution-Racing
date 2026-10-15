//! PPO infrastructure without autodiff. The only derivative code belongs in `manual`.
pub mod backend;
pub mod config;
pub mod distribution;
pub mod manual;
pub mod network;
pub mod optimizer;
pub mod persistence;
pub mod ppo;
pub mod rollout;
pub mod trainer;
pub mod verification;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("MissingManualImplementation {{ function: {function} }} (PENDENTE)")]
    MissingManualImplementation { function: &'static str },
    #[error("invalid RL data/configuration: {0}")]
    Invalid(String),
    #[error("non-finite numerical result: {0}")]
    Numerical(String),
    #[error("device unavailable: {0}")]
    Device(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
}
pub type Result<T> = std::result::Result<T, Error>;

pub fn ensure(condition: bool, message: impl Into<String>) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::Invalid(message.into()))
    }
}
pub fn finite(values: &[f32], context: &str) -> Result<()> {
    if values.iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(Error::Numerical(context.into()))
    }
}
