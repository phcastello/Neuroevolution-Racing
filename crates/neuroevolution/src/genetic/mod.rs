pub mod config;
mod genome;
mod individual;
mod operators;
mod population;

pub use config::Config;
pub use genome::Genome;
pub use individual::Individual;
pub use population::Population;

#[cfg(test)]
mod tests;
