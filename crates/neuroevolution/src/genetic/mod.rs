pub mod config;
mod genome;
mod individual;
mod population;
mod operators;

pub use config::Config;

#[cfg(test)]
mod tests;