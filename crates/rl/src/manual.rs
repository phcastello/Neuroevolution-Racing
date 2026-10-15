//! Finite learning surface. No derivative formulas are supplied here or elsewhere.
//! See docs/rl/BACKPROP_PARA_IMPLEMENTAR.md for ownership, scale and test contracts.
use crate::{
    Error, Result,
    backend::{Backend, Tensor},
    network::{Dense, ForwardCache, LayerCache, Network},
    ppo::LossCache,
};

pub const FUNCTIONS: &[&str] = &[
    "activation_backward",
    "dense_backward",
    "trunk_backward",
    "shared_heads_backward",
    "ppo_output_gradients",
];
pub fn missing(function: &'static str) -> Error {
    Error::MissingManualImplementation { function }
}
/// Call real hooks on a tiny frozen case, never a success flag or dummy gradient.
pub fn preflight() -> Result<()> {
    crate::verification::probe_hooks()
}

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Relu,
    Linear,
    Tanh,
}
/// Elementwise activation VJP. z, activation and upstream are all [B,H]; returns [B,H].
pub fn activation_backward<B: Backend>(
    _kind: Activation,
    _z: Tensor<B, 2>,
    _activation: Tensor<B, 2>,
    _upstream: Tensor<B, 2>,
) -> Result<Tensor<B, 2>> {
    Err(missing("activation_backward"))
}
pub struct DenseGradients<B: Backend> {
    pub input: Tensor<B, 2>,
    pub weights: Tensor<B, 2>,
    pub bias: Tensor<B, 2>,
}
/// Linear dense VJP: X[B,I], W[O,I], upstream dZ[B,O] -> dX[B,I],dW[O,I],db[1,O]. No mean here.
pub fn dense_backward<B: Backend>(
    _layer: &Dense<B>,
    _input: Tensor<B, 2>,
    _upstream: Tensor<B, 2>,
) -> Result<DenseGradients<B>> {
    Err(missing("dense_backward"))
}
pub struct TrunkGradients<B: Backend> {
    pub input: Tensor<B, 2>,
    pub parameters: Vec<Tensor<B, 2>>,
}
/// Shared ReLU trunk VJP, reverse traversal. Parameters returned in forward W,b order; no mean.
pub fn trunk_backward<B: Backend>(
    _layers: &[Dense<B>],
    _caches: &[LayerCache<B>],
    _upstream: Tensor<B, 2>,
) -> Result<TrunkGradients<B>> {
    Err(missing("trunk_backward"))
}
pub struct OutputGradients<B: Backend> {
    pub mean: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
    pub log_std: Tensor<B, 2>,
}
/// PPO+transformed distribution VJP. Frozen historical u; returns dmu[B,2],dV[B,1],dlog_std[1,2].
/// Include exactly one 1/B for each averaged loss term and the configured coefficients.
pub fn ppo_output_gradients<B: Backend>(_cache: &LossCache<B>) -> Result<OutputGradients<B>> {
    Err(missing("ppo_output_gradients"))
}
/// Heads VJP, sum BOTH trunk contributions before trunk_backward. No updates here.
/// Return every parameter gradient, including global log_std, in Network::parameters order.
/// This hook must call dense_backward and trunk_backward; trunk must call activation_backward.
pub fn shared_heads_backward<B: Backend>(
    _network: &Network<B>,
    _cache: ForwardCache<B>,
    _outputs: OutputGradients<B>,
) -> Result<Vec<Tensor<B, 2>>> {
    Err(missing("shared_heads_backward"))
}
