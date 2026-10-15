use crate::{
    Result,
    backend::{Backend, Tensor, read},
    config::Config,
    distribution::DistributionCache,
    ensure,
};
use serde::{Deserialize, Serialize};
pub struct LossInputs<B: Backend> {
    pub old_log_prob: Tensor<B, 2>,
    pub advantage: Tensor<B, 2>,
    pub value_target: Tensor<B, 2>,
}
pub struct LossCache<B: Backend> {
    pub distribution: DistributionCache<B>,
    pub inputs: LossInputs<B>,
    pub value: Tensor<B, 2>,
    pub ratio: Tensor<B, 2>,
    pub clipped_ratio: Tensor<B, 2>,
    pub total: Tensor<B, 1>,
    pub epsilon: f32,
    pub value_coef: f32,
    pub latent_entropy_coef: f32,
    pub batch_size: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LossMetrics {
    pub policy_loss: f32,
    pub value_loss: f32,
    pub latent_entropy: f32,
    pub total_loss: f32,
    pub mean_ratio: f32,
    pub clip_fraction: f32,
    pub approximate_kl: f32,
}
/// Each loss term is averaged exactly once here over B. Manual output gradients MUST include
/// this same single mean; dense/network backward sum contributions without dividing by B.
pub fn forward<B: Backend>(
    distribution: DistributionCache<B>,
    value: Tensor<B, 2>,
    inputs: LossInputs<B>,
    c: &Config,
) -> Result<(LossCache<B>, LossMetrics)> {
    let [b, d] = value.dims();
    ensure(
        b > 0
            && d == 1
            && inputs.old_log_prob.dims() == [b, 1]
            && inputs.advantage.dims() == [b, 1]
            && inputs.value_target.dims() == [b, 1]
            && distribution.mean.dims() == [b, 2],
        "PPO loss shapes",
    )?;
    let log_ratio = distribution.joint_log_density.clone() - inputs.old_log_prob.clone();
    let ratio = log_ratio.clone().exp();
    let clipped_ratio = ratio
        .clone()
        .clamp(1. - c.clip_epsilon, 1. + c.clip_epsilon);
    let first = ratio.clone() * inputs.advantage.clone();
    let second = clipped_ratio.clone() * inputs.advantage.clone();
    let policy = first
        .clone()
        .mask_where(first.greater(second.clone()), second)
        .mean()
        .neg();
    let value_loss = (value.clone() - inputs.value_target.clone())
        .powf_scalar(2.)
        .mean()
        * 0.5;
    let entropy = distribution.latent_entropy.clone().mean();
    let total = policy.clone() + value_loss.clone() * c.value_coef
        - entropy.clone() * c.latent_entropy_coef;
    let clipped = (ratio.clone() - 1.)
        .abs()
        .greater_elem(c.clip_epsilon)
        .float()
        .mean();
    let kl = (ratio.clone() - 1. - log_ratio).mean();
    let metrics = LossMetrics {
        policy_loss: read(policy)?[0],
        value_loss: read(value_loss)?[0],
        latent_entropy: read(entropy)?[0],
        total_loss: read(total.clone())?[0],
        mean_ratio: read(ratio.clone().mean())?[0],
        clip_fraction: read(clipped)?[0],
        approximate_kl: read(kl)?[0],
    };
    Ok((
        LossCache {
            distribution,
            inputs,
            value,
            ratio,
            clipped_ratio,
            total,
            epsilon: c.clip_epsilon,
            value_coef: c.value_coef,
            latent_entropy_coef: c.latent_entropy_coef,
            batch_size: b,
        },
        metrics,
    ))
}
