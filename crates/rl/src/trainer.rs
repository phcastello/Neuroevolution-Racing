use crate::{
    Result,
    backend::{Backend, Int, Tensor, TensorData, matrix},
    config::Config,
    distribution, ensure, manual,
    network::Network,
    optimizer::{Optimizer, clip_global_norm, validate_gradients},
    ppo::{self, LossInputs, LossMetrics},
    rollout::Rollout,
};
use rand_chacha::ChaCha12Rng;
#[derive(serde::Serialize, Debug)]
pub struct UpdateMetrics {
    pub loss: LossMetrics,
    pub gradient_norm_before_clip: Option<f32>,
    pub optimizer_step: u64,
    pub stopped_by_kl: bool,
}
/// Upload immutable rollout once; select aligned rows on the selected device each epoch.
pub fn update<B: Backend>(
    network: &mut Network<B>,
    optimizer: &mut Optimizer<B>,
    rollout: &Rollout,
    c: &Config,
    rng: &mut ChaCha12Rng,
    device: &B::Device,
) -> Result<Vec<UpdateMetrics>> {
    rollout.validate()?;
    ensure(
        c.observation == rollout.observation_spec
            && c.num_envs == rollout.num_envs
            && c.rollout_length == rollout.rollout_length,
        "update config/rollout mismatch",
    )?;
    ensure(
        network.version() == rollout.policy_version,
        "rollout policy version mismatch",
    )?;
    let size = rollout.transitions.len();
    let d = c.observation.dimension();
    let observations = matrix(
        &rollout
            .transitions
            .iter()
            .flat_map(|s| s.observation.iter().copied())
            .collect::<Vec<_>>(),
        size,
        d,
        device,
    )?;
    let u = matrix(
        &rollout
            .transitions
            .iter()
            .flat_map(|s| s.u)
            .collect::<Vec<_>>(),
        size,
        2,
        device,
    )?;
    let old = matrix(
        &rollout
            .transitions
            .iter()
            .map(|s| s.old_log_prob)
            .collect::<Vec<_>>(),
        size,
        1,
        device,
    )?;
    let advantages = matrix(&rollout.policy_advantages, size, 1, device)?;
    let targets = matrix(&rollout.value_targets, size, 1, device)?;
    let mut metrics = vec![];
    for _ in 0..c.update_epochs {
        for indices in rollout.minibatches(c.minibatch_size, rng)? {
            let values: Vec<i64> = indices.iter().map(|i| *i as i64).collect();
            let indices = Tensor::<B, 1, Int>::from_data(
                TensorData::new(values, [indices.len()]).convert::<B::IntElem>(),
                device,
            );
            let cache = network.forward(observations.clone().select(0, indices.clone()))?;
            let dist = distribution::evaluate(
                cache.mean.clone(),
                cache.log_std.clone(),
                u.clone().select(0, indices.clone()),
            )?;
            let inputs = LossInputs {
                old_log_prob: old.clone().select(0, indices.clone()),
                advantage: advantages.clone().select(0, indices.clone()),
                value_target: targets.clone().select(0, indices),
            };
            let (loss, metric) = ppo::forward(dist, cache.value.clone(), inputs, c)?;
            if metric.approximate_kl > c.target_kl {
                metrics.push(UpdateMetrics {
                    loss: metric,
                    gradient_norm_before_clip: None,
                    optimizer_step: optimizer.step,
                    stopped_by_kl: true,
                });
                return Ok(metrics);
            }
            network.validate_cache(&cache)?;
            let outputs = manual::ppo_output_gradients(&loss)?;
            ensure(
                outputs.mean.dims() == cache.mean.dims()
                    && outputs.value.dims() == cache.value.dims()
                    && outputs.log_std.dims() == [1, 2],
                "manual output gradient shapes",
            )?;
            for output in [&outputs.mean, &outputs.value, &outputs.log_std] {
                crate::backend::validate(output, "manual output gradients")?;
            }
            // Fresh returned gradient buffers per minibatch. Complete both branches before mutation.
            let gradients = manual::shared_heads_backward(network, cache, outputs)?;
            let parameters = network.parameters();
            validate_gradients(&parameters, &gradients)?;
            let (gradients, norm) = clip_global_norm(gradients, c.max_gradient_norm)?;
            let updated = optimizer.update(&parameters, gradients, c.learning_rate)?;
            network.replace(updated, c.log_std_bounds)?;
            metrics.push(UpdateMetrics {
                loss: metric,
                gradient_norm_before_clip: Some(norm),
                optimizer_step: optimizer.step,
                stopped_by_kl: false,
            });
        }
    }
    Ok(metrics)
}
