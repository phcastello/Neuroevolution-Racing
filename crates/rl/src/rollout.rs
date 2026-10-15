//! Time-major [T,N,...]. GAE is computed before permutation; no replay history.
use crate::{
    Result,
    backend::{Backend, matrix, read},
    config::Config,
    distribution, ensure, finite,
    network::Network,
};
use rand::seq::SliceRandom;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EndReason {
    Collision,
    Completed,
    Timeout,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RewardParts {
    pub progress: f32,
    pub completion: f32,
    pub collision: f32,
}
impl RewardParts {
    pub fn total(&self) -> f32 {
        self.progress + self.completion + self.collision
    }
}
pub struct Step {
    pub observation: Vec<f32>,
    pub reward: RewardParts,
    pub terminated: bool,
    pub truncated: bool,
    pub reason: Option<EndReason>,
    pub physical_ticks: u64,
    pub progress: f32,
}
/// Stable env_id is its slice index. step returns FINAL observation; collector resets separately.
pub trait Environment {
    fn env_id(&self) -> usize;
    fn observe(&mut self) -> Result<Vec<f32>>;
    fn reset(&mut self) -> Result<Vec<f32>>;
    fn step(&mut self, action: [f32; 2]) -> Result<Step>;
    fn episode_id(&self) -> u64;
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transition {
    pub env_id: usize,
    pub episode_id: u64,
    pub observation: Vec<f32>,
    pub action: [f32; 2],
    pub u: [f32; 2],
    pub old_log_prob: f32,
    pub old_value: f32,
    pub next_observation: Vec<f32>,
    pub next_value: f32,
    pub reward: RewardParts,
    pub terminated: bool,
    pub truncated: bool,
    pub reason: Option<EndReason>,
    pub physical_ticks: u64,
    pub progress: f32,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Timings {
    pub observations_seconds: f64,
    pub upload_seconds: f64,
    pub inference_seconds: f64,
    pub download_seconds: f64,
    pub simulation_seconds: f64,
    pub preparation_seconds: f64,
    pub total_seconds: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rollout {
    pub version: u32,
    pub policy_version: u64,
    pub num_envs: usize,
    pub rollout_length: usize,
    pub observation_spec: crate::config::ObservationSpec,
    pub transitions: Vec<Transition>,
    pub raw_advantages: Vec<f32>,
    pub policy_advantages: Vec<f32>,
    pub value_targets: Vec<f32>,
    pub timings: Timings,
}
impl Rollout {
    pub fn validate(&self) -> Result<()> {
        self.observation_spec.validate()?;
        let d = self.observation_spec.dimension();
        ensure(
            self.version == 1
                && self.num_envs > 0
                && self.rollout_length > 0
                && self.num_envs.checked_mul(self.rollout_length) == Some(self.transitions.len()),
            "rollout shape/version",
        )?;
        for (i, s) in self.transitions.iter().enumerate() {
            ensure(
                s.env_id == i % self.num_envs
                    && s.observation.len() == d
                    && s.next_observation.len() == d,
                "transition alignment/D",
            )?;
            finite(&s.observation, "observations")?;
            finite(&s.next_observation, "final observations")?;
            finite(
                &[
                    s.u[0],
                    s.u[1],
                    s.old_log_prob,
                    s.old_value,
                    s.next_value,
                    s.reward.progress,
                    s.reward.completion,
                    s.reward.collision,
                    s.progress,
                ],
                "transition",
            )?;
            ensure(
                s.action.iter().all(|v| v.is_finite() && v.abs() <= 1.)
                    && s.action
                        .iter()
                        .zip(s.u)
                        .all(|(a, u)| (*a - u.tanh()).abs() < 2e-6),
                "action/tanh consistency",
            )?;
            ensure(
                !(s.terminated && s.truncated) && s.physical_ticks > 0,
                "flags/ticks",
            )?;
            ensure(
                match s.reason {
                    Some(EndReason::Timeout) => s.truncated && !s.terminated,
                    Some(_) => s.terminated && !s.truncated,
                    None => !s.terminated && !s.truncated,
                },
                "termination reason",
            )?;
            if i >= self.num_envs {
                let previous = &self.transitions[i - self.num_envs];
                ensure(
                    s.episode_id
                        == previous.episode_id
                            + u64::from(previous.terminated || previous.truncated),
                    "episode boundary",
                )?;
                if !previous.terminated && !previous.truncated {
                    ensure(
                        previous.next_observation == s.observation,
                        "next observation continuity",
                    )?;
                }
            }
        }
        for v in [
            &self.raw_advantages,
            &self.policy_advantages,
            &self.value_targets,
        ] {
            ensure(v.len() == self.transitions.len(), "GAE length")?;
            finite(v, "GAE")?;
        }
        Ok(())
    }
    pub fn prepare(&mut self, gamma: f32, lambda: f32) -> Result<()> {
        let now = Instant::now();
        let count = self.transitions.len();
        ensure(
            self.num_envs > 0 && self.rollout_length.checked_mul(self.num_envs) == Some(count),
            "GAE buffer shape",
        )?;
        ensure(
            [gamma, lambda]
                .iter()
                .all(|x| x.is_finite() && (0.0..=1.0).contains(x)),
            "GAE gamma/lambda",
        )?;
        self.raw_advantages = vec![0.; count];
        self.value_targets = vec![0.; count];
        for env in 0..self.num_envs {
            let mut tail = 0.;
            for t in (0..self.rollout_length).rev() {
                let i = t * self.num_envs + env;
                let s = &self.transitions[i];
                // Bootstrap truncations from FINAL state; never recurse across either boundary.
                let bootstrap = if s.terminated { 0. } else { 1. };
                let continue_gae = if s.terminated || s.truncated { 0. } else { 1. };
                let delta = s.reward.total() + gamma * bootstrap * s.next_value - s.old_value;
                tail = delta + gamma * lambda * continue_gae * tail;
                self.raw_advantages[i] = tail;
                self.value_targets[i] = tail + s.old_value;
            }
        }
        let mean = self.raw_advantages.iter().map(|v| *v as f64).sum::<f64>() / count as f64;
        let variance = self
            .raw_advantages
            .iter()
            .map(|v| (*v as f64 - mean).powi(2))
            .sum::<f64>()
            / count as f64;
        self.policy_advantages = self
            .raw_advantages
            .iter()
            .map(|v| ((*v as f64 - mean) / variance.sqrt().max(1e-8)) as f32)
            .collect();
        self.timings.preparation_seconds = now.elapsed().as_secs_f64();
        self.validate()
    }
    pub fn minibatches(&self, size: usize, rng: &mut ChaCha12Rng) -> Result<Vec<Vec<usize>>> {
        ensure(size > 0, "minibatch size")?;
        let mut indices: Vec<_> = (0..self.transitions.len()).collect();
        indices.shuffle(rng);
        Ok(indices.chunks(size).map(|s| s.to_vec()).collect())
    }
}
pub fn collect<B: Backend, E: Environment>(
    network: &Network<B>,
    envs: &mut [E],
    config: &Config,
    rng: &mut ChaCha12Rng,
    device: &B::Device,
) -> Result<Rollout> {
    config.validate()?;
    ensure(envs.len() == config.num_envs, "environment count")?;
    ensure(
        envs.iter().enumerate().all(|(i, e)| e.env_id() == i),
        "environment identity/order mismatch",
    )?;
    let start = Instant::now();
    let mut timings = Timings::default();
    let n = envs.len();
    let d = config.observation.dimension();
    let obs_time = Instant::now();
    let mut observations = envs
        .iter_mut()
        .map(Environment::observe)
        .collect::<Result<Vec<_>>>()?;
    timings.observations_seconds += obs_time.elapsed().as_secs_f64();
    let version = network.version();
    let mut transitions = Vec::with_capacity(config.rollout_size());
    for _ in 0..config.rollout_length {
        ensure(
            network.version() == version,
            "weights changed during rollout",
        )?;
        let clock = Instant::now();
        let flat: Vec<_> = observations.iter().flatten().copied().collect();
        let x = matrix(&flat, n, d, device)?;
        crate::backend::sync::<B>(device)?;
        timings.upload_seconds += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let out = network.infer(x)?;
        let sampled = distribution::sample(out.mean, network.log_std.clone(), rng)?;
        crate::backend::sync::<B>(device)?;
        timings.inference_seconds += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let actions = read(sampled.action)?;
        let u = read(sampled.u)?;
        let logs = read(sampled.joint_log_density)?;
        let values = read(out.value)?;
        timings.download_seconds += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let mut steps = vec![];
        let mut episodes = vec![];
        for (i, env) in envs.iter_mut().enumerate() {
            episodes.push(env.episode_id());
            steps.push(env.step([actions[2 * i], actions[2 * i + 1]])?);
        }
        timings.simulation_seconds += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let final_obs: Vec<_> = steps
            .iter()
            .flat_map(|s| s.observation.iter().copied())
            .collect();
        let next_x = matrix(&final_obs, n, d, device)?;
        crate::backend::sync::<B>(device)?;
        timings.upload_seconds += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let next = network.infer(next_x)?.value;
        crate::backend::sync::<B>(device)?;
        timings.inference_seconds += clock.elapsed().as_secs_f64();
        let clock = Instant::now();
        let next_values = read(next)?;
        timings.download_seconds += clock.elapsed().as_secs_f64();
        for (i, step) in steps.into_iter().enumerate() {
            let done = step.terminated || step.truncated;
            transitions.push(Transition {
                env_id: i,
                episode_id: episodes[i],
                observation: observations[i].clone(),
                action: [actions[2 * i], actions[2 * i + 1]],
                u: [u[2 * i], u[2 * i + 1]],
                old_log_prob: logs[i],
                old_value: values[i],
                next_observation: step.observation.clone(),
                next_value: next_values[i],
                reward: step.reward,
                terminated: step.terminated,
                truncated: step.truncated,
                reason: step.reason,
                physical_ticks: step.physical_ticks,
                progress: step.progress,
            });
            let clock = Instant::now();
            observations[i] = if done {
                envs[i].reset()?
            } else {
                step.observation
            };
            timings.observations_seconds += clock.elapsed().as_secs_f64();
        }
    }
    timings.total_seconds = start.elapsed().as_secs_f64();
    let mut rollout = Rollout {
        version: 1,
        policy_version: version,
        num_envs: n,
        rollout_length: config.rollout_length,
        observation_spec: config.observation.clone(),
        transitions,
        raw_advantages: vec![],
        policy_advantages: vec![],
        value_targets: vec![],
        timings,
    };
    rollout.prepare(config.gamma, config.lambda)?;
    rollout.timings.total_seconds = start.elapsed().as_secs_f64();
    Ok(rollout)
}
