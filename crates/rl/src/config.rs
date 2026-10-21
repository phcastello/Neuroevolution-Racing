use crate::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum ObservationKind {
    Legacy6,
    Extended { ray_angles_degrees: Vec<f32> },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Feature {
    pub name: String,
    pub source: String,
    pub unit: String,
    pub normalization: String,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ObservationSpec {
    pub version: u32,
    pub kind: ObservationKind,
    pub sensor_range: f32,
    pub speed_scale: f32,
    /// Missing in historical presets/checkpoints: retain the original signed clamp.
    #[serde(default)]
    pub speed_normalization: SpeedNormalization,
    pub angular_speed_scale: f32,
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub enum SpeedNormalization {
    #[default]
    LinearClamp,
    SignedAsymptotic,
}
impl Default for ObservationSpec {
    fn default() -> Self {
        Self {
            version: 1,
            kind: ObservationKind::Extended {
                ray_angles_degrees: (0..21).map(|i| 90.0 - i as f32 * 9.0).collect(),
            },
            sensor_range: 1000.0,
            speed_scale: 250.0,
            speed_normalization: SpeedNormalization::LinearClamp,
            angular_speed_scale: 2.15,
        }
    }
}
impl ObservationSpec {
    pub fn angles(&self) -> Vec<f32> {
        match &self.kind {
            ObservationKind::Legacy6 => vec![60., 30., 0., -30., -60.],
            ObservationKind::Extended { ray_angles_degrees } => ray_angles_degrees.clone(),
        }
    }
    pub fn dimension(&self) -> usize {
        self.angles().len()
            + if matches!(self.kind, ObservationKind::Legacy6) {
                1
            } else {
                4
            }
    }
    /// Ordered metadata is persisted; this method accepts no privileged environment state.
    pub fn features(&self) -> Vec<Feature> {
        let mut features: Vec<_> = self
            .angles()
            .iter()
            .enumerate()
            .map(|(i, a)| Feature {
                name: format!("ray_{i}_{a}deg"),
                source: "local wall raycast".into(),
                unit: "world unit".into(),
                normalization: format!("distance / {}, [0,1]", self.sensor_range),
            })
            .collect();
        features.push(Feature {
            name: "longitudinal_speed".into(),
            source: "kinematic signed speed".into(),
            unit: "world unit/s".into(),
            normalization: if matches!(self.kind, ObservationKind::Legacy6) {
                format!("abs(v)/(abs(v)+{})", self.speed_scale)
            } else if self.speed_normalization == SpeedNormalization::SignedAsymptotic {
                format!("v/({}+abs(v))", self.speed_scale)
            } else {
                format!("clamp(v/{},-1,1)", self.speed_scale)
            },
        });
        if !matches!(self.kind, ObservationKind::Legacy6) {
            features.push(Feature {
                name: "measured_yaw_rate".into(),
                source: "resolved heading difference / physical dt".into(),
                unit: "rad/s".into(),
                normalization: format!("clamp(w/{},-1,1)", self.angular_speed_scale),
            });
            for name in ["previous_steering", "previous_acceleration"] {
                features.push(Feature {
                    name: name.into(),
                    source: "last executed action".into(),
                    unit: "dimensionless".into(),
                    normalization: "identity [-1,1]".into(),
                });
            }
        }
        features
    }
    pub fn validate(&self) -> Result<()> {
        ensure(self.version == 1, "observation version")?;
        ensure(
            !matches!(self.kind, ObservationKind::Legacy6)
                || self.speed_normalization == SpeedNormalization::LinearClamp,
            "signed speed normalization requires Extended observations",
        )?;
        ensure(
            [
                self.sensor_range,
                self.speed_scale,
                self.angular_speed_scale,
            ]
            .iter()
            .all(|x| x.is_finite() && *x > 0.),
            "observation scales",
        )?;
        let angles = self.angles();
        ensure(
            !angles.is_empty()
                && angles.len() <= 512
                && angles.iter().all(|a| a.is_finite() && a.abs() <= 180.),
            "ray angles",
        )
    }
}
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub enum RewardMode {
    #[default]
    Baseline,
    GaInspired {
        /// Extra reward per normalized lap, gated by useful forward speed.
        speed_weight: f32,
        /// Useful progress speed in world units/s where the speed factor is 0.5.
        progress_speed_half_saturation: f32,
    },
}
impl RewardMode {
    fn is_baseline(&self) -> bool {
        matches!(self, Self::Baseline)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RewardConfig {
    pub version: u32,
    /// Omitted by legacy presets/checkpoints; their reward remains exactly Baseline.
    #[serde(default, skip_serializing_if = "RewardMode::is_baseline")]
    pub mode: RewardMode,
    pub progress_weight: f32,
    pub completion_bonus: f32,
    pub collision_penalty: f32,
}
impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            version: 1,
            mode: RewardMode::Baseline,
            progress_weight: 0.01,
            completion_bonus: 10.,
            collision_penalty: 1.,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum DeviceChoice {
    Cpu,
    Gpu { adapter: usize },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum OptimizerKind {
    Sgd,
    Adam {
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum LogStdMode {
    GlobalPerActionDimension,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub version: u32,
    pub device: DeviceChoice,
    pub observation: ObservationSpec,
    pub reward: RewardConfig,
    pub hidden: Vec<usize>,
    pub log_std_mode: LogStdMode,
    pub initial_log_std: [f32; 2],
    pub log_std_bounds: [f32; 2],
    pub num_envs: usize,
    pub rollout_length: usize,
    pub minibatch_size: usize,
    pub update_epochs: usize,
    pub total_rollouts: usize,
    pub gamma: f32,
    pub lambda: f32,
    pub learning_rate: f32,
    pub clip_epsilon: f32,
    pub value_coef: f32,
    pub latent_entropy_coef: f32,
    pub target_kl: f32,
    pub max_gradient_norm: f32,
    pub optimizer: OptimizerKind,
    pub network_seed: u64,
    pub environment_seed: u64,
    pub action_seed: u64,
    pub shuffle_seed: u64,
    pub evaluation_seed: u64,
    pub physical_dt: f32,
    pub action_repeat: usize,
    pub timeout_ticks: u64,
    pub memory_limit_mib: usize,
    pub training_tracks: Vec<String>,
    pub evaluation_tracks: Vec<String>,
    pub results_root: String,
    pub benchmark_envs: Vec<usize>,
    pub benchmark_iterations: usize,
    pub benchmark_warmup: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            version: 1,
            device: DeviceChoice::Cpu,
            observation: ObservationSpec::default(),
            reward: RewardConfig::default(),
            hidden: vec![64, 64],
            log_std_mode: LogStdMode::GlobalPerActionDimension,
            initial_log_std: [-0.5; 2],
            log_std_bounds: [-5., 2.],
            num_envs: 8,
            rollout_length: 128,
            minibatch_size: 64,
            update_epochs: 4,
            total_rollouts: 100,
            gamma: 0.99,
            lambda: 0.95,
            learning_rate: 0.0003,
            clip_epsilon: 0.2,
            value_coef: 0.5,
            latent_entropy_coef: 0.,
            target_kl: 0.03,
            max_gradient_norm: 0.5,
            optimizer: OptimizerKind::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            network_seed: 11,
            environment_seed: 22,
            action_seed: 33,
            shuffle_seed: 44,
            evaluation_seed: 55,
            physical_dt: 1. / 60.,
            action_repeat: 1,
            timeout_ticks: 10800,
            memory_limit_mib: 2048,
            training_tracks: vec![],
            evaluation_tracks: vec![],
            results_root: "results_ppo".into(),
            benchmark_envs: vec![1, 8, 32, 128, 256],
            benchmark_iterations: 100,
            benchmark_warmup: 10,
        }
    }
}
impl Config {
    pub fn rollout_size(&self) -> usize {
        self.num_envs * self.rollout_length
    }
    pub fn parameter_count(&self) -> usize {
        let mut n = 0;
        let mut input = self.observation.dimension();
        for &out in &self.hidden {
            n += out * (input + 1);
            input = out;
        }
        n + 3 * (input + 1) + 2
    }
    pub fn estimated_bytes(&self) -> u128 {
        let d = self.observation.dimension() as u128;
        let r = self.num_envs as u128 * self.rollout_length as u128;
        let m = self.minibatch_size as u128;
        let h = self.hidden.iter().map(|x| *x as u128).sum::<u128>();
        4 * (r * (4 * d + 64) + m * (8 * h + 8 * d + 128) + self.parameter_count() as u128 * 16)
    }
    pub fn validate(&self) -> Result<()> {
        ensure(self.version == 1, "config version")?;
        self.observation.validate()?;
        ensure(
            !self.hidden.is_empty()
                && self.hidden.len() <= 16
                && self.hidden.iter().all(|x| *x > 0 && *x <= 8192),
            "hidden dimensions",
        )?;
        ensure(
            self.num_envs > 0
                && self.num_envs <= 4096
                && self.rollout_length > 0
                && self.rollout_length <= 1_000_000,
            "N/T limits",
        )?;
        ensure(
            self.minibatch_size > 0
                && self.minibatch_size <= self.rollout_size()
                && self.update_epochs > 0
                && self.total_rollouts > 0,
            "M/E/rollouts",
        )?;
        ensure(
            self.action_repeat > 0
                && self.action_repeat <= 600
                && self.timeout_ticks > 0
                && (self.physical_dt - 1. / 60.).abs() < 1e-9,
            "physical_dt=1/60, repeat, timeout",
        )?;
        ensure(
            [self.gamma, self.lambda]
                .iter()
                .all(|x| x.is_finite() && (0.0..=1.0).contains(x)),
            "gamma/lambda",
        )?;
        ensure(
            [self.learning_rate, self.target_kl, self.max_gradient_norm]
                .iter()
                .all(|x| x.is_finite() && *x > 0.),
            "optimizer/KL/norm",
        )?;
        ensure(
            self.clip_epsilon.is_finite() && self.clip_epsilon > 0. && self.clip_epsilon < 1.,
            "clip epsilon",
        )?;
        ensure(
            [
                self.value_coef,
                self.latent_entropy_coef,
                self.reward.progress_weight,
                self.reward.completion_bonus,
                self.reward.collision_penalty,
            ]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.)
                && self.reward.version == 1,
            "loss/reward coefficients",
        )?;
        if let RewardMode::GaInspired {
            speed_weight,
            progress_speed_half_saturation,
        } = self.reward.mode
        {
            ensure(
                speed_weight.is_finite()
                    && speed_weight >= 0.
                    && progress_speed_half_saturation.is_finite()
                    && progress_speed_half_saturation > 0.
                    && (self.reward.progress_weight + speed_weight + self.reward.completion_bonus)
                        .is_finite(),
                "GA-inspired reward coefficients",
            )?;
        }
        let [lo, hi] = self.log_std_bounds;
        ensure(
            lo.is_finite()
                && hi.is_finite()
                && lo >= -20.
                && hi <= 5.
                && lo < hi
                && self
                    .initial_log_std
                    .iter()
                    .all(|x| x.is_finite() && *x >= lo && *x <= hi),
            "log_std bounds",
        )?;
        if let OptimizerKind::Adam {
            beta1,
            beta2,
            epsilon,
        } = self.optimizer
        {
            ensure(
                [beta1, beta2]
                    .iter()
                    .all(|x| x.is_finite() && (0.0..1.0).contains(x))
                    && epsilon.is_finite()
                    && epsilon > 0.,
                "Adam settings",
            )?;
        }
        ensure(
            self.benchmark_iterations > 0
                && self.benchmark_warmup > 0
                && self.benchmark_envs.iter().all(|n| *n > 0 && *n <= 4096),
            "benchmark dimensions",
        )?;
        ensure(!self.results_root.trim().is_empty(), "results_root")?;
        ensure(
            self.estimated_bytes() <= self.memory_limit_mib as u128 * 1024 * 1024,
            "estimated working set exceeds memory_limit_mib",
        )
    }
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let c: Self = ron::from_str(&std::fs::read_to_string(path)?)
            .map_err(|e| crate::Error::Invalid(e.to_string()))?;
        c.validate()?;
        Ok(c)
    }
}
