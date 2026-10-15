use crate::{Result, config::Config, ensure, optimizer::OptimizerState};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
#[derive(Clone, Serialize, Deserialize)]
#[serde(into = "SavedRngs", from = "SavedRngs")]
pub struct Rngs {
    pub action: ChaCha12Rng,
    pub shuffle: ChaCha12Rng,
    pub environments: Vec<ChaCha12Rng>,
}
// RON does not support u128. Persist ChaCha's 128-bit word offset as two u64s.
#[derive(Serialize, Deserialize)]
struct SavedRng {
    seed: [u8; 32],
    stream: u64,
    word_position: [u64; 2],
}
impl From<ChaCha12Rng> for SavedRng {
    fn from(r: ChaCha12Rng) -> Self {
        let p = r.get_word_pos();
        Self {
            seed: r.get_seed(),
            stream: r.get_stream(),
            word_position: [p as u64, (p >> 64) as u64],
        }
    }
}
impl From<SavedRng> for ChaCha12Rng {
    fn from(r: SavedRng) -> Self {
        let mut rng = Self::from_seed(r.seed);
        rng.set_stream(r.stream);
        rng.set_word_pos(r.word_position[0] as u128 | ((r.word_position[1] as u128) << 64));
        rng
    }
}
#[derive(Serialize, Deserialize)]
struct SavedRngs {
    action: SavedRng,
    shuffle: SavedRng,
    environments: Vec<SavedRng>,
}
impl From<Rngs> for SavedRngs {
    fn from(r: Rngs) -> Self {
        Self {
            action: r.action.into(),
            shuffle: r.shuffle.into(),
            environments: r.environments.into_iter().map(Into::into).collect(),
        }
    }
}
impl From<SavedRngs> for Rngs {
    fn from(r: SavedRngs) -> Self {
        Self {
            action: r.action.into(),
            shuffle: r.shuffle.into(),
            environments: r.environments.into_iter().map(Into::into).collect(),
        }
    }
}
impl Rngs {
    pub fn new(c: &Config) -> Self {
        Self {
            action: ChaCha12Rng::seed_from_u64(c.action_seed),
            shuffle: ChaCha12Rng::seed_from_u64(c.shuffle_seed),
            environments: (0..c.num_envs)
                .map(|i| ChaCha12Rng::seed_from_u64(c.environment_seed.wrapping_add(i as u64)))
                .collect(),
        }
    }
}
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub format: String,
    pub version: u32,
    pub config: Config,
    pub features: Vec<crate::config::Feature>,
    pub parameter_shapes: Vec<[usize; 2]>,
    pub parameters: Vec<Vec<f32>>,
    pub policy_version: u64,
    pub optimizer: OptimizerState,
    pub rngs: Rngs,
    pub completed_rollouts: u64,
    pub decisions: u64,
    pub physical_ticks: u64,
    pub rendered_frames: u64,
    pub backend: String,
    pub trained: bool,
    /// Application-owned serialized physical configuration; app validates it on load.
    pub physics_ron: String,
    pub resume_policy: String,
}
impl Checkpoint {
    pub fn validate(&self) -> Result<()> {
        ensure(
            self.format == "racing-ppo" && self.version == 1,
            "PPO checkpoint format",
        )?;
        self.config.validate()?;
        ensure(
            self.features == self.config.observation.features(),
            "observation metadata",
        )?;
        ensure(
            self.parameters.len() == self.parameter_shapes.len()
                && self
                    .parameters
                    .iter()
                    .zip(&self.parameter_shapes)
                    .all(|(p, s)| s[0].checked_mul(s[1]) == Some(p.len())),
            "parameter shapes",
        )?;
        for p in &self.parameters {
            crate::finite(p, "checkpoint")?;
        }
        ensure(
            self.trained == (self.optimizer.step > 0),
            "checkpoint trained status/optimizer step",
        )?;
        ensure(
            self.optimizer.kind == self.config.optimizer,
            "checkpoint optimizer configuration",
        )?;
        ensure(
            self.optimizer.first.len() == self.parameters.len()
                && self.optimizer.second.len() == self.parameters.len(),
            "checkpoint optimizer tensor count",
        )?;
        for ((first, second), shape) in self
            .optimizer
            .first
            .iter()
            .zip(&self.optimizer.second)
            .zip(&self.parameter_shapes)
        {
            ensure(
                shape[0].checked_mul(shape[1]) == Some(first.len()) && first.len() == second.len(),
                "checkpoint optimizer tensor shape",
            )?;
            crate::finite(first, "checkpoint first moment")?;
            crate::finite(second, "checkpoint second moment")?;
            ensure(
                second.iter().all(|v| *v >= 0.),
                "negative Adam second moment",
            )?;
        }
        ensure(
            self.rngs.environments.len() == self.config.num_envs,
            "environment RNG count",
        )
    }
}
pub fn load<T: DeserializeOwned>(path: &Path) -> Result<T> {
    ron::from_str(&fs::read_to_string(path)?).map_err(|e| crate::Error::Invalid(e.to_string()))
}
/// New files only; synced temp file followed by rename in the same directory.
/// Call at a rollout boundary. Never replaces an existing artifact/run.
pub fn save<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    ensure(
        !path.exists(),
        format!("refusing to overwrite {}", path.display()),
    )?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension(format!("tmp-{}", std::process::id()));
    let text = ron::ser::to_string_pretty(value, ron::ser::PrettyConfig::default())
        .map_err(|e| crate::Error::Invalid(e.to_string()))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()?;
    drop(file);
    fs::rename(temp, path)?;
    Ok(())
}
pub fn run_directory(root: &Path) -> Result<PathBuf> {
    fs::create_dir_all(root)?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| crate::Error::Invalid(e.to_string()))?
        .as_nanos();
    let path = root.join(format!("ppo-{timestamp}-{}", std::process::id()));
    fs::create_dir(&path)?;
    Ok(path)
}
