use std::{
    fmt,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use bevy::prelude::Resource;
use neuroevolution::neural::{Activation, Architecture, Mlp};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};

use super::{
    SimulationConfig,
    controller::{MLP_INPUT_SIZE, MLP_OUTPUT_SIZE, MlpController},
    training::{
        EvaluationConfig, FinishReason, FinishReasonCounts, LaserConfig, TrainingState,
        TrainingTrackSelection,
    },
};

pub const LEGACY_CHECKPOINT_FORMAT_VERSION: u32 = 1;
pub const HARD_CLAMP_CHECKPOINT_FORMAT_VERSION: u32 = 2;
pub const CHECKPOINT_FORMAT_VERSION: u32 = 3;
pub const DEFAULT_CHECKPOINT_DIRECTORY: &str = "checkpoints";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CheckpointError(String);

impl CheckpointError {
    fn invalid(message: impl Into<String>) -> Self {
        Self(format!("invalid checkpoint: {}", message.into()))
    }
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CheckpointError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SavedActivation {
    Tanh,
    Relu,
    Linear,
}

impl From<Activation> for SavedActivation {
    fn from(value: Activation) -> Self {
        match value {
            Activation::Tanh => Self::Tanh,
            Activation::Relu => Self::Relu,
            Activation::Linear => Self::Linear,
        }
    }
}

impl From<SavedActivation> for Activation {
    fn from(value: SavedActivation) -> Self {
        match value {
            SavedActivation::Tanh => Self::Tanh,
            SavedActivation::Relu => Self::Relu,
            SavedActivation::Linear => Self::Linear,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedArchitecture {
    pub layer_sizes: Vec<usize>,
    pub activations: Vec<SavedActivation>,
}

impl SavedArchitecture {
    pub fn from_architecture(architecture: &Architecture) -> Self {
        Self {
            layer_sizes: architecture.layer_sizes().to_vec(),
            activations: architecture
                .activations()
                .iter()
                .copied()
                .map(SavedActivation::from)
                .collect(),
        }
    }

    pub fn to_architecture(&self) -> Result<Architecture, CheckpointError> {
        Architecture::new(
            self.layer_sizes.clone(),
            self.activations
                .iter()
                .copied()
                .map(Activation::from)
                .collect(),
        )
        .map_err(|error| CheckpointError::invalid(format!("architecture: {error}")))
    }

    pub fn display(&self) -> String {
        self.layer_sizes
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" -> ")
    }

    fn checked_parameter_count(&self) -> Result<usize, CheckpointError> {
        self.layer_sizes.windows(2).try_fold(0usize, |total, pair| {
            pair[0]
                .checked_mul(pair[1])
                .and_then(|weights| weights.checked_add(pair[1]))
                .and_then(|layer| total.checked_add(layer))
                .ok_or_else(|| CheckpointError::invalid("architecture parameter count overflow"))
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SavedFinishReason {
    Completed,
    Collision,
    Stalled,
    EliminatedByLaser,
    Timeout,
}

impl From<FinishReason> for SavedFinishReason {
    fn from(value: FinishReason) -> Self {
        match value {
            FinishReason::Completed => Self::Completed,
            FinishReason::Collision => Self::Collision,
            FinishReason::EliminatedByLaser => Self::EliminatedByLaser,
            FinishReason::Timeout => Self::Timeout,
        }
    }
}

impl SavedFinishReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Collision => "collision",
            Self::Stalled => "stalled",
            Self::EliminatedByLaser => "eliminated by laser",
            Self::Timeout => "timeout",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SavedFinishReasonCounts {
    pub completed: usize,
    pub collision: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub stalled: usize,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub laser_eliminated: usize,
    pub timeout: usize,
}

impl From<FinishReasonCounts> for SavedFinishReasonCounts {
    fn from(value: FinishReasonCounts) -> Self {
        Self {
            completed: value.completed,
            collision: value.collision,
            stalled: 0,
            laser_eliminated: value.laser_eliminated,
            timeout: value.timeout,
        }
    }
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SavedTrainingTrackSelection {
    All,
    RandomSubset(usize),
}

impl From<TrainingTrackSelection> for SavedTrainingTrackSelection {
    fn from(value: TrainingTrackSelection) -> Self {
        match value {
            TrainingTrackSelection::All => Self::All,
            TrainingTrackSelection::RandomSubset(count) => Self::RandomSubset(count),
        }
    }
}

impl From<SavedTrainingTrackSelection> for TrainingTrackSelection {
    fn from(value: SavedTrainingTrackSelection) -> Self {
        match value {
            SavedTrainingTrackSelection::All => Self::All,
            SavedTrainingTrackSelection::RandomSubset(count) => Self::RandomSubset(count),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedEvaluationParameters {
    pub maximum_episode_duration: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub stall_timeout: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub significant_progress_epsilon: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub laser: Option<SavedLaserConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sensor_max_distance: Option<f32>,
    pub progress_weight: f32,
    pub speed_weight: f32,
    pub collision_penalty: f32,
    pub completion_bonus: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub useful_speed_normalization: Option<SavedUsefulSpeedNormalization>,
    #[serde(alias = "progress_speed_normalization")]
    pub progress_speed_half_saturation: f32,
    pub training_track_selection: SavedTrainingTrackSelection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SavedUsefulSpeedNormalization {
    AsymptoticHalfSaturation,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedLaserConfig {
    pub grace_period: f32,
    pub acceleration: f32,
    pub maximum_speed: f32,
}

impl From<LaserConfig> for SavedLaserConfig {
    fn from(value: LaserConfig) -> Self {
        Self {
            grace_period: value.grace_period,
            acceleration: value.acceleration,
            maximum_speed: value.maximum_speed,
        }
    }
}

impl From<SavedLaserConfig> for LaserConfig {
    fn from(value: SavedLaserConfig) -> Self {
        Self {
            grace_period: value.grace_period,
            acceleration: value.acceleration,
            maximum_speed: value.maximum_speed,
        }
    }
}

fn is_zero_f32(value: &f32) -> bool {
    *value == 0.0
}

impl SavedEvaluationParameters {
    fn from_config(config: &EvaluationConfig, simulation: &SimulationConfig) -> Self {
        Self {
            maximum_episode_duration: config.maximum_episode_duration,
            stall_timeout: 0.0,
            significant_progress_epsilon: 0.0,
            laser: Some(config.laser.into()),
            sensor_max_distance: Some(simulation.sensor_max_distance),
            progress_weight: config.progress_weight,
            speed_weight: config.speed_weight,
            collision_penalty: config.collision_penalty,
            completion_bonus: config.completion_bonus,
            useful_speed_normalization: Some(
                SavedUsefulSpeedNormalization::AsymptoticHalfSaturation,
            ),
            progress_speed_half_saturation: config.progress_speed_half_saturation,
            training_track_selection: config.training_track_selection.into(),
        }
    }

    fn validate(&self, format_version: u32) -> Result<(), CheckpointError> {
        match format_version {
            LEGACY_CHECKPOINT_FORMAT_VERSION => {
                validate_positive("maximum_episode_duration", self.maximum_episode_duration)?;
                validate_positive("stall_timeout", self.stall_timeout)?;
                validate_positive(
                    "significant_progress_epsilon",
                    self.significant_progress_epsilon,
                )?;
                validate_scoring_parameters(self)
            }
            HARD_CLAMP_CHECKPOINT_FORMAT_VERSION | CHECKPOINT_FORMAT_VERSION => {
                let laser = self
                    .laser
                    .ok_or_else(|| CheckpointError::invalid("checkpoint is missing laser"))?;
                let sensor_max_distance = self.sensor_max_distance.ok_or_else(|| {
                    CheckpointError::invalid("checkpoint is missing sensor_max_distance")
                })?;
                validate_positive("sensor_max_distance", sensor_max_distance)?;
                if format_version == CHECKPOINT_FORMAT_VERSION
                    && self.useful_speed_normalization
                        != Some(SavedUsefulSpeedNormalization::AsymptoticHalfSaturation)
                {
                    return Err(CheckpointError::invalid(
                        "V3 checkpoint must declare asymptotic useful-speed normalization",
                    ));
                }
                EvaluationConfig {
                    maximum_episode_duration: self.maximum_episode_duration,
                    laser: laser.into(),
                    progress_weight: self.progress_weight,
                    speed_weight: self.speed_weight,
                    collision_penalty: self.collision_penalty,
                    completion_bonus: self.completion_bonus,
                    progress_speed_half_saturation: self.progress_speed_half_saturation,
                    training_track_selection: self.training_track_selection.clone().into(),
                }
                .validate()
                .map_err(CheckpointError::invalid)
            }
            _ => Err(CheckpointError::invalid(format!(
                "unsupported format_version {format_version}; expected 1, 2, or 3"
            ))),
        }
    }
}

fn validate_positive(name: &str, value: f32) -> Result<(), CheckpointError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(CheckpointError::invalid(format!(
            "{name} must be finite and greater than zero"
        )))
    }
}

fn validate_scoring_parameters(
    parameters: &SavedEvaluationParameters,
) -> Result<(), CheckpointError> {
    validate_positive(
        "progress_speed_half_saturation",
        parameters.progress_speed_half_saturation,
    )?;
    for (name, value) in [
        ("progress_weight", parameters.progress_weight),
        ("speed_weight", parameters.speed_weight),
        ("collision_penalty", parameters.collision_penalty),
        ("completion_bonus", parameters.completion_bonus),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(CheckpointError::invalid(format!(
                "{name} must be finite and non-negative"
            )));
        }
    }
    if parameters.completion_bonus <= parameters.speed_weight {
        return Err(CheckpointError::invalid(
            "completion_bonus must be greater than speed_weight so completion always wins",
        ));
    }
    if matches!(
        parameters.training_track_selection,
        SavedTrainingTrackSelection::RandomSubset(0)
    ) {
        return Err(CheckpointError::invalid(
            "training track subset size must be greater than zero",
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedTrainingMetadata {
    pub champion_training_fitness: f32,
    pub population_average_fitness: f32,
    pub average_useful_progress_speed: f32,
    pub completion_rate: f32,
    pub finish_counts: SavedFinishReasonCounts,
    pub training_tracks: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedValidationMetadata {
    pub track_id: String,
    pub score: f32,
    pub normalized_progress: f32,
    pub useful_progress_speed: f32,
    pub elapsed: f32,
    pub finish_reason: SavedFinishReason,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedNetwork {
    pub format_version: u32,
    pub saved_at_unix_seconds: u64,
    pub generation: usize,
    pub architecture: SavedArchitecture,
    pub genome: Vec<f32>,
    pub training_metadata: SavedTrainingMetadata,
    pub validation_metadata: SavedValidationMetadata,
    pub evaluation_parameters: SavedEvaluationParameters,
}

impl SavedNetwork {
    pub fn from_training(
        training: &TrainingState,
        simulation: &SimulationConfig,
    ) -> Result<Self, CheckpointError> {
        let completed = training.completed_champion().ok_or_else(|| {
            CheckpointError::invalid("no champion with completed validation is available")
        })?;
        let saved = Self {
            format_version: CHECKPOINT_FORMAT_VERSION,
            saved_at_unix_seconds: unix_timestamp_seconds()?,
            generation: completed.generation,
            architecture: SavedArchitecture::from_architecture(training.architecture()),
            genome: completed.genome.clone(),
            training_metadata: SavedTrainingMetadata {
                champion_training_fitness: completed.training.training_fitness,
                population_average_fitness: completed.training.population_average_fitness,
                average_useful_progress_speed: completed.training.average_useful_progress_speed,
                completion_rate: completed.training.completion_rate,
                finish_counts: completed.training.finish_counts.into(),
                training_tracks: completed.training.training_tracks.clone(),
            },
            validation_metadata: SavedValidationMetadata {
                track_id: completed.validation.track_id.clone(),
                score: completed.validation.score,
                normalized_progress: completed.validation.normalized_progress,
                useful_progress_speed: completed.validation.useful_progress_speed,
                elapsed: completed.validation.elapsed,
                finish_reason: completed.validation.finish_reason.into(),
            },
            evaluation_parameters: SavedEvaluationParameters::from_config(
                training.evaluation_config(),
                simulation,
            ),
        };
        saved.validate()?;
        Ok(saved)
    }

    pub fn validate(&self) -> Result<(), CheckpointError> {
        if !matches!(
            self.format_version,
            LEGACY_CHECKPOINT_FORMAT_VERSION
                | HARD_CLAMP_CHECKPOINT_FORMAT_VERSION
                | CHECKPOINT_FORMAT_VERSION
        ) {
            return Err(CheckpointError::invalid(format!(
                "unsupported format_version {}; expected 1, 2, or 3",
                self.format_version
            )));
        }
        let architecture = self.architecture.to_architecture()?;
        if architecture.input_size() != MLP_INPUT_SIZE {
            return Err(CheckpointError::invalid(format!(
                "input size {} is incompatible with app contract {}",
                architecture.input_size(),
                MLP_INPUT_SIZE
            )));
        }
        if architecture.output_size() != MLP_OUTPUT_SIZE {
            return Err(CheckpointError::invalid(format!(
                "output size {} is incompatible with app contract {}",
                architecture.output_size(),
                MLP_OUTPUT_SIZE
            )));
        }
        let expected_parameters = self.architecture.checked_parameter_count()?;
        if self.genome.len() != expected_parameters {
            return Err(CheckpointError::invalid(format!(
                "genome has {} parameters; architecture requires {expected_parameters}",
                self.genome.len()
            )));
        }
        validate_finite_slice("genome", &self.genome)?;
        validate_finite_slice(
            "training metadata",
            &[
                self.training_metadata.champion_training_fitness,
                self.training_metadata.population_average_fitness,
                self.training_metadata.average_useful_progress_speed,
                self.training_metadata.completion_rate,
            ],
        )?;
        validate_finite_slice(
            "validation metadata",
            &[
                self.validation_metadata.score,
                self.validation_metadata.normalized_progress,
                self.validation_metadata.useful_progress_speed,
                self.validation_metadata.elapsed,
            ],
        )?;
        if !(0.0..=1.0).contains(&self.training_metadata.completion_rate) {
            return Err(CheckpointError::invalid(
                "training completion_rate must be between zero and one",
            ));
        }
        if !(0.0..=1.0).contains(&self.validation_metadata.normalized_progress) {
            return Err(CheckpointError::invalid(
                "validation normalized_progress must be between zero and one",
            ));
        }
        self.evaluation_parameters.validate(self.format_version)?;
        Mlp::from_parameters(&architecture, &self.genome)
            .map_err(|error| CheckpointError::invalid(format!("MLP reconstruction: {error}")))?;
        Ok(())
    }

    pub fn reconstruct(&self) -> Result<(Architecture, Mlp), CheckpointError> {
        self.validate()?;
        let architecture = self.architecture.to_architecture()?;
        let mlp = Mlp::from_parameters(&architecture, &self.genome)
            .map_err(|error| CheckpointError::invalid(format!("MLP reconstruction: {error}")))?;
        Ok((architecture, mlp))
    }
}

fn validate_finite_slice(label: &str, values: &[f32]) -> Result<(), CheckpointError> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(CheckpointError::invalid(format!(
            "{label} contains NaN or Infinity"
        )));
    }
    Ok(())
}

fn unix_timestamp_seconds() -> Result<u64, CheckpointError> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| CheckpointError(format!("system clock error: {error}")))
}

pub fn parse_saved_network(contents: &str) -> Result<SavedNetwork, CheckpointError> {
    let saved: SavedNetwork = ron::from_str(contents)
        .map_err(|error| CheckpointError(format!("RON parse error: {error}")))?;
    saved.validate()?;
    Ok(saved)
}

pub fn load_saved_network(path: &Path) -> Result<SavedNetwork, CheckpointError> {
    let contents = fs::read_to_string(path)
        .map_err(|error| CheckpointError(format!("failed to read {}: {error}", path.display())))?;
    parse_saved_network(&contents)
}

#[derive(Clone, Debug)]
pub struct CheckpointSummary {
    pub format_version: u32,
    pub generation: usize,
    pub architecture: String,
    pub training: SavedTrainingMetadata,
    pub validation: SavedValidationMetadata,
    pub evaluation: SavedEvaluationParameters,
}

impl From<&SavedNetwork> for CheckpointSummary {
    fn from(value: &SavedNetwork) -> Self {
        Self {
            format_version: value.format_version,
            generation: value.generation,
            architecture: value.architecture.display(),
            training: value.training_metadata.clone(),
            validation: value.validation_metadata.clone(),
            evaluation: value.evaluation_parameters.clone(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CheckpointEntry {
    pub path: PathBuf,
    pub filename: String,
    pub summary: Option<CheckpointSummary>,
    pub error: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AutoSaveSettings {
    pub enabled: bool,
    pub interval_generations: usize,
}

impl Default for AutoSaveSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            interval_generations: 10,
        }
    }
}

#[derive(Resource, Debug)]
pub struct CheckpointStore {
    directory: PathBuf,
    pub settings: AutoSaveSettings,
    pub entries: Vec<CheckpointEntry>,
    pub status: String,
    last_auto_saved_generation: Option<usize>,
}

impl Default for CheckpointStore {
    fn default() -> Self {
        Self::new(PathBuf::from(DEFAULT_CHECKPOINT_DIRECTORY))
    }
}

impl CheckpointStore {
    pub fn new(directory: PathBuf) -> Self {
        let mut store = Self {
            directory,
            settings: AutoSaveSettings::default(),
            entries: Vec::new(),
            status: String::new(),
            last_auto_saved_generation: None,
        };
        store.refresh();
        store
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    pub fn refresh(&mut self) {
        self.entries.clear();
        let mut paths = Vec::new();
        match fs::read_dir(&self.directory) {
            Ok(directory) => paths.extend(
                directory
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|path| path.extension().is_some_and(|extension| extension == "ron")),
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                self.status = format!("Failed to scan {}: {error}", self.directory.display());
                return;
            }
        }
        // The interactive browser keeps its historical `checkpoints/` source and also
        // discovers best-network files produced under the default batch `results/` root.
        if self.directory == Path::new(DEFAULT_CHECKPOINT_DIRECTORY) {
            if let Ok(architectures) = fs::read_dir("results") {
                for bests in architectures
                    .filter_map(Result::ok)
                    .map(|entry| entry.path().join("bests_by_gen"))
                {
                    if let Ok(entries) = fs::read_dir(bests) {
                        paths.extend(
                            entries
                                .filter_map(Result::ok)
                                .map(|entry| entry.path())
                                .filter(|path| {
                                    path.extension().is_some_and(|extension| extension == "ron")
                                }),
                        );
                    }
                }
            }
        }
        if paths.is_empty() {
            self.status = "No checkpoints saved yet".into();
            return;
        }
        paths.sort_by(|left, right| right.file_name().cmp(&left.file_name()));
        for path in paths {
            let filename = path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("<invalid filename>")
                .to_string();
            match load_saved_network(&path) {
                Ok(saved) => self.entries.push(CheckpointEntry {
                    path,
                    filename,
                    summary: Some(CheckpointSummary::from(&saved)),
                    error: None,
                }),
                Err(error) => self.entries.push(CheckpointEntry {
                    path,
                    filename,
                    summary: None,
                    error: Some(error.to_string()),
                }),
            }
        }
        self.status = format!("{} checkpoint(s) found", self.entries.len());
    }

    pub fn save_current_champion(
        &mut self,
        training: &TrainingState,
        simulation: &SimulationConfig,
    ) -> Result<PathBuf, CheckpointError> {
        let saved = SavedNetwork::from_training(training, simulation)?;
        let path = self.write_unique(&saved)?;
        self.status = format!("Saved {}", path.display());
        self.refresh();
        self.status = format!("Saved {}", path.display());
        Ok(path)
    }

    pub fn auto_save_if_due(
        &mut self,
        training: &TrainingState,
        simulation: &SimulationConfig,
    ) -> Result<Option<PathBuf>, CheckpointError> {
        let Some(completed) = training.completed_champion() else {
            return Ok(None);
        };
        if !auto_save_is_due(
            &self.settings,
            completed.generation,
            self.last_auto_saved_generation,
        ) {
            return Ok(None);
        }
        let path = self.save_current_champion(training, simulation)?;
        self.last_auto_saved_generation = Some(completed.generation);
        Ok(Some(path))
    }

    pub fn load(&mut self, index: usize) -> Result<LoadedCheckpoint, CheckpointError> {
        let entry = self
            .entries
            .get(index)
            .ok_or_else(|| CheckpointError::invalid("checkpoint selection no longer exists"))?;
        let saved = load_saved_network(&entry.path)?;
        let (architecture, mlp) = saved.reconstruct()?;
        MlpController::new(mlp, &saved.genome)
            .map_err(|error| CheckpointError::invalid(format!("controller: {error}")))?;
        let loaded = LoadedCheckpoint {
            source_filename: entry.filename.clone(),
            architecture,
            saved,
        };
        self.status = format!("Loaded {} for Champion mode", loaded.source_filename);
        Ok(loaded)
    }

    fn write_unique(&self, saved: &SavedNetwork) -> Result<PathBuf, CheckpointError> {
        fs::create_dir_all(&self.directory).map_err(|error| {
            CheckpointError(format!(
                "failed to create {}: {error}",
                self.directory.display()
            ))
        })?;
        let ron = ron::ser::to_string_pretty(saved, PrettyConfig::default())
            .map_err(|error| CheckpointError(format!("failed to serialize checkpoint: {error}")))?;
        let stem = format!(
            "generation_{:06}_{}",
            saved.generation, saved.saved_at_unix_seconds
        );
        for suffix in 0..10_000usize {
            let filename = if suffix == 0 {
                format!("{stem}.ron")
            } else {
                format!("{stem}_{suffix}.ron")
            };
            let path = self.directory.join(filename);
            match OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(mut file) => {
                    file.write_all(ron.as_bytes()).map_err(|error| {
                        CheckpointError(format!("failed to write {}: {error}", path.display()))
                    })?;
                    return Ok(path);
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => {
                    return Err(CheckpointError(format!(
                        "failed to create {}: {error}",
                        path.display()
                    )));
                }
            }
        }
        Err(CheckpointError(
            "could not allocate a unique checkpoint filename".into(),
        ))
    }
}

fn auto_save_is_due(
    settings: &AutoSaveSettings,
    generation: usize,
    last_auto_saved_generation: Option<usize>,
) -> bool {
    settings.enabled
        && settings.interval_generations > 0
        && generation > 0
        && generation.is_multiple_of(settings.interval_generations)
        && last_auto_saved_generation != Some(generation)
}

pub struct LoadedCheckpoint {
    pub source_filename: String,
    pub architecture: Architecture,
    pub saved: SavedNetwork,
}

#[derive(Resource, Default)]
pub struct LoadedNetwork {
    pub checkpoint: Option<LoadedCheckpoint>,
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::simulation::{
        TrackLibrary,
        training::{EpisodeResult, TrackAdvance},
    };

    fn result(score: f32, speed: f32, reason: FinishReason) -> EpisodeResult {
        EpisodeResult {
            score,
            normalized_progress: score.clamp(0.0, 1.0),
            useful_progress_speed: speed,
            normalized_progress_speed: crate::simulation::training::normalize_useful_progress_speed(
                speed, 120.0,
            ),
            elapsed: 5.0,
            finish_reason: reason,
        }
    }

    fn completed_training_state() -> TrainingState {
        let library = TrackLibrary::load_default().unwrap();
        let mut config = EvaluationConfig::default();
        config.training_track_selection = TrainingTrackSelection::RandomSubset(1);
        let mut training = TrainingState::with_config(2, &library, config).unwrap();
        assert!(matches!(
            training
                .record_training_results(&[
                    result(0.7, 80.0, FinishReason::Completed),
                    result(0.2, 20.0, FinishReason::Collision),
                ])
                .unwrap(),
            TrackAdvance::Validation(_)
        ));
        training
            .record_validation_result(result(0.5, 60.0, FinishReason::Timeout))
            .unwrap();
        training
    }

    fn test_directory(label: &str) -> PathBuf {
        static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "neuroevolution-{label}-{}-{}-{}",
            std::process::id(),
            unix_timestamp_seconds().unwrap(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn remove_test_directory(directory: &Path) {
        if let Ok(entries) = fs::read_dir(directory) {
            for entry in entries.flatten() {
                fs::remove_file(entry.path()).unwrap();
            }
            fs::remove_dir(directory).unwrap();
        }
    }

    #[test]
    fn saved_network_round_trips_through_ron_and_reconstructs_mlp() {
        let training = completed_training_state();
        let saved = SavedNetwork::from_training(&training, &SimulationConfig::default()).unwrap();
        let ron = ron::ser::to_string_pretty(&saved, PrettyConfig::default()).unwrap();
        let parsed = parse_saved_network(&ron).unwrap();
        assert_eq!(parsed, saved);
        let (architecture, mlp) = parsed.reconstruct().unwrap();
        assert_eq!(
            architecture.layer_sizes(),
            &[MLP_INPUT_SIZE, 8, MLP_OUTPUT_SIZE]
        );
        assert_eq!(mlp.parameter_count(), parsed.genome.len());
        assert_eq!(parsed.format_version, 3);
        assert_eq!(
            parsed.evaluation_parameters.useful_speed_normalization,
            Some(SavedUsefulSpeedNormalization::AsymptoticHalfSaturation)
        );
        assert_eq!(
            parsed.evaluation_parameters.progress_speed_half_saturation,
            120.0
        );
        assert_eq!(
            parsed.evaluation_parameters.sensor_max_distance,
            Some(1000.0)
        );
        assert_eq!(
            parsed.evaluation_parameters.laser,
            Some(SavedLaserConfig {
                grace_period: 3.0,
                acceleration: 30.0,
                maximum_speed: 130.0,
            })
        );
    }

    #[test]
    fn v2_hard_clamp_checkpoint_field_name_remains_loadable() {
        let training = completed_training_state();
        let mut v2 = SavedNetwork::from_training(&training, &SimulationConfig::default()).unwrap();
        v2.format_version = HARD_CLAMP_CHECKPOINT_FORMAT_VERSION;
        v2.evaluation_parameters.useful_speed_normalization = None;
        let ron = ron::ser::to_string_pretty(&v2, PrettyConfig::default())
            .unwrap()
            .replace(
                "progress_speed_half_saturation",
                "progress_speed_normalization",
            );

        let loaded = parse_saved_network(&ron).unwrap();
        assert_eq!(loaded.format_version, 2);
        assert_eq!(
            loaded.evaluation_parameters.useful_speed_normalization,
            None
        );
        assert_eq!(
            loaded.evaluation_parameters.progress_speed_half_saturation,
            120.0
        );
        assert!(loaded.reconstruct().is_ok());
    }

    #[test]
    fn legacy_v1_checkpoint_keeps_stall_metadata_and_remains_loadable() {
        let training = completed_training_state();
        let mut legacy =
            SavedNetwork::from_training(&training, &SimulationConfig::default()).unwrap();
        legacy.format_version = LEGACY_CHECKPOINT_FORMAT_VERSION;
        legacy.evaluation_parameters.maximum_episode_duration = 60.0;
        legacy.evaluation_parameters.stall_timeout = 2.0;
        legacy.evaluation_parameters.significant_progress_epsilon = 60.0;
        legacy.evaluation_parameters.laser = None;
        legacy.evaluation_parameters.sensor_max_distance = None;
        legacy.training_metadata.finish_counts.stalled = 1;
        legacy.training_metadata.finish_counts.laser_eliminated = 0;
        legacy.validation_metadata.finish_reason = SavedFinishReason::Stalled;

        let ron = ron::ser::to_string_pretty(&legacy, PrettyConfig::default()).unwrap();
        let loaded = parse_saved_network(&ron).unwrap();
        assert_eq!(loaded.format_version, 1);
        assert_eq!(loaded.evaluation_parameters.stall_timeout, 2.0);
        assert_eq!(loaded.evaluation_parameters.laser, None);
        assert_eq!(loaded.training_metadata.finish_counts.stalled, 1);
        assert_eq!(loaded.validation_metadata.finish_reason.label(), "stalled");
        assert!(loaded.reconstruct().is_ok());
    }

    #[test]
    fn existing_repository_v1_checkpoints_remain_loadable_when_present() {
        let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../checkpoints");
        let Ok(entries) = fs::read_dir(directory) else {
            return;
        };
        let mut loaded_count = 0;
        for path in entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|extension| extension == "ron"))
        {
            let saved = load_saved_network(&path).unwrap_or_else(|error| {
                panic!("failed to load legacy {}: {error}", path.display())
            });
            if saved.format_version == LEGACY_CHECKPOINT_FORMAT_VERSION {
                assert!(saved.reconstruct().is_ok());
                loaded_count += 1;
            }
        }
        if loaded_count == 0 {
            // The repository may contain only newer user-generated checkpoints.
            // V1 parsing is covered by the synthetic fixture above.
            return;
        }
    }

    #[test]
    fn invalid_genome_non_finite_value_version_and_contract_are_rejected() {
        let training = completed_training_state();
        let valid = SavedNetwork::from_training(&training, &SimulationConfig::default()).unwrap();

        let mut wrong_length = valid.clone();
        wrong_length.genome.pop();
        assert!(wrong_length.validate().is_err());

        let mut nan = valid.clone();
        nan.genome[0] = f32::NAN;
        assert!(nan.validate().is_err());

        let mut infinity = valid.clone();
        infinity.validation_metadata.score = f32::INFINITY;
        assert!(infinity.validate().is_err());

        let mut unknown_version = valid.clone();
        unknown_version.format_version += 1;
        assert!(unknown_version.validate().is_err());

        let mut wrong_input = valid.clone();
        wrong_input.architecture.layer_sizes[0] = MLP_INPUT_SIZE - 1;
        assert!(wrong_input.validate().is_err());

        let mut wrong_output = valid;
        *wrong_output.architecture.layer_sizes.last_mut().unwrap() = MLP_OUTPUT_SIZE + 1;
        assert!(wrong_output.validate().is_err());
    }

    #[test]
    fn invalid_file_returns_readable_error_without_panicking() {
        let error = parse_saved_network("this is not RON").unwrap_err();
        assert!(error.to_string().contains("RON parse error"));

        let directory = test_directory("invalid-checkpoint-test");
        fs::create_dir(&directory).unwrap();
        fs::write(directory.join("broken.ron"), "this is not RON").unwrap();
        let store = CheckpointStore::new(directory.clone());
        assert_eq!(store.entries.len(), 1);
        assert!(store.entries[0].summary.is_none());
        assert!(
            store.entries[0]
                .error
                .as_deref()
                .unwrap()
                .contains("RON parse error")
        );
        remove_test_directory(&directory);
    }

    #[test]
    fn auto_save_requires_interval_completed_validation_and_new_generation() {
        let settings = AutoSaveSettings {
            enabled: true,
            interval_generations: 10,
        };
        assert!(!auto_save_is_due(&settings, 0, None));
        assert!(!auto_save_is_due(&settings, 9, None));
        assert!(auto_save_is_due(&settings, 10, None));
        assert!(!auto_save_is_due(&settings, 10, Some(10)));
        assert!(auto_save_is_due(&settings, 20, Some(10)));
        let mut fast_forward = crate::simulation::TrainingFastForward::default();
        assert!(fast_forward.start(200, 50, 2.0, std::time::Duration::from_millis(250)));
        assert!(auto_save_is_due(&settings, 100, Some(90)));
        assert!(!auto_save_is_due(
            &AutoSaveSettings {
                enabled: false,
                ..settings
            },
            20,
            None
        ));

        let library = TrackLibrary::load_default().unwrap();
        let mut config = EvaluationConfig::default();
        config.training_track_selection = TrainingTrackSelection::RandomSubset(1);
        let training = TrainingState::with_config(2, &library, config).unwrap();
        assert!(training.completed_champion().is_none());
    }

    #[test]
    fn auto_save_writes_once_only_after_due_generation_validation() {
        let library = TrackLibrary::load_default().unwrap();
        let mut config = EvaluationConfig::default();
        config.training_track_selection = TrainingTrackSelection::RandomSubset(1);
        let mut training = TrainingState::with_config(2, &library, config).unwrap();
        let directory = test_directory("auto-save-test");
        let mut store = CheckpointStore::new(directory.clone());
        store.settings.interval_generations = 10;

        for _ in 0..10 {
            training
                .record_training_results(&[
                    result(0.7, 80.0, FinishReason::Completed),
                    result(0.2, 20.0, FinishReason::Collision),
                ])
                .unwrap();
            training
                .record_validation_result(result(0.5, 60.0, FinishReason::Timeout))
                .unwrap();
            assert!(
                store
                    .auto_save_if_due(&training, &SimulationConfig::default())
                    .unwrap()
                    .is_none()
            );
            training.evolve_generation().unwrap();
        }
        assert_eq!(training.generation(), 10);

        training
            .record_training_results(&[
                result(0.8, 90.0, FinishReason::Completed),
                result(0.3, 30.0, FinishReason::EliminatedByLaser),
            ])
            .unwrap();
        assert!(
            store
                .auto_save_if_due(&training, &SimulationConfig::default())
                .unwrap()
                .is_none()
        );
        training
            .record_validation_result(result(0.6, 70.0, FinishReason::Completed))
            .unwrap();
        assert!(
            store
                .auto_save_if_due(&training, &SimulationConfig::default())
                .unwrap()
                .is_some()
        );
        assert!(
            store
                .auto_save_if_due(&training, &SimulationConfig::default())
                .unwrap()
                .is_none()
        );
        assert_eq!(store.entries.len(), 1);

        remove_test_directory(&directory);
    }

    #[test]
    fn loading_checkpoint_does_not_mutate_training_state() {
        let training = completed_training_state();
        let generation = training.generation();
        let fitness = training
            .population()
            .individuals()
            .iter()
            .map(|individual| individual.fitness())
            .collect::<Vec<_>>();
        let population_genomes = training
            .population()
            .individuals()
            .iter()
            .map(|individual| individual.genome().genes().to_vec())
            .collect::<Vec<_>>();
        let directory = test_directory("load-test");
        let mut store = CheckpointStore::new(directory.clone());
        store
            .save_current_champion(&training, &SimulationConfig::default())
            .unwrap();
        let loaded = store.load(0).unwrap();
        let mlp = Mlp::from_parameters(&loaded.architecture, &loaded.saved.genome).unwrap();
        let controller = MlpController::new(mlp, &loaded.saved.genome);

        assert!(controller.is_ok());
        assert_eq!(loaded.architecture.input_size(), MLP_INPUT_SIZE);
        assert_eq!(training.generation(), generation);
        assert_eq!(
            training
                .population()
                .individuals()
                .iter()
                .map(|individual| individual.fitness())
                .collect::<Vec<_>>(),
            fitness
        );
        assert_eq!(
            training
                .population()
                .individuals()
                .iter()
                .map(|individual| individual.genome().genes().to_vec())
                .collect::<Vec<_>>(),
            population_genomes
        );
        remove_test_directory(&directory);
    }
}
