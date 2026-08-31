use super::{
    TrackLibrary,
    controller::{MLP_INPUT_SIZE, MLP_OUTPUT_SIZE},
};
use bevy::prelude::{Component, Resource};
use neuroevolution::{
    genetic::{Config, Genome, Individual, Population},
    neural::{Activation, Architecture},
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

const EVALUATION_RNG_SALT: u64 = 0x4556_414c_5541_5445;
const DEFAULT_TRAINING_TRACKS_PER_GENERATION: Option<usize> = Some(3);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Completed,
    Collision,
    EliminatedByLaser,
    Timeout,
}

impl FinishReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Collision => "collision",
            Self::EliminatedByLaser => "eliminated by laser",
            Self::Timeout => "timeout",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FinishReasonCounts {
    pub completed: usize,
    pub collision: usize,
    pub laser_eliminated: usize,
    pub timeout: usize,
}

impl FinishReasonCounts {
    pub fn record(&mut self, reason: FinishReason) {
        match reason {
            FinishReason::Completed => self.completed += 1,
            FinishReason::Collision => self.collision += 1,
            FinishReason::EliminatedByLaser => self.laser_eliminated += 1,
            FinishReason::Timeout => self.timeout += 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingTrackSelection {
    All,
    RandomSubset(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LaserConfig {
    pub grace_period: f32,
    pub acceleration: f32,
    pub maximum_speed: f32,
}

impl Default for LaserConfig {
    fn default() -> Self {
        Self {
            grace_period: 3.0,
            acceleration: 30.0,
            maximum_speed: 130.0,
        }
    }
}

impl LaserConfig {
    fn validate(self) -> Result<(), String> {
        for (name, value) in [
            ("laser grace_period", self.grace_period),
            ("laser acceleration", self.acceleration),
            ("laser maximum_speed", self.maximum_speed),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be finite and non-negative"));
            }
        }
        if self.acceleration <= 0.0 || self.maximum_speed <= 0.0 {
            return Err("laser acceleration and maximum_speed must be greater than zero".into());
        }
        Ok(())
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct LaserState {
    pub elapsed: f32,
    /// Accumulated track progress at the episode spawn point.
    pub origin_progress: f32,
    /// Laser progress relative to `origin_progress`.
    pub progress: f32,
    pub speed: f32,
}

impl LaserState {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn reset_at(&mut self, origin_progress: f32) {
        *self = Self {
            origin_progress: if origin_progress.is_finite() {
                origin_progress.max(0.0)
            } else {
                0.0
            },
            ..Self::default()
        };
    }

    pub fn track_progress(self) -> f32 {
        self.origin_progress + self.progress
    }

    /// Advances from elapsed simulation time using the exact constant-acceleration
    /// solution, then a constant-speed segment. This avoids integration drift.
    pub fn advance(&mut self, delta_seconds: f32, track_length: f32, config: LaserConfig) {
        self.elapsed += delta_seconds.max(0.0);
        let laser_time = (self.elapsed - config.grace_period).max(0.0);
        let acceleration_time = config.maximum_speed / config.acceleration;
        if laser_time <= acceleration_time {
            self.speed = config.acceleration * laser_time;
            self.progress = 0.5 * config.acceleration * laser_time * laser_time;
        } else {
            self.speed = config.maximum_speed;
            let acceleration_distance =
                0.5 * config.acceleration * acceleration_time * acceleration_time;
            self.progress =
                acceleration_distance + config.maximum_speed * (laser_time - acceleration_time);
        }
        self.speed = self.speed.min(config.maximum_speed);
        self.progress = self.progress.clamp(0.0, track_length.max(0.0));
    }

    pub fn is_active(self, config: LaserConfig) -> bool {
        self.elapsed > config.grace_period
    }

    pub fn has_reached(
        self,
        current_track_progress: f32,
        initial_progress: f32,
        config: LaserConfig,
    ) -> bool {
        let car_relative_progress = current_track_progress - initial_progress;
        self.is_active(config) && car_relative_progress <= self.progress
    }
}

/// Parameters for episode termination and the deliberately small score formula.
/// Progress is dominant, useful speed is a tie-breaker, and collision is a bounded penalty.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub maximum_episode_duration: f32,
    pub laser: LaserConfig,
    pub progress_weight: f32,
    pub speed_weight: f32,
    pub collision_penalty: f32,
    pub completion_bonus: f32,
    /// Useful speed at which the asymptotic speed term reaches 0.5.
    pub progress_speed_half_saturation: f32,
    pub training_track_selection: TrainingTrackSelection,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            maximum_episode_duration: 180.0,
            laser: LaserConfig::default(),
            progress_weight: 1.0,
            speed_weight: 0.40,
            collision_penalty: 0.08,
            completion_bonus: 0.45,
            progress_speed_half_saturation: 120.0,
            training_track_selection: match DEFAULT_TRAINING_TRACKS_PER_GENERATION {
                Some(count) => TrainingTrackSelection::RandomSubset(count),
                None => TrainingTrackSelection::All,
            },
        }
    }
}

impl EvaluationConfig {
    pub fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("maximum_episode_duration", self.maximum_episode_duration),
            (
                "progress_speed_half_saturation",
                self.progress_speed_half_saturation,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and greater than zero"));
            }
        }
        self.laser.validate()?;
        for (name, value) in [
            ("progress_weight", self.progress_weight),
            ("speed_weight", self.speed_weight),
            ("collision_penalty", self.collision_penalty),
            ("completion_bonus", self.completion_bonus),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(format!("{name} must be finite and non-negative"));
            }
        }
        if self.completion_bonus <= self.speed_weight {
            return Err(
                "completion_bonus must be greater than speed_weight so completion always wins"
                    .into(),
            );
        }
        if matches!(
            self.training_track_selection,
            TrainingTrackSelection::RandomSubset(0)
        ) {
            return Err("training track subset size must be greater than zero".into());
        }
        Ok(())
    }
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct EvaluationState {
    pub finish_reason: Option<FinishReason>,
    pub elapsed: f32,
    pub initial_progress: f32,
}

impl EvaluationState {
    pub fn new(initial_progress: f32) -> Self {
        Self {
            finish_reason: None,
            elapsed: 0.0,
            initial_progress,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.finish_reason.is_some()
    }

    pub fn finish(&mut self, reason: FinishReason) {
        if self.finish_reason.is_none() {
            self.finish_reason = Some(reason);
        }
    }

    /// Advances termination logic after track progress has been updated for the tick.
    pub fn update(
        &mut self,
        delta_seconds: f32,
        current_track_distance: f32,
        best_track_distance: f32,
        total_track_length: f32,
        laser: &LaserState,
        config: &EvaluationConfig,
    ) {
        if self.is_finished() {
            return;
        }
        self.elapsed += delta_seconds.max(0.0);

        // The accumulated tracker reaches total_length only by crossing the lap boundary
        // in the forward direction; regression/wrap-around therefore cannot complete a lap.
        if best_track_distance >= total_track_length {
            self.finish(FinishReason::Completed);
            return;
        }

        if laser.has_reached(current_track_distance, self.initial_progress, config.laser) {
            self.finish(FinishReason::EliminatedByLaser);
        } else if self.elapsed >= config.maximum_episode_duration {
            self.finish(FinishReason::Timeout);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpisodeResult {
    pub score: f32,
    pub normalized_progress: f32,
    /// Effective forward progress along the track, in simulation units/second.
    pub useful_progress_speed: f32,
    pub normalized_progress_speed: f32,
    pub elapsed: f32,
    pub finish_reason: FinishReason,
}

/// Maps useful forward-progress speed to `[0, 1)` with `speed = k` at 0.5.
/// Invalid inputs are neutral rather than contaminating population fitness.
pub fn normalize_useful_progress_speed(useful_speed: f32, half_saturation: f32) -> f32 {
    if !half_saturation.is_finite() || half_saturation <= 0.0 || useful_speed.is_nan() {
        return 0.0;
    }
    if useful_speed == f32::INFINITY {
        return 1.0;
    }
    if !useful_speed.is_finite() || useful_speed <= 0.0 {
        return 0.0;
    }

    // Algebraically speed / (speed + k), written this way to avoid overflow.
    // Keep every finite speed strictly below one even after f32 rounding.
    const MAX_BELOW_ONE: f32 = f32::from_bits(1.0_f32.to_bits() - 1);
    (1.0 / (1.0 + half_saturation / useful_speed)).min(MAX_BELOW_ONE)
}

/// Computes one track score. Progress starts at zero at the episode spawn and is
/// normalized by the remaining lap. Useful speed uses only new best distance,
/// never the car's raw/absolute velocity.
pub fn episode_score(
    state: &EvaluationState,
    best_track_distance: f32,
    total_track_length: f32,
    config: &EvaluationConfig,
) -> EpisodeResult {
    let useful_distance = (best_track_distance - state.initial_progress).max(0.0);
    let remaining_lap_distance = (total_track_length - state.initial_progress).max(0.0);
    let normalized_progress = if remaining_lap_distance > f32::EPSILON {
        (useful_distance / remaining_lap_distance).clamp(0.0, 1.0)
    } else if best_track_distance >= total_track_length {
        1.0
    } else {
        0.0
    };
    let useful_speed = if state.elapsed > f32::EPSILON {
        useful_distance / state.elapsed
    } else {
        0.0
    };
    let normalized_progress_speed =
        normalize_useful_progress_speed(useful_speed, config.progress_speed_half_saturation);
    let finish_reason = state
        .finish_reason
        .expect("episode score requires a finished evaluation");
    let collision_penalty = if finish_reason == FinishReason::Collision {
        config.collision_penalty
    } else {
        0.0
    };
    let completion_bonus = if finish_reason == FinishReason::Completed {
        config.completion_bonus
    } else {
        0.0
    };
    let score = config.progress_weight * normalized_progress
        + config.speed_weight * normalized_progress_speed
        + completion_bonus
        - collision_penalty;

    EpisodeResult {
        score,
        normalized_progress,
        useful_progress_speed: useful_speed,
        normalized_progress_speed,
        elapsed: state.elapsed,
        finish_reason,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationStats {
    pub generation: usize,
    pub best_fitness: f32,
    pub average_fitness: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrackEvaluationStats {
    pub track_id: String,
    pub best_score: f32,
    pub average_score: f32,
    pub average_useful_progress_speed: f32,
    pub completion_rate: f32,
    pub finish_counts: FinishReasonCounts,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChampionTrainingStats {
    pub training_fitness: f32,
    pub population_average_fitness: f32,
    pub average_useful_progress_speed: f32,
    pub completion_rate: f32,
    pub finish_counts: FinishReasonCounts,
    pub training_tracks: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValidationStats {
    pub generation: usize,
    pub track_id: String,
    pub score: f32,
    pub normalized_progress: f32,
    pub useful_progress_speed: f32,
    pub elapsed: f32,
    pub finish_reason: FinishReason,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompletedChampion {
    pub generation: usize,
    pub genome: Vec<f32>,
    pub training: ChampionTrainingStats,
    pub track_stats: Vec<TrackEvaluationStats>,
    pub validation: ValidationStats,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrainingPhase {
    TrainingTrack {
        track_id: String,
        index: usize,
        total: usize,
    },
    Validation {
        track_id: String,
    },
    Evolving,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TrackAdvance {
    Training(String),
    Validation(String),
    ReadyToEvolve,
}

#[derive(Resource)]
pub struct TrainingState {
    architecture: Architecture,
    genetic_config: Config,
    evaluation_config: EvaluationConfig,
    population: Population,
    evolution_rng: ChaCha12Rng,
    evaluation_rng: ChaCha12Rng,
    training_track_ids: Vec<String>,
    validation_track_ids: Vec<String>,
    selected_training_tracks: Vec<String>,
    training_score_sums: Vec<f32>,
    training_useful_speed_sums: Vec<f32>,
    individual_finish_counts: Vec<FinishReasonCounts>,
    current_track_stats: Vec<TrackEvaluationStats>,
    completed_training_tracks: usize,
    phase: TrainingPhase,
    history: Vec<GenerationStats>,
    validation_history: Vec<ValidationStats>,
    pending_stats: Option<GenerationStats>,
    champion_genome: Option<Vec<f32>>,
    champion_population_index: Option<usize>,
    pending_champion_training: Option<ChampionTrainingStats>,
    completed_champion: Option<CompletedChampion>,
}

pub const TRAINING_CHECKPOINT_FORMAT_VERSION: u32 = 1;
pub const TRAINING_RNG_ID: &str = "rand_chacha::ChaCha12Rng/0.10";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedGeneticConfig {
    pub population_size: usize,
    pub genome_length: usize,
    pub elite_fraction: f32,
    pub tournament_size: usize,
    pub crossover_probability: f32,
    pub mutation_probability: f32,
    pub mutation_sigma: f32,
    pub initial_gene_min: f32,
    pub initial_gene_max: f32,
    pub seed: u64,
}

impl SavedGeneticConfig {
    pub fn from_config(config: &Config) -> Self {
        Self {
            population_size: config.population_size,
            genome_length: config.genome_length,
            elite_fraction: config.elite_fraction,
            tournament_size: config.tournament_size,
            crossover_probability: config.crossover_probability,
            mutation_probability: config.mutation_probability,
            mutation_sigma: config.mutation_sigma,
            initial_gene_min: config.initial_gene_min,
            initial_gene_max: config.initial_gene_max,
            seed: config.seed,
        }
    }

    fn to_config(&self) -> Result<Config, String> {
        let config = Config {
            population_size: self.population_size,
            genome_length: self.genome_length,
            elite_fraction: self.elite_fraction,
            tournament_size: self.tournament_size,
            crossover_probability: self.crossover_probability,
            mutation_probability: self.mutation_probability,
            mutation_sigma: self.mutation_sigma,
            initial_gene_min: self.initial_gene_min,
            initial_gene_max: self.initial_gene_max,
            seed: self.seed,
        };
        config.validate().map_err(str::to_string)?;
        Ok(config)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedTrainingIndividual {
    pub genome: Vec<f32>,
    pub fitness: Option<f32>,
}

/// Clean-boundary snapshot: generation N is fully prepared but has not received its first tick.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingCheckpoint {
    pub format_version: u32,
    pub architecture: Vec<usize>,
    pub generation: usize,
    pub individuals: Vec<SavedTrainingIndividual>,
    pub genetic_config: SavedGeneticConfig,
    pub evaluation_config: EvaluationConfig,
    pub seed: u64,
    pub rng_id: String,
    pub evolution_rng: ChaCha12Rng,
    pub evaluation_rng: ChaCha12Rng,
    pub selected_training_tracks: Vec<String>,
}

impl TrainingState {
    #[cfg(test)]
    pub fn with_config(
        population_size: usize,
        library: &TrackLibrary,
        evaluation_config: EvaluationConfig,
    ) -> Result<Self, String> {
        Self::with_architecture(
            population_size,
            library,
            evaluation_config,
            vec![MLP_INPUT_SIZE, 8, MLP_OUTPUT_SIZE],
            Config::default().seed,
        )
    }

    pub fn with_architecture(
        population_size: usize,
        library: &TrackLibrary,
        evaluation_config: EvaluationConfig,
        layer_sizes: Vec<usize>,
        seed: u64,
    ) -> Result<Self, String> {
        evaluation_config.validate()?;
        let training_track_ids = library
            .training_tracks()
            .map(|track| track.id.clone())
            .collect::<Vec<_>>();
        let validation_track_ids = library
            .validation_tracks()
            .map(|track| track.id.clone())
            .collect::<Vec<_>>();
        if training_track_ids.is_empty() || validation_track_ids.is_empty() {
            return Err("training and validation track suites must not be empty".into());
        }

        let architecture = racing_architecture(layer_sizes)?;
        let genetic_config = Config {
            population_size,
            genome_length: architecture.parameter_count(),
            seed,
            ..Config::default()
        };
        let mut evolution_rng = ChaCha12Rng::seed_from_u64(genetic_config.seed);
        let population = Population::new(&genetic_config, &mut evolution_rng)?;
        let evaluation_rng = ChaCha12Rng::seed_from_u64(genetic_config.seed ^ EVALUATION_RNG_SALT);
        let mut state = Self {
            architecture,
            genetic_config,
            evaluation_config,
            population,
            evolution_rng,
            evaluation_rng,
            training_track_ids,
            validation_track_ids,
            selected_training_tracks: Vec::new(),
            training_score_sums: vec![0.0; population_size],
            training_useful_speed_sums: vec![0.0; population_size],
            individual_finish_counts: vec![FinishReasonCounts::default(); population_size],
            current_track_stats: Vec::new(),
            completed_training_tracks: 0,
            phase: TrainingPhase::Evolving,
            history: Vec::new(),
            validation_history: Vec::new(),
            pending_stats: None,
            champion_genome: None,
            champion_population_index: None,
            pending_champion_training: None,
            completed_champion: None,
        };
        state.start_generation();
        Ok(state)
    }

    pub fn training_checkpoint(&self) -> TrainingCheckpoint {
        TrainingCheckpoint {
            format_version: TRAINING_CHECKPOINT_FORMAT_VERSION,
            architecture: self.architecture.layer_sizes().to_vec(),
            generation: self.population.generation(),
            individuals: self
                .population
                .individuals()
                .iter()
                .map(|individual| SavedTrainingIndividual {
                    genome: individual.genome().genes().to_vec(),
                    fitness: individual.fitness(),
                })
                .collect(),
            genetic_config: SavedGeneticConfig::from_config(&self.genetic_config),
            evaluation_config: self.evaluation_config.clone(),
            seed: self.genetic_config.seed,
            rng_id: TRAINING_RNG_ID.into(),
            evolution_rng: self.evolution_rng.clone(),
            evaluation_rng: self.evaluation_rng.clone(),
            selected_training_tracks: self.selected_training_tracks.clone(),
        }
    }

    pub fn from_training_checkpoint(
        checkpoint: TrainingCheckpoint,
        library: &TrackLibrary,
    ) -> Result<Self, String> {
        checkpoint.validate(library)?;
        let architecture = racing_architecture(checkpoint.architecture.clone())?;
        let genetic_config = checkpoint.genetic_config.to_config()?;
        let individuals = checkpoint
            .individuals
            .into_iter()
            .map(|saved| {
                Individual::from_parts(Genome::new(saved.genome), saved.fitness)
                    .map_err(str::to_string)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let population = Population::from_individuals(individuals, checkpoint.generation)
            .map_err(str::to_string)?;
        let training_track_ids = library
            .training_tracks()
            .map(|track| track.id.clone())
            .collect::<Vec<_>>();
        let validation_track_ids = library
            .validation_tracks()
            .map(|track| track.id.clone())
            .collect::<Vec<_>>();
        let population_size = population.len();
        let first_track = checkpoint.selected_training_tracks[0].clone();
        let total_tracks = checkpoint.selected_training_tracks.len();
        Ok(Self {
            architecture,
            genetic_config,
            evaluation_config: checkpoint.evaluation_config,
            population,
            evolution_rng: checkpoint.evolution_rng,
            evaluation_rng: checkpoint.evaluation_rng,
            training_track_ids,
            validation_track_ids,
            selected_training_tracks: checkpoint.selected_training_tracks,
            training_score_sums: vec![0.0; population_size],
            training_useful_speed_sums: vec![0.0; population_size],
            individual_finish_counts: vec![FinishReasonCounts::default(); population_size],
            current_track_stats: Vec::new(),
            completed_training_tracks: 0,
            phase: TrainingPhase::TrainingTrack {
                track_id: first_track,
                index: 0,
                total: total_tracks,
            },
            history: Vec::new(),
            validation_history: Vec::new(),
            pending_stats: None,
            champion_genome: None,
            champion_population_index: None,
            pending_champion_training: None,
            completed_champion: None,
        })
    }

    fn start_generation(&mut self) {
        self.selected_training_tracks = self.training_track_ids.clone();
        if let TrainingTrackSelection::RandomSubset(count) =
            self.evaluation_config.training_track_selection
        {
            for index in (1..self.selected_training_tracks.len()).rev() {
                let swap_with = self.evaluation_rng.random_range(0..=index);
                self.selected_training_tracks.swap(index, swap_with);
            }
            self.selected_training_tracks
                .truncate(count.min(self.selected_training_tracks.len()));
        }
        self.training_score_sums.fill(0.0);
        self.training_useful_speed_sums.fill(0.0);
        self.individual_finish_counts
            .fill(FinishReasonCounts::default());
        self.current_track_stats.clear();
        self.completed_training_tracks = 0;
        self.pending_stats = None;
        self.pending_champion_training = None;
        self.champion_population_index = None;
        self.phase = TrainingPhase::TrainingTrack {
            track_id: self.selected_training_tracks[0].clone(),
            index: 0,
            total: self.selected_training_tracks.len(),
        };
    }

    pub fn record_training_results(
        &mut self,
        results: &[EpisodeResult],
    ) -> Result<TrackAdvance, String> {
        let TrainingPhase::TrainingTrack {
            track_id,
            index,
            total,
        } = self.phase.clone()
        else {
            return Err("training results can only be recorded during a training track".into());
        };
        if results.len() != self.population.len() {
            return Err(format!(
                "expected {} episode results, got {}",
                self.population.len(),
                results.len()
            ));
        }
        let mut track_counts = FinishReasonCounts::default();
        let mut track_score_sum = 0.0;
        let mut track_useful_speed_sum = 0.0;
        let mut track_best_score = f32::NEG_INFINITY;
        for (individual_index, result) in results.iter().enumerate() {
            if !result.score.is_finite() || !result.useful_progress_speed.is_finite() {
                return Err(format!(
                    "episode metrics for individual {individual_index} are not finite"
                ));
            }
            self.training_score_sums[individual_index] += result.score;
            self.training_useful_speed_sums[individual_index] += result.useful_progress_speed;
            self.individual_finish_counts[individual_index].record(result.finish_reason);
            track_counts.record(result.finish_reason);
            track_score_sum += result.score;
            track_useful_speed_sum += result.useful_progress_speed;
            track_best_score = track_best_score.max(result.score);
        }
        self.current_track_stats.push(TrackEvaluationStats {
            track_id,
            best_score: track_best_score,
            average_score: track_score_sum / results.len() as f32,
            average_useful_progress_speed: track_useful_speed_sum / results.len() as f32,
            completion_rate: track_counts.completed as f32 / results.len() as f32,
            finish_counts: track_counts,
        });
        self.completed_training_tracks += 1;

        if index + 1 < total {
            let next_index = index + 1;
            let track_id = self.selected_training_tracks[next_index].clone();
            self.phase = TrainingPhase::TrainingTrack {
                track_id: track_id.clone(),
                index: next_index,
                total,
            };
            return Ok(TrackAdvance::Training(track_id));
        }

        for (individual, score_sum) in self
            .population
            .individuals_mut()
            .iter_mut()
            .zip(&self.training_score_sums)
        {
            individual.set_fitness(*score_sum / self.completed_training_tracks as f32);
        }
        let (champion_population_index, champion) = self
            .population
            .individuals()
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| {
                left.fitness().unwrap().total_cmp(&right.fitness().unwrap())
            })
            .ok_or_else(|| "population must not be empty".to_string())?;
        let best_fitness = champion.fitness().unwrap();
        let average_fitness = self
            .population
            .individuals()
            .iter()
            .map(|individual| individual.fitness().unwrap())
            .sum::<f32>()
            / self.population.len() as f32;
        self.pending_stats = Some(GenerationStats {
            generation: self.population.generation(),
            best_fitness,
            average_fitness,
        });
        self.champion_genome = Some(champion.genome().genes().to_vec());
        self.champion_population_index = Some(champion_population_index);
        let champion_counts = self.individual_finish_counts[champion_population_index];
        self.pending_champion_training = Some(ChampionTrainingStats {
            training_fitness: best_fitness,
            population_average_fitness: average_fitness,
            average_useful_progress_speed: self.training_useful_speed_sums
                [champion_population_index]
                / self.completed_training_tracks as f32,
            completion_rate: champion_counts.completed as f32
                / self.completed_training_tracks as f32,
            finish_counts: champion_counts,
            training_tracks: self.selected_training_tracks.clone(),
        });

        let validation_index = self
            .evaluation_rng
            .random_range(0..self.validation_track_ids.len());
        let track_id = self.validation_track_ids[validation_index].clone();
        self.phase = TrainingPhase::Validation {
            track_id: track_id.clone(),
        };
        Ok(TrackAdvance::Validation(track_id))
    }

    pub fn record_validation_result(
        &mut self,
        result: EpisodeResult,
    ) -> Result<TrackAdvance, String> {
        let TrainingPhase::Validation { track_id } = &self.phase else {
            return Err("validation result can only be recorded during validation".into());
        };
        self.validation_history.push(ValidationStats {
            generation: self.population.generation(),
            track_id: track_id.clone(),
            score: result.score,
            normalized_progress: result.normalized_progress,
            useful_progress_speed: result.useful_progress_speed,
            elapsed: result.elapsed,
            finish_reason: result.finish_reason,
        });
        self.completed_champion = Some(CompletedChampion {
            generation: self.population.generation(),
            genome: self
                .champion_genome
                .clone()
                .ok_or_else(|| "champion genome must exist before validation".to_string())?,
            training: self.pending_champion_training.clone().ok_or_else(|| {
                "champion training metrics must exist before validation".to_string()
            })?,
            track_stats: self.current_track_stats.clone(),
            validation: self
                .validation_history
                .last()
                .cloned()
                .ok_or_else(|| "validation metrics were not recorded".to_string())?,
        });
        self.phase = TrainingPhase::Evolving;
        Ok(TrackAdvance::ReadyToEvolve)
    }

    pub fn evolve_generation(&mut self) -> Result<GenerationStats, &'static str> {
        if self.phase != TrainingPhase::Evolving {
            return Err("generation can only evolve after held-out validation");
        }
        let stats = self
            .pending_stats
            .ok_or("training fitness must be finalized before evolution")?;
        let next_population = self
            .population
            .evolve(&self.genetic_config, &mut self.evolution_rng)?;
        self.population = next_population;
        self.history.push(stats);
        self.start_generation();
        Ok(stats)
    }

    pub fn architecture(&self) -> &Architecture {
        &self.architecture
    }

    pub fn evaluation_config(&self) -> &EvaluationConfig {
        &self.evaluation_config
    }

    pub fn population(&self) -> &Population {
        &self.population
    }

    pub fn generation(&self) -> usize {
        self.population.generation()
    }

    pub fn history(&self) -> &[GenerationStats] {
        &self.history
    }

    pub fn latest_validation(&self) -> Option<&ValidationStats> {
        self.validation_history.last()
    }

    pub fn champion_genome(&self) -> Option<&[f32]> {
        self.champion_genome.as_deref()
    }

    pub fn champion_population_index(&self) -> Option<usize> {
        self.champion_population_index
    }

    pub fn completed_champion(&self) -> Option<&CompletedChampion> {
        self.completed_champion.as_ref()
    }

    #[cfg(test)]
    pub fn selected_training_tracks(&self) -> &[String] {
        &self.selected_training_tracks
    }

    pub fn phase(&self) -> &TrainingPhase {
        &self.phase
    }

    pub fn current_track_id(&self) -> Option<&str> {
        match &self.phase {
            TrainingPhase::TrainingTrack { track_id, .. }
            | TrainingPhase::Validation { track_id } => Some(track_id),
            TrainingPhase::Evolving => None,
        }
    }

    pub fn current_training_fitness(&self) -> Option<GenerationStats> {
        self.pending_stats
    }
}

pub fn racing_architecture(layer_sizes: Vec<usize>) -> Result<Architecture, String> {
    if layer_sizes.len() < 3 {
        return Err("architecture must contain at least one hidden layer".into());
    }
    if layer_sizes.first().copied() != Some(MLP_INPUT_SIZE) {
        return Err(format!("architecture input must be {MLP_INPUT_SIZE}"));
    }
    if layer_sizes.last().copied() != Some(MLP_OUTPUT_SIZE) {
        return Err(format!("architecture output must be {MLP_OUTPUT_SIZE}"));
    }
    if layer_sizes[1..layer_sizes.len() - 1]
        .iter()
        .any(|size| *size == 0)
    {
        return Err("hidden layer sizes must be greater than zero".into());
    }
    let activations = vec![Activation::Tanh; layer_sizes.len() - 1];
    Architecture::new(layer_sizes, activations).map_err(str::to_string)
}

impl TrainingCheckpoint {
    pub fn validate(&self, library: &TrackLibrary) -> Result<(), String> {
        if self.format_version != TRAINING_CHECKPOINT_FORMAT_VERSION {
            return Err(format!(
                "unsupported training checkpoint version {}",
                self.format_version
            ));
        }
        if self.rng_id != TRAINING_RNG_ID {
            return Err(format!("unsupported RNG {}", self.rng_id));
        }
        let architecture = racing_architecture(self.architecture.clone())?;
        let config = self.genetic_config.to_config()?;
        if self.seed != config.seed {
            return Err("checkpoint seed differs from genetic config seed".into());
        }
        if config.genome_length != architecture.parameter_count() {
            return Err("genome length differs from architecture parameter count".into());
        }
        if self.individuals.len() != config.population_size {
            return Err("checkpoint population size differs from genetic config".into());
        }
        for (index, individual) in self.individuals.iter().enumerate() {
            if individual.genome.len() != config.genome_length {
                return Err(format!(
                    "individual {index} has an incompatible genome length"
                ));
            }
            if individual.genome.iter().any(|value| !value.is_finite())
                || individual.fitness.is_some_and(|value| !value.is_finite())
            {
                return Err(format!("individual {index} contains a non-finite value"));
            }
        }
        self.evaluation_config.validate()?;
        if self.selected_training_tracks.is_empty() {
            return Err("checkpoint has no selected training tracks".into());
        }
        if self.selected_training_tracks.iter().any(|id| {
            library
                .definition(id)
                .is_none_or(|definition| definition.role != super::track::TrackRole::Training)
        }) {
            return Err("checkpoint contains an unknown non-training track".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finished_state(reason: FinishReason, elapsed: f32) -> EvaluationState {
        finished_state_from(reason, elapsed, 0.0)
    }

    fn finished_state_from(
        reason: FinishReason,
        elapsed: f32,
        initial_progress: f32,
    ) -> EvaluationState {
        EvaluationState {
            finish_reason: Some(reason),
            elapsed,
            initial_progress,
        }
    }

    fn result(score: f32) -> EpisodeResult {
        EpisodeResult {
            score,
            normalized_progress: score.clamp(0.0, 1.0),
            useful_progress_speed: score.max(0.0) * 10.0,
            normalized_progress_speed: 0.0,
            elapsed: 1.0,
            finish_reason: FinishReason::EliminatedByLaser,
        }
    }

    fn measured_result(
        score: f32,
        useful_progress_speed: f32,
        finish_reason: FinishReason,
    ) -> EpisodeResult {
        EpisodeResult {
            score,
            normalized_progress: score.clamp(0.0, 1.0),
            useful_progress_speed,
            normalized_progress_speed: normalize_useful_progress_speed(
                useful_progress_speed,
                120.0,
            ),
            elapsed: 2.0,
            finish_reason,
        }
    }

    #[test]
    fn evaluation_defaults_use_laser_and_high_failsafe_timeout() {
        let config = EvaluationConfig::default();
        assert_eq!(config.maximum_episode_duration, 180.0);
        assert_eq!(config.laser.grace_period, 3.0);
        assert_eq!(config.laser.acceleration, 30.0);
        assert_eq!(config.laser.maximum_speed, 130.0);
        assert_eq!(config.progress_weight, 1.0);
        assert_eq!(config.speed_weight, 0.40);
        assert_eq!(config.collision_penalty, 0.08);
        assert_eq!(config.completion_bonus, 0.45);
        assert_eq!(config.progress_speed_half_saturation, 120.0);
    }

    #[test]
    fn collision_finishes_episode_immediately() {
        let mut state = EvaluationState::new(0.0);
        state.finish(FinishReason::Collision);
        state.update(
            1.0,
            10.0,
            10.0,
            100.0,
            &LaserState::default(),
            &EvaluationConfig::default(),
        );
        assert_eq!(state.finish_reason, Some(FinishReason::Collision));
        assert_eq!(state.elapsed, 0.0);
    }

    #[test]
    fn emergency_timeout_still_terminates_an_episode() {
        let mut config = EvaluationConfig::default();
        config.maximum_episode_duration = 1.0;
        let mut timed_out = EvaluationState::new(0.0);
        timed_out.update(1.0, 10.0, 10.0, 100.0, &LaserState::default(), &config);
        assert_eq!(timed_out.finish_reason, Some(FinishReason::Timeout));
    }

    #[test]
    fn laser_grace_acceleration_cap_and_analytic_progress_are_correct() {
        let config = LaserConfig::default();
        let mut laser = LaserState::default();
        laser.advance(3.0, 10_000.0, config);
        assert_eq!(laser.speed, 0.0);
        assert_eq!(laser.progress, 0.0);

        laser.advance(2.0, 10_000.0, config);
        assert_eq!(laser.speed, 60.0);
        assert!((laser.progress - 60.0).abs() < 1.0e-5);

        laser.advance(10.0, 10_000.0, config);
        assert_eq!(laser.speed, 130.0);
        let acceleration_time = 130.0 / 30.0;
        let expected =
            0.5 * 30.0 * acceleration_time * acceleration_time + 130.0 * (12.0 - acceleration_time);
        assert!((laser.progress - expected).abs() < 1.0e-3);
    }

    #[test]
    fn laser_uses_current_progress_and_can_catch_a_reversing_car() {
        let config = EvaluationConfig::default();
        let laser = LaserState {
            elapsed: 10.0,
            origin_progress: 16.0,
            progress: 70.0,
            speed: 130.0,
        };
        let mut ahead = EvaluationState::new(16.0);
        ahead.update(0.1, 96.0, 96.0, 116.0, &laser, &config);
        assert!(!ahead.is_finished());

        let mut reversing = EvaluationState::new(16.0);
        reversing.update(0.1, 76.0, 106.0, 116.0, &laser, &config);
        assert_eq!(
            reversing.finish_reason,
            Some(FinishReason::EliminatedByLaser)
        );
    }

    #[test]
    fn reaching_accumulated_lap_length_completes_episode() {
        let mut state = EvaluationState::new(16.0);
        state.update(
            0.1,
            50.0,
            100.0,
            100.0,
            &LaserState {
                elapsed: 10.0,
                origin_progress: 16.0,
                progress: 90.0,
                speed: 130.0,
            },
            &EvaluationConfig::default(),
        );
        assert_eq!(state.finish_reason, Some(FinishReason::Completed));
    }

    #[test]
    fn asymptotic_useful_speed_normalization_has_half_saturation_and_no_finite_ceiling() {
        let k = 120.0;
        assert_eq!(normalize_useful_progress_speed(0.0, k), 0.0);
        assert!((normalize_useful_progress_speed(k, k) - 0.5).abs() < 1.0e-6);
        assert!((normalize_useful_progress_speed(2.0 * k, k) - 2.0 / 3.0).abs() < 1.0e-6);

        let values = [60.0, 120.0, 200.0, 300.0, 600.0, f32::MAX]
            .map(|speed| normalize_useful_progress_speed(speed, k));
        assert!(values.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(values.iter().all(|value| *value < 1.0));
        assert_eq!(normalize_useful_progress_speed(f32::INFINITY, k), 1.0);
        assert_eq!(normalize_useful_progress_speed(f32::NEG_INFINITY, k), 0.0);
        assert_eq!(normalize_useful_progress_speed(f32::NAN, k), 0.0);
        for invalid_k in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            assert_eq!(normalize_useful_progress_speed(120.0, invalid_k), 0.0);
        }
    }

    #[test]
    fn finite_useful_speeds_above_120_remain_selectively_distinct() {
        let config = EvaluationConfig::default();
        let at_200 = episode_score(
            &finished_state(FinishReason::Completed, 0.5),
            100.0,
            100.0,
            &config,
        );
        let at_300 = episode_score(
            &finished_state(FinishReason::Completed, 1.0 / 3.0),
            100.0,
            100.0,
            &config,
        );

        assert!((at_200.useful_progress_speed - 200.0).abs() < 1.0e-3);
        assert!((at_300.useful_progress_speed - 300.0).abs() < 1.0e-3);
        assert!(at_300.score > at_200.score);
        assert!(at_300.score < 1.85);
    }

    #[test]
    fn laser_progress_is_relative_to_spawn_and_grace_still_protects_the_car() {
        let config = EvaluationConfig::default();
        let mut state = EvaluationState::new(16.0);
        let during_grace = LaserState {
            elapsed: config.laser.grace_period,
            origin_progress: 16.0,
            progress: 0.0,
            speed: 0.0,
        };
        state.update(0.0, 16.0, 16.0, 100.0, &during_grace, &config);
        assert!(!state.is_finished());

        let after_grace = LaserState {
            elapsed: config.laser.grace_period + 0.1,
            ..during_grace
        };
        state.update(0.0, 16.0, 16.0, 100.0, &after_grace, &config);
        assert_eq!(state.finish_reason, Some(FinishReason::EliminatedByLaser));
        assert_eq!(after_grace.track_progress(), 16.0);
    }

    #[test]
    fn score_rewards_progress_and_useful_speed_and_penalizes_collision() {
        let config = EvaluationConfig::default();
        let clean = finished_state(FinishReason::EliminatedByLaser, 10.0);
        let collision = finished_state(FinishReason::Collision, 10.0);
        assert!(
            episode_score(&clean, 60.0, 100.0, &config).score
                > episode_score(&collision, 60.0, 100.0, &config).score
        );
        assert!(
            episode_score(&clean, 70.0, 100.0, &config).score
                > episode_score(&clean, 60.0, 100.0, &config).score
        );
        assert!(
            episode_score(&clean, 60.0, 100.0, &config).score
                > episode_score(
                    &finished_state(FinishReason::EliminatedByLaser, 20.0),
                    60.0,
                    100.0,
                    &config,
                )
                .score
        );
        assert!(
            episode_score(&collision, 90.0, 100.0, &config).score
                > episode_score(&collision, 10.0, 100.0, &config).score
        );
        assert!(
            episode_score(
                &finished_state(FinishReason::Completed, 20.0),
                100.0,
                100.0,
                &config,
            )
            .score
                > episode_score(&clean, 99.0, 100.0, &config).score
        );
    }

    #[test]
    fn normalized_progress_uses_remaining_lap_from_episode_start() {
        let config = EvaluationConfig::default();
        let start = episode_score(
            &finished_state_from(FinishReason::EliminatedByLaser, 10.0, 20.0),
            20.0,
            100.0,
            &config,
        );
        let halfway = episode_score(
            &finished_state_from(FinishReason::EliminatedByLaser, 10.0, 20.0),
            60.0,
            100.0,
            &config,
        );
        let finish = episode_score(
            &finished_state_from(FinishReason::Completed, 10.0, 20.0),
            100.0,
            100.0,
            &config,
        );

        assert_eq!(start.normalized_progress, 0.0);
        assert!((halfway.normalized_progress - 0.5).abs() < 1.0e-6);
        assert_eq!(finish.normalized_progress, 1.0);
    }

    #[test]
    fn valid_config_guarantees_completed_score_exceeds_any_non_completed_score() {
        let config = EvaluationConfig::default();
        assert!(config.validate().is_ok());

        let minimum_completed_bound = config.progress_weight + config.completion_bonus;
        let maximum_non_completed_bound = config.progress_weight + config.speed_weight;
        assert!(minimum_completed_bound > maximum_non_completed_bound);

        let completed = episode_score(
            &finished_state_from(FinishReason::Completed, f32::MAX, 20.0),
            100.0,
            100.0,
            &config,
        );
        let nearly_complete_at_max_speed = episode_score(
            &finished_state_from(FinishReason::Timeout, 0.001, 20.0),
            99.999_99,
            100.0,
            &config,
        );
        assert!(completed.score > nearly_complete_at_max_speed.score);
    }

    #[test]
    fn normalized_progress_is_comparable_between_track_lengths() {
        let config = EvaluationConfig::default();
        let short = episode_score(
            &finished_state_from(FinishReason::EliminatedByLaser, 10.0, 20.0),
            60.0,
            100.0,
            &config,
        );
        let long = episode_score(
            &finished_state_from(FinishReason::EliminatedByLaser, 20.0, 40.0),
            120.0,
            200.0,
            &config,
        );
        assert_eq!(short.score, long.score);
    }

    #[test]
    fn invalid_evaluation_parameters_are_rejected() {
        let mut config = EvaluationConfig::default();
        config.laser.acceleration = 0.0;
        assert!(config.validate().is_err());
        config = EvaluationConfig::default();
        config.training_track_selection = TrainingTrackSelection::RandomSubset(0);
        assert!(config.validate().is_err());
        config = EvaluationConfig::default();
        config.completion_bonus = config.speed_weight;
        assert!(config.validate().is_err());
        config = EvaluationConfig::default();
        config.progress_speed_half_saturation = f32::NAN;
        assert!(config.validate().is_err());
    }

    #[test]
    fn generation_uses_one_reproducible_track_subset_for_every_individual() {
        let library = TrackLibrary::load_default().unwrap();
        let a = TrainingState::with_config(3, &library, EvaluationConfig::default()).unwrap();
        let b = TrainingState::with_config(3, &library, EvaluationConfig::default()).unwrap();
        assert_eq!(a.selected_training_tracks(), b.selected_training_tracks());
        assert_eq!(a.selected_training_tracks().len(), 3);
    }

    #[test]
    fn training_fitness_is_mean_and_validation_cannot_change_it() {
        let library = TrackLibrary::load_default().unwrap();
        let config = EvaluationConfig {
            training_track_selection: TrainingTrackSelection::RandomSubset(2),
            ..EvaluationConfig::default()
        };
        let mut state = TrainingState::with_config(2, &library, config).unwrap();
        assert!(matches!(
            state
                .record_training_results(&[result(1.0), result(3.0)])
                .unwrap(),
            TrackAdvance::Training(_)
        ));
        assert!(matches!(
            state
                .record_training_results(&[result(3.0), result(5.0)])
                .unwrap(),
            TrackAdvance::Validation(_)
        ));
        assert_eq!(state.population().individuals()[0].fitness(), Some(2.0));
        assert_eq!(state.population().individuals()[1].fitness(), Some(4.0));
        let fitness_before = state
            .population()
            .individuals()
            .iter()
            .map(|item| item.fitness())
            .collect::<Vec<_>>();
        state.record_validation_result(result(-100.0)).unwrap();
        let fitness_after = state
            .population()
            .individuals()
            .iter()
            .map(|item| item.fitness())
            .collect::<Vec<_>>();
        assert_eq!(fitness_before, fitness_after);
    }

    #[test]
    fn track_and_champion_metrics_are_aggregated_from_the_winning_individual() {
        let library = TrackLibrary::load_default().unwrap();
        let config = EvaluationConfig {
            training_track_selection: TrainingTrackSelection::RandomSubset(2),
            ..EvaluationConfig::default()
        };
        let mut state = TrainingState::with_config(2, &library, config).unwrap();
        let first_track = state.current_track_id().unwrap().to_string();
        state
            .record_training_results(&[
                measured_result(1.0, 10.0, FinishReason::Completed),
                measured_result(3.0, 30.0, FinishReason::Collision),
            ])
            .unwrap();
        let second_track = state.current_track_id().unwrap().to_string();
        state
            .record_training_results(&[
                measured_result(5.0, 50.0, FinishReason::Timeout),
                measured_result(4.0, 40.0, FinishReason::Completed),
            ])
            .unwrap();
        state
            .record_validation_result(measured_result(0.5, 25.0, FinishReason::EliminatedByLaser))
            .unwrap();

        let completed = state.completed_champion().unwrap();
        assert_eq!(completed.track_stats.len(), 2);
        assert_eq!(completed.track_stats[0].track_id, first_track);
        assert_eq!(completed.track_stats[0].best_score, 3.0);
        assert_eq!(completed.track_stats[0].average_score, 2.0);
        assert_eq!(completed.track_stats[0].average_useful_progress_speed, 20.0);
        assert_eq!(completed.track_stats[0].completion_rate, 0.5);
        assert_eq!(completed.track_stats[0].finish_counts.completed, 1);
        assert_eq!(completed.track_stats[0].finish_counts.collision, 1);
        assert_eq!(completed.track_stats[1].track_id, second_track);
        assert_eq!(completed.track_stats[1].average_score, 4.5);
        assert_eq!(completed.track_stats[1].average_useful_progress_speed, 45.0);

        assert_eq!(state.champion_population_index(), Some(1));
        assert_eq!(completed.training.training_fitness, 3.5);
        assert_eq!(completed.training.population_average_fitness, 3.25);
        assert_eq!(completed.training.average_useful_progress_speed, 35.0);
        assert_eq!(completed.training.completion_rate, 0.5);
        assert_eq!(completed.training.finish_counts.completed, 1);
        assert_eq!(completed.training.finish_counts.collision, 1);
        assert_eq!(completed.training.finish_counts.laser_eliminated, 0);
        assert_eq!(completed.training.finish_counts.timeout, 0);
        assert_eq!(
            completed.training.training_tracks,
            vec![first_track, second_track]
        );
    }

    #[test]
    fn evolution_waits_for_all_training_tracks_and_validation() {
        let library = TrackLibrary::load_default().unwrap();
        let config = EvaluationConfig {
            training_track_selection: TrainingTrackSelection::RandomSubset(2),
            ..EvaluationConfig::default()
        };
        let mut state = TrainingState::with_config(2, &library, config).unwrap();
        state
            .record_training_results(&[result(1.0), result(2.0)])
            .unwrap();
        assert!(state.evolve_generation().is_err());
        state
            .record_training_results(&[result(2.0), result(3.0)])
            .unwrap();
        assert!(state.evolve_generation().is_err());
        state.record_validation_result(result(0.5)).unwrap();
        let stats = state.evolve_generation().unwrap();
        assert_eq!(stats.generation, 0);
        assert_eq!(state.generation(), 1);
    }

    #[test]
    fn all_mode_selects_entire_training_suite() {
        let library = TrackLibrary::load_default().unwrap();
        let config = EvaluationConfig {
            training_track_selection: TrainingTrackSelection::All,
            ..EvaluationConfig::default()
        };
        let state = TrainingState::with_config(2, &library, config).unwrap();
        assert_eq!(
            state.selected_training_tracks().len(),
            library.training_tracks().len()
        );
    }
}
