use super::{
    TrackLibrary,
    controller::{MLP_INPUT_SIZE, MLP_OUTPUT_SIZE},
};
use bevy::prelude::{Component, Resource};
use neuroevolution::{
    genetic::{Config, Population},
    neural::{Activation, Architecture},
};
use rand::{RngExt, SeedableRng, rngs::StdRng};

const EVALUATION_RNG_SALT: u64 = 0x4556_414c_5541_5445;
const DEFAULT_TRAINING_TRACKS_PER_GENERATION: Option<usize> = Some(3);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Completed,
    Collision,
    Stalled,
    Timeout,
}

impl FinishReason {
    pub fn label(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Collision => "collision",
            Self::Stalled => "stalled",
            Self::Timeout => "timeout",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FinishReasonCounts {
    pub completed: usize,
    pub collision: usize,
    pub stalled: usize,
    pub timeout: usize,
}

impl FinishReasonCounts {
    fn record(&mut self, reason: FinishReason) {
        match reason {
            FinishReason::Completed => self.completed += 1,
            FinishReason::Collision => self.collision += 1,
            FinishReason::Stalled => self.stalled += 1,
            FinishReason::Timeout => self.timeout += 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingTrackSelection {
    All,
    RandomSubset(usize),
}

/// Parameters for episode termination and the deliberately small score formula.
/// Progress is dominant, useful speed is a tie-breaker, and collision is a bounded penalty.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluationConfig {
    pub maximum_episode_duration: f32,
    pub stall_timeout: f32,
    pub significant_progress_epsilon: f32,
    pub progress_weight: f32,
    pub speed_weight: f32,
    pub collision_penalty: f32,
    pub completion_bonus: f32,
    pub progress_speed_normalization: f32,
    pub training_track_selection: TrainingTrackSelection,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            maximum_episode_duration: 60.0,
            stall_timeout: 2.5,
            significant_progress_epsilon: 6.0,
            progress_weight: 1.0,
            speed_weight: 0.20,
            collision_penalty: 0.08,
            completion_bonus: 0.25,
            progress_speed_normalization: 120.0,
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
            ("stall_timeout", self.stall_timeout),
            (
                "significant_progress_epsilon",
                self.significant_progress_epsilon,
            ),
            (
                "progress_speed_normalization",
                self.progress_speed_normalization,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!("{name} must be finite and greater than zero"));
            }
        }
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
    pub time_without_progress: f32,
    pub last_significant_progress: f32,
    pub initial_progress: f32,
}

impl EvaluationState {
    pub fn new(initial_progress: f32) -> Self {
        Self {
            finish_reason: None,
            elapsed: 0.0,
            time_without_progress: 0.0,
            last_significant_progress: initial_progress,
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
        best_track_distance: f32,
        total_track_length: f32,
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

        if best_track_distance - self.last_significant_progress
            >= config.significant_progress_epsilon
        {
            self.last_significant_progress = best_track_distance;
            self.time_without_progress = 0.0;
        } else {
            self.time_without_progress += delta_seconds.max(0.0);
        }

        if self.time_without_progress >= config.stall_timeout {
            self.finish(FinishReason::Stalled);
        } else if self.elapsed >= config.maximum_episode_duration {
            self.finish(FinishReason::Timeout);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpisodeResult {
    pub score: f32,
    pub normalized_progress: f32,
    pub normalized_progress_speed: f32,
    pub elapsed: f32,
    pub finish_reason: FinishReason,
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
        (useful_speed / config.progress_speed_normalization).clamp(0.0, 1.0);
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
pub struct ValidationStats {
    pub generation: usize,
    pub track_id: String,
    pub score: f32,
    pub normalized_progress: f32,
    pub elapsed: f32,
    pub finish_reason: FinishReason,
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
    evolution_rng: StdRng,
    evaluation_rng: StdRng,
    training_track_ids: Vec<String>,
    validation_track_ids: Vec<String>,
    selected_training_tracks: Vec<String>,
    training_score_sums: Vec<f32>,
    completed_training_tracks: usize,
    phase: TrainingPhase,
    history: Vec<GenerationStats>,
    validation_history: Vec<ValidationStats>,
    pending_stats: Option<GenerationStats>,
    champion_genome: Option<Vec<f32>>,
    champion_population_index: Option<usize>,
    finish_counts: FinishReasonCounts,
    last_finish_counts: FinishReasonCounts,
}

impl TrainingState {
    pub fn with_config(
        population_size: usize,
        library: &TrackLibrary,
        evaluation_config: EvaluationConfig,
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

        let architecture = Architecture::new(
            vec![MLP_INPUT_SIZE, 8, MLP_OUTPUT_SIZE],
            vec![Activation::Tanh, Activation::Tanh],
        )?;
        let genetic_config = Config {
            population_size,
            genome_length: architecture.parameter_count(),
            ..Config::default()
        };
        let mut evolution_rng = StdRng::seed_from_u64(genetic_config.seed);
        let population = Population::new(&genetic_config, &mut evolution_rng)?;
        let evaluation_rng = StdRng::seed_from_u64(genetic_config.seed ^ EVALUATION_RNG_SALT);
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
            completed_training_tracks: 0,
            phase: TrainingPhase::Evolving,
            history: Vec::new(),
            validation_history: Vec::new(),
            pending_stats: None,
            champion_genome: None,
            champion_population_index: None,
            finish_counts: FinishReasonCounts::default(),
            last_finish_counts: FinishReasonCounts::default(),
        };
        state.start_generation();
        Ok(state)
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
        self.completed_training_tracks = 0;
        self.finish_counts = FinishReasonCounts::default();
        self.pending_stats = None;
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
        let TrainingPhase::TrainingTrack { index, total, .. } = self.phase else {
            return Err("training results can only be recorded during a training track".into());
        };
        if results.len() != self.population.len() {
            return Err(format!(
                "expected {} episode results, got {}",
                self.population.len(),
                results.len()
            ));
        }
        for (individual_index, (sum, result)) in
            self.training_score_sums.iter_mut().zip(results).enumerate()
        {
            if !result.score.is_finite() {
                return Err(format!(
                    "episode score for individual {individual_index} is not finite"
                ));
            }
            *sum += result.score;
            self.finish_counts.record(result.finish_reason);
        }
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
            elapsed: result.elapsed,
            finish_reason: result.finish_reason,
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
        self.last_finish_counts = self.finish_counts;
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

    pub fn last_finish_counts(&self) -> FinishReasonCounts {
        self.last_finish_counts
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
            time_without_progress: 0.0,
            last_significant_progress: initial_progress,
            initial_progress,
        }
    }

    fn result(score: f32) -> EpisodeResult {
        EpisodeResult {
            score,
            normalized_progress: score.clamp(0.0, 1.0),
            normalized_progress_speed: 0.0,
            elapsed: 1.0,
            finish_reason: FinishReason::Stalled,
        }
    }

    #[test]
    fn collision_finishes_episode_immediately() {
        let mut state = EvaluationState::new(0.0);
        state.finish(FinishReason::Collision);
        state.update(1.0, 10.0, 100.0, &EvaluationConfig::default());
        assert_eq!(state.finish_reason, Some(FinishReason::Collision));
        assert_eq!(state.elapsed, 0.0);
    }

    #[test]
    fn stall_and_timeout_terminate_episodes() {
        let mut config = EvaluationConfig::default();
        config.stall_timeout = 2.0;
        config.maximum_episode_duration = 10.0;
        let mut stalled = EvaluationState::new(0.0);
        stalled.update(2.0, 0.0, 100.0, &config);
        assert_eq!(stalled.finish_reason, Some(FinishReason::Stalled));

        config.stall_timeout = 20.0;
        config.maximum_episode_duration = 1.0;
        let mut timed_out = EvaluationState::new(0.0);
        timed_out.update(1.0, 0.0, 100.0, &config);
        assert_eq!(timed_out.finish_reason, Some(FinishReason::Timeout));
    }

    #[test]
    fn significant_progress_resets_stall_even_at_low_instantaneous_speed() {
        let mut config = EvaluationConfig::default();
        config.stall_timeout = 2.0;
        config.significant_progress_epsilon = 1.0;
        let mut state = EvaluationState::new(0.0);
        state.update(1.5, 0.5, 100.0, &config);
        assert_eq!(state.time_without_progress, 1.5);
        state.update(0.6, 1.1, 100.0, &config);
        assert_eq!(state.time_without_progress, 0.0);
        assert!(!state.is_finished());
    }

    #[test]
    fn reaching_accumulated_lap_length_completes_episode() {
        let mut state = EvaluationState::new(16.0);
        state.update(0.1, 100.0, 100.0, &EvaluationConfig::default());
        assert_eq!(state.finish_reason, Some(FinishReason::Completed));
    }

    #[test]
    fn score_rewards_progress_and_useful_speed_and_penalizes_collision() {
        let config = EvaluationConfig::default();
        let clean = finished_state(FinishReason::Stalled, 10.0);
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
                    &finished_state(FinishReason::Stalled, 20.0),
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
            &finished_state_from(FinishReason::Stalled, 10.0, 20.0),
            20.0,
            100.0,
            &config,
        );
        let halfway = episode_score(
            &finished_state_from(FinishReason::Stalled, 10.0, 20.0),
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
            &finished_state_from(FinishReason::Stalled, 10.0, 20.0),
            60.0,
            100.0,
            &config,
        );
        let long = episode_score(
            &finished_state_from(FinishReason::Stalled, 20.0, 40.0),
            120.0,
            200.0,
            &config,
        );
        assert_eq!(short.score, long.score);
    }

    #[test]
    fn invalid_evaluation_parameters_are_rejected() {
        let mut config = EvaluationConfig::default();
        config.stall_timeout = 0.0;
        assert!(config.validate().is_err());
        config = EvaluationConfig::default();
        config.training_track_selection = TrainingTrackSelection::RandomSubset(0);
        assert!(config.validate().is_err());
        config = EvaluationConfig::default();
        config.completion_bonus = config.speed_weight;
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
