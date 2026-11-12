//! Max-Min Ant System on a layered decision graph (sector radius choices + width).
//! The search has no access to training/validation libraries or PPO optimizers.
pub mod cli;
pub mod inspection;
#[cfg(test)]
mod inspection_tests;
pub mod live;
use inspection::{GeometryDiagnostic, RetentionPolicy};

use crate::simulation::procedural::{self, GeneratedTrack, GenerationResult, GeneratorConfig};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, path::Path, time::Instant};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    #[default]
    AntColony,
    RandomDifficulty,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FitnessObjective {
    #[default]
    LegacyAdversarial,
    UsefulProgress,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AcoConfig {
    #[serde(default = "GeneratorConfig::legacy")]
    pub generator: GeneratorConfig,
    pub ants: usize,
    pub iterations: usize,
    pub max_candidates: usize,
    pub max_selected: usize,
    pub minimum_diversity: f64,
    pub evaporation: f64,
    pub alpha: f64,
    pub beta: f64,
    pub exploitation_probability: f64,
    pub pheromone_min: f64,
    pub pheromone_max: f64,
    pub initial_pheromone: f64,
    pub stagnation_iterations: usize,
    pub max_evaluation_steps_per_candidate: usize,
    pub max_evaluation_steps: u64,
    pub strategy: SearchStrategy,
    #[serde(default)]
    pub fitness_objective: FitnessObjective,
    pub target_progress: f64,
    /// Identity of the frozen training policy; never a validation checkpoint score.
    pub policy_label: Option<String>,
    pub retention: RetentionPolicy,
}
impl Default for AcoConfig {
    fn default() -> Self {
        Self {
            generator: Default::default(),
            ants: 8,
            iterations: 3,
            max_candidates: 24,
            max_selected: 4,
            minimum_diversity: 0.15,
            evaporation: 0.2,
            alpha: 1.,
            beta: 1.,
            exploitation_probability: 0.1,
            pheromone_min: 0.05,
            pheromone_max: 5.,
            initial_pheromone: 1.,
            stagnation_iterations: 8,
            max_evaluation_steps_per_candidate: 6000,
            max_evaluation_steps: 144_000,
            strategy: SearchStrategy::AntColony,
            fitness_objective: FitnessObjective::UsefulProgress,
            target_progress: 0.7,
            policy_label: None,
            retention: RetentionPolicy::default(),
        }
    }
}
impl AcoConfig {
    pub fn validate(&self) -> GenerationResult<()> {
        self.generator.validate()?;
        if self.retention.detailed_candidates > 100_000
            || self.retention.pheromone_iterations > 10_000
        {
            return Err("invalid retention limits".into());
        }
        if self.ants == 0
            || self.ants > 256
            || self.iterations == 0
            || self.iterations > 10000
            || self.max_candidates == 0
            || self.max_candidates > 100_000
            || self.max_selected == 0
            || self.max_selected > 128
            || !self.minimum_diversity.is_finite()
            || !(0.0..=1.).contains(&self.minimum_diversity)
            || !self.evaporation.is_finite()
            || !(0.0..1.0).contains(&self.evaporation)
            || !self.alpha.is_finite()
            || !(0.0..=8.).contains(&self.alpha)
            || !self.beta.is_finite()
            || !(0.0..=8.).contains(&self.beta)
            || !self.exploitation_probability.is_finite()
            || !(0.0..=1.).contains(&self.exploitation_probability)
            || !self.pheromone_min.is_finite()
            || !self.pheromone_max.is_finite()
            || !self.initial_pheromone.is_finite()
            || self.pheromone_min <= 0.
            || self.pheromone_max < self.pheromone_min
            || self.initial_pheromone < self.pheromone_min
            || self.initial_pheromone > self.pheromone_max
            || self.stagnation_iterations == 0
            || self.max_evaluation_steps_per_candidate == 0
            || self.max_evaluation_steps == 0
            || !self.target_progress.is_finite()
            || !(0.01..1.).contains(&self.target_progress)
        {
            return Err("invalid ACO population/budget/probability/pheromone bounds".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DecisionSnapshot {
    pub position: usize,
    pub choice: usize,
    pub probabilities: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub construction: Option<procedural::sequential::ConstructionView>,
}
/// Aggregate of frozen policy episodes; the reference pilot is independent of PPO.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PilotEvaluation {
    pub completion_rate: f64,
    pub mean_progress: f64,
    pub collision_rate: f64,
    pub reference_completed: bool,
    #[serde(default)]
    pub reference_collision: bool,
    /// Total policy AND reference physical decisions; counts against one shared budget.
    pub simulation_steps: u64,
    #[serde(default)]
    pub reference_steps: u64,
    #[serde(default)]
    pub policy_steps: u64,
    #[serde(default)]
    pub inference_seconds: f64,
    #[serde(default)]
    pub simulation_seconds: f64,
    #[serde(default)]
    pub reference_progress: f64,
    #[serde(default)]
    pub reference_episodes: usize,
    #[serde(default)]
    pub policy_episodes: usize,
    #[serde(default)]
    pub budget_insufficient: bool,
}
impl PilotEvaluation {
    fn validate(&self) -> GenerationResult<()> {
        if [
            self.completion_rate,
            self.mean_progress,
            self.collision_rate,
        ]
        .iter()
        .any(|x| !x.is_finite() || !(0.0..=1.).contains(x))
            || !self.simulation_seconds.is_finite()
            || self.simulation_seconds < 0.
            || !self.reference_progress.is_finite()
            || !(0.0..=1.).contains(&self.reference_progress)
            || !self.inference_seconds.is_finite()
            || self.inference_seconds < 0.
            || self.reference_steps.saturating_add(self.policy_steps) > self.simulation_steps
        {
            Err("invalid frozen-policy evaluation metrics".into())
        } else {
            Ok(())
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitnessComponents {
    pub completion_rate: Option<f64>,
    pub mean_progress: Option<f64>,
    pub collision_rate: Option<f64>,
    pub geometric_difficulty: f64,
    pub diversity: f64,
    pub simulation_steps: u64,
    pub reference_completed: Option<bool>,
    pub score: f64,
    /// Triangular learning challenge, maximal near the configured target progress.
    #[serde(default)]
    pub useful_difficulty: f64,
    #[serde(default)]
    pub episodes: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub organic_quality: Option<f64>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectionReason {
    InvalidGeometry,
    ClosureInfeasible,
    CorridorClearance,
    SelfIntersection,
    CurvatureLimit,
    Duplicate,
    InsufficientDiversity,
    ReferenceCollision,
    ReferenceTimeout,
    InsufficientBudget,
    ExecutionFailure,
    NumericalError,
    SelectionCapacity,
}
impl RejectionReason {
    fn geometry(message: &str) -> Self {
        if message.contains("closure") {
            Self::ClosureInfeasible
        } else if message.contains("corridor") || message.contains("border gap") {
            Self::CorridorClearance
        } else if message.contains("intersect") {
            Self::SelfIntersection
        } else if message.contains("curvature") {
            Self::CurvatureLimit
        } else if message.contains("finite") {
            Self::NumericalError
        } else {
            Self::InvalidGeometry
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AntSnapshot {
    pub ant: usize,
    pub decisions: Vec<DecisionSnapshot>,
    pub candidate: Option<GeneratedTrack>,
    pub rejection: Option<String>,
    pub fitness: Option<FitnessComponents>,
    #[serde(default)]
    pub rejection_reason: Option<RejectionReason>,
    #[serde(default)]
    pub evaluation: Option<PilotEvaluation>,
    #[serde(default)]
    pub geometry_valid: Option<bool>,
    #[serde(default)]
    pub diagnostic: Option<GeometryDiagnostic>,
    #[serde(default)]
    pub retention_evicted: bool,
    #[serde(default)]
    pub candidate_id: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterationSnapshot {
    pub iteration: usize,
    pub ants: Vec<AntSnapshot>,
    pub pheromones_before: Vec<Vec<f64>>,
    pub pheromones_evaporated: Vec<Vec<f64>>,
    pub pheromones_after: Vec<Vec<f64>>,
    pub best_iteration: Option<String>,
    pub best_global: Option<String>,
    pub mean_fitness: f64,
    pub diversity: f64,
    pub stagnation_reset: bool,
    pub elapsed_seconds: f64,
    #[serde(default)]
    pub generation_seconds: f64,
    #[serde(default)]
    pub evaluation_seconds: f64,
    #[serde(default)]
    pub pheromone_update_seconds: f64,
    #[serde(default)]
    pub candidates_built: usize,
    #[serde(default)]
    pub candidates_geometrically_valid: usize,
    #[serde(default)]
    pub candidates_evaluated: usize,
    #[serde(default)]
    pub reference_decisions: u64,
    #[serde(default)]
    pub policy_decisions: u64,
    #[serde(default)]
    pub inference_seconds: f64,
    #[serde(default)]
    pub reference_completed: usize,
    #[serde(default)]
    pub simulation_seconds: f64,
    #[serde(default)]
    pub reference_collisions: usize,
    #[serde(default)]
    pub reference_timeouts: usize,
    #[serde(default)]
    pub evaluation_failures: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RankedTrack {
    pub track: GeneratedTrack,
    pub fitness: FitnessComponents,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcoState {
    pub seed: u64,
    pub pheromones: Vec<Vec<f64>>,
    pub iteration: usize,
    pub candidates: usize,
    pub evaluation_steps: u64,
    pub iterations_without_improvement: usize,
    pub selected: Vec<RankedTrack>,
    pub seen_hashes: BTreeSet<String>,
    pub decision_archive: Vec<Vec<usize>>,
    #[serde(default)]
    pub geometry_archive: Vec<Vec<f32>>,
    #[serde(with = "saved_rng")]
    construction_rng: ChaCha12Rng,
    #[serde(with = "saved_rng")]
    evaluation_rng: ChaCha12Rng,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchArchive {
    pub version: u32,
    pub config: AcoConfig,
    pub state: AcoState,
    pub iterations: Vec<IterationSnapshot>,
    #[serde(default)]
    pub inspection_version: u32,
    /// Inspection-only records from a UI save made before an iteration completed.
    #[serde(default)]
    pub in_progress: Vec<AntSnapshot>,
}
/// Caller owns the immutable network and independent environment RNGs. There is
/// deliberately no optimizer or mutable policy argument at this boundary.
pub trait CandidateEvaluator {
    fn evaluate(
        &mut self,
        track: &GeneratedTrack,
        seed: u64,
        decision_budget: usize,
    ) -> GenerationResult<PilotEvaluation>;
}
impl<F> CandidateEvaluator for F
where
    F: FnMut(&GeneratedTrack, u64, usize) -> GenerationResult<PilotEvaluation>,
{
    fn evaluate(
        &mut self,
        track: &GeneratedTrack,
        seed: u64,
        budget: usize,
    ) -> GenerationResult<PilotEvaluation> {
        self(track, seed, budget)
    }
}

impl SearchArchive {
    pub fn new(config: AcoConfig, seed: u64) -> GenerationResult<Self> {
        config.validate()?;
        let state = AcoState {
            seed,
            pheromones: config
                .generator
                .choice_counts()
                .iter()
                .map(|n| vec![config.initial_pheromone; *n])
                .collect(),
            iteration: 0,
            candidates: 0,
            evaluation_steps: 0,
            iterations_without_improvement: 0,
            selected: Vec::new(),
            seen_hashes: BTreeSet::new(),
            decision_archive: Vec::new(),
            geometry_archive: Vec::new(),
            construction_rng: ChaCha12Rng::seed_from_u64(seed ^ 0x41434f5f434f4e53),
            evaluation_rng: ChaCha12Rng::seed_from_u64(seed ^ 0x41434f5f4556414c),
        };
        Ok(Self {
            version: config.generator.version(),
            config,
            state,
            iterations: Vec::new(),
            inspection_version: 1,
            in_progress: vec![],
        })
    }
    pub fn is_finished(&self) -> bool {
        self.state.iteration >= self.config.iterations
            || self.state.candidates >= self.config.max_candidates
    }
    pub fn run_iteration(
        &mut self,
        evaluator: Option<&mut dyn CandidateEvaluator>,
    ) -> GenerationResult<&IterationSnapshot> {
        self.run_iteration_observed(evaluator, &mut live::Unobserved)
    }
    /// The headless and interactive paths execute exactly the same search code.
    /// Observers can block at safe boundaries, or cancel by returning an error.
    pub fn run_iteration_observed(
        &mut self,
        mut evaluator: Option<&mut dyn CandidateEvaluator>,
        observer: &mut dyn live::SearchObserver,
    ) -> GenerationResult<&IterationSnapshot> {
        self.validate()?;
        if !self.in_progress.is_empty() {
            return Err("partial inspection snapshot: replay/export available; restart search to construct a new iteration".into());
        }
        if self.is_finished() {
            return Err("ACO configured iteration/candidate budget exhausted".into());
        }
        if evaluator.is_some() && self.config.policy_label.is_none() {
            return Err("frozen-policy evaluation requires a policy_label for provenance".into());
        }
        let started = Instant::now();
        let before = self.state.pheromones.clone();
        let previous_best = self
            .state
            .selected
            .first()
            .map_or(f64::NEG_INFINITY, |x| x.fitness.score);
        let mut ants = Vec::new();
        let mut generation_seconds = 0.;
        let mut evaluation_seconds = 0.;
        let mut candidates_geometrically_valid = 0;
        let mut candidates_evaluated = 0;
        let mut reference_decisions = 0;
        let mut policy_decisions = 0;
        let mut inference_seconds = 0.;
        let mut simulation_seconds = 0.;
        let mut reference_completed = 0;
        let mut reference_collisions = 0;
        let mut reference_timeouts = 0;
        let mut evaluation_failures = 0;
        let mut iteration_best: Option<(String, f64)> = None;
        for ant in 0..self.config.ants {
            if self.state.candidates >= self.config.max_candidates {
                break;
            }
            let remaining_steps = self
                .config
                .max_evaluation_steps
                .saturating_sub(self.state.evaluation_steps);
            if evaluator.is_some() && remaining_steps == 0 {
                break;
            }
            observer.checkpoint()?;
            observer.event(live::SearchEvent::AntStarted {
                iteration: self.state.iteration,
                ant,
                pheromones: self.state.pheromones.clone(),
            })?;
            self.state.candidates += 1;
            let generation_started = Instant::now();
            let mut terminal = None;
            let decisions = self.construct_observed(observer, &mut terminal)?;
            let choices: Vec<_> = decisions.iter().map(|d| d.choice).collect();
            let candidate = procedural::from_decisions(
                &self.config.generator,
                self.state.seed,
                self.state.candidates - 1,
                &choices,
            );
            let raw_definition =
                procedural::definition_from_decisions(&self.config.generator, &choices).ok();
            if let Some(definition) = raw_definition.clone() {
                observer.event(live::SearchEvent::Geometry {
                    definition,
                    error: candidate.as_ref().err().cloned(),
                })?;
            }
            observer.checkpoint()?;
            generation_seconds += generation_started.elapsed().as_secs_f64();
            let mut snapshot = AntSnapshot {
                ant,
                decisions,
                candidate: None,
                rejection: None,
                fitness: None,
                rejection_reason: None,
                evaluation: None,
                geometry_valid: Some(candidate.is_ok()),
                diagnostic: None,
                retention_evicted: false,
                candidate_id: candidate.as_ref().ok().map(|c| c.definition.id.clone()),
            };
            match candidate {
                Err(reason) => {
                    snapshot.diagnostic = Some(GeometryDiagnostic::capture(
                        raw_definition,
                        &snapshot.decisions,
                        terminal,
                        &reason,
                    ));
                    snapshot.rejection_reason = Some(RejectionReason::geometry(&reason));
                    snapshot.rejection = Some(reason);
                }
                Ok(candidate) => {
                    candidates_geometrically_valid += 1;
                    let hash = candidate.provenance.geometry_hash.clone();
                    if !self.state.seen_hashes.insert(hash.clone()) {
                        snapshot.rejection = Some("duplicate geometry hash".into());
                        snapshot.rejection_reason = Some(RejectionReason::Duplicate);
                        snapshot.candidate = Some(candidate);
                        observer.event(live::SearchEvent::Candidate(snapshot.clone()))?;
                        ants.push(snapshot);
                        continue;
                    }
                    let diversity = if self.config.generator.is_sequential() {
                        self.state
                            .geometry_archive
                            .iter()
                            .map(|other| {
                                procedural::sequential::descriptor_distance(
                                    &candidate.geometry.descriptor,
                                    other,
                                ) as f64
                            })
                            .min_by(f64::total_cmp)
                            .unwrap_or(1.)
                    } else {
                        decision_diversity(&choices, &self.state.decision_archive)
                    };
                    if diversity < self.config.minimum_diversity {
                        snapshot.rejection = Some("near-duplicate decision signature".into());
                        snapshot.rejection_reason = Some(RejectionReason::InsufficientDiversity);
                        snapshot.candidate = Some(candidate);
                        observer.event(live::SearchEvent::Candidate(snapshot.clone()))?;
                        ants.push(snapshot);
                        continue;
                    }
                    let geometry = (candidate.geometry.max_curvature as f64
                        / self.config.generator.max_curvature as f64)
                        .clamp(0., 1.);
                    let evaluation = if let Some(ref mut evaluator) = evaluator {
                        let seed = self.state.evaluation_rng.random::<u64>();
                        let budget = self
                            .config
                            .max_evaluation_steps_per_candidate
                            .min(remaining_steps.min(usize::MAX as u64) as usize);
                        if budget < 2 {
                            snapshot.rejection =
                                Some("insufficient budget for reference and policy".into());
                            snapshot.rejection_reason = Some(RejectionReason::InsufficientBudget);
                            snapshot.candidate = Some(candidate);
                            observer.event(live::SearchEvent::Candidate(snapshot.clone()))?;
                            ants.push(snapshot);
                            continue;
                        }
                        let evaluation_started = Instant::now();
                        let result = evaluator.evaluate(&candidate, seed, budget);
                        evaluation_seconds += evaluation_started.elapsed().as_secs_f64();
                        match result {
                            Ok(result) => {
                                if let Err(error) = result.validate() {
                                    // Unknown consumed work is charged conservatively, never reused.
                                    self.state.evaluation_steps += budget as u64;
                                    snapshot.rejection = Some(error);
                                    snapshot.rejection_reason =
                                        Some(RejectionReason::NumericalError);
                                    snapshot.candidate = Some(candidate);
                                    observer
                                        .event(live::SearchEvent::Candidate(snapshot.clone()))?;
                                    ants.push(snapshot);
                                    continue;
                                }
                                if result.simulation_steps > budget as u64 {
                                    return Err(
                                        "candidate evaluator exceeded its explicit decision budget"
                                            .into(),
                                    );
                                }
                                self.state.evaluation_steps += result.simulation_steps;
                                candidates_evaluated += 1;
                                reference_decisions += result.reference_steps;
                                policy_decisions += result.policy_steps;
                                inference_seconds += result.inference_seconds;
                                simulation_seconds += result.simulation_seconds;
                                if result.reference_completed {
                                    reference_completed += 1;
                                } else if result.reference_collision {
                                    reference_collisions += 1;
                                } else {
                                    reference_timeouts += 1;
                                }
                                if !result.reference_completed {
                                    snapshot.rejection_reason =
                                        Some(if result.budget_insufficient {
                                            RejectionReason::InsufficientBudget
                                        } else if result.reference_collision {
                                            RejectionReason::ReferenceCollision
                                        } else {
                                            RejectionReason::ReferenceTimeout
                                        });
                                    snapshot.rejection = Some(format!(
                                        "reference did not verify drivability: {:?}",
                                        snapshot.rejection_reason
                                    ));
                                }
                                snapshot.evaluation = Some(result.clone());
                                Some(result)
                            }
                            Err(error) => {
                                evaluation_failures += 1;
                                self.state.evaluation_steps += budget as u64;
                                snapshot.rejection_reason =
                                    Some(if error.contains("finite") || error.contains("NaN") {
                                        RejectionReason::NumericalError
                                    } else {
                                        RejectionReason::ExecutionFailure
                                    });
                                snapshot.rejection = Some(format!("evaluation failed: {error}"));
                                None
                            }
                        }
                    } else {
                        None
                    };
                    let mut score = if let Some(e) = &evaluation {
                        match self.config.fitness_objective {
                            FitnessObjective::UsefulProgress => {
                                0.7 * useful_difficulty(
                                    e.mean_progress,
                                    self.config.target_progress,
                                ) + 0.1 * geometry
                                    + 0.2 * diversity
                            }
                            FitnessObjective::LegacyAdversarial => {
                                0.4 * (1. - e.completion_rate)
                                    + 0.2 * (1. - e.mean_progress)
                                    + 0.1 * e.collision_rate
                                    + 0.1 * geometry
                                    + 0.2 * diversity
                            }
                        }
                    } else {
                        0.4 * geometry + 0.6 * diversity
                    };
                    let organic_quality = (self.config.generator.is_sequential()
                        && self.config.generator.organic)
                        .then(|| {
                            let m = &candidate.geometry.organic;
                            (0.35 * m.complexity
                                + 0.25 * (1. - m.rectangularity)
                                + 0.20 * m.heading_entropy
                                + 0.20 * (1. - (m.longest_straight_ratio / 0.45).min(1.)))
                                as f64
                        });
                    if let Some(quality) = organic_quality {
                        score = 0.8 * score + 0.2 * quality;
                    }
                    let fitness = FitnessComponents {
                        organic_quality,
                        completion_rate: evaluation
                            .as_ref()
                            .filter(|e| e.policy_episodes > 0)
                            .map(|e| e.completion_rate),
                        mean_progress: evaluation
                            .as_ref()
                            .filter(|e| e.policy_episodes > 0)
                            .map(|e| e.mean_progress),
                        collision_rate: evaluation
                            .as_ref()
                            .filter(|e| e.policy_episodes > 0)
                            .map(|e| e.collision_rate),
                        geometric_difficulty: geometry,
                        diversity,
                        simulation_steps: evaluation.as_ref().map_or(0, |e| e.simulation_steps),
                        reference_completed: evaluation.as_ref().map(|e| e.reference_completed),
                        score,
                        useful_difficulty: evaluation.as_ref().map_or(0., |e| {
                            useful_difficulty(e.mean_progress, self.config.target_progress)
                        }),
                        episodes: evaluation.as_ref().map_or(0, |e| e.policy_episodes),
                    };
                    snapshot.fitness = Some(fitness.clone());
                    if snapshot.rejection.is_none() {
                        if iteration_best.as_ref().is_none_or(|(_, s)| score > *s) {
                            iteration_best = Some((candidate.definition.id.clone(), score));
                        }
                        self.state.selected.push(RankedTrack {
                            track: candidate.clone(),
                            fitness,
                        });
                        self.state
                            .selected
                            .sort_by(|a, b| b.fitness.score.total_cmp(&a.fitness.score));
                        self.state.selected.truncate(self.config.max_selected);
                        if self.config.generator.is_sequential() {
                            self.state
                                .geometry_archive
                                .push(candidate.geometry.descriptor.clone());
                            if self.state.geometry_archive.len() > 256 {
                                self.state.geometry_archive.remove(0);
                            }
                        }
                        self.state.decision_archive.push(choices);
                        if self.state.decision_archive.len() > 256 {
                            self.state.decision_archive.remove(0);
                        }
                    }
                    snapshot.candidate = Some(candidate);
                }
            }
            observer.event(live::SearchEvent::Candidate(snapshot.clone()))?;
            ants.push(snapshot);
        }
        observer.checkpoint()?;
        let pheromone_update_started = Instant::now();
        // Only accepted solutions deposit. Evaporation is observable separately.
        if self.config.strategy == SearchStrategy::AntColony {
            for row in &mut self.state.pheromones {
                for p in row {
                    *p = (*p * (1. - self.config.evaporation)).max(self.config.pheromone_min);
                }
            }
        }
        let evaporated = self.state.pheromones.clone();
        if self.config.strategy == SearchStrategy::AntColony {
            let elite = if self.config.generator.is_sequential() {
                ants.iter()
                    .filter(|a| a.rejection.is_none())
                    .filter_map(|a| a.candidate.as_ref().zip(a.fitness.as_ref()))
                    .max_by(|a, b| a.1.score.total_cmp(&b.1.score))
                    .or_else(|| self.state.selected.first().map(|b| (&b.track, &b.fitness)))
            } else {
                self.state.selected.first().map(|b| (&b.track, &b.fitness))
            };
            if let Some((track, fitness)) = elite {
                for (position, choice) in track.provenance.decisions.iter().enumerate() {
                    let row = if self.config.generator.is_sequential()
                        && position + 1 == track.provenance.decisions.len()
                    {
                        self.config.generator.sectors
                    } else {
                        position
                    };
                    let p = &mut self.state.pheromones[row][*choice];
                    *p = (*p + self.config.evaporation * (0.25 + fitness.score))
                        .min(self.config.pheromone_max);
                }
            }
        }
        let improved = self
            .state
            .selected
            .first()
            .is_some_and(|x| x.fitness.score > previous_best + 1.0e-12);
        self.state.iterations_without_improvement = if improved {
            0
        } else {
            self.state.iterations_without_improvement + 1
        };
        let reset = self.state.iterations_without_improvement >= self.config.stagnation_iterations;
        if reset {
            for row in &mut self.state.pheromones {
                row.fill(self.config.initial_pheromone);
            }
            self.state.iterations_without_improvement = 0;
        }
        let accepted: Vec<_> = ants
            .iter()
            .filter(|a| a.rejection.is_none())
            .filter_map(|a| a.fitness.as_ref())
            .collect();
        let candidates_built = ants.len();
        let count = accepted.len().max(1) as f64;
        let snapshot = IterationSnapshot {
            iteration: self.state.iteration,
            mean_fitness: accepted.iter().map(|f| f.score).sum::<f64>() / count,
            diversity: accepted.iter().map(|f| f.diversity).sum::<f64>() / count,
            ants,
            pheromones_before: before,
            pheromones_evaporated: evaporated,
            pheromones_after: self.state.pheromones.clone(),
            best_iteration: iteration_best.map(|x| x.0),
            best_global: self
                .state
                .selected
                .first()
                .map(|x| x.track.definition.id.clone()),
            stagnation_reset: reset,
            elapsed_seconds: started.elapsed().as_secs_f64(),
            generation_seconds,
            evaluation_seconds,
            pheromone_update_seconds: pheromone_update_started.elapsed().as_secs_f64(),
            candidates_built,
            candidates_geometrically_valid,
            candidates_evaluated,
            reference_decisions,
            policy_decisions,
            inference_seconds,
            reference_completed,
            simulation_seconds,
            reference_collisions,
            reference_timeouts,
            evaluation_failures,
        };
        self.state.iteration += 1;
        self.iterations.push(snapshot);
        let selected_ids: BTreeSet<_> = self
            .state
            .selected
            .iter()
            .map(|s| s.track.definition.id.clone())
            .collect();
        for it in &mut self.iterations {
            for a in &mut it.ants {
                if a.rejection.is_none()
                    && a.candidate_id
                        .as_ref()
                        .or_else(|| a.candidate.as_ref().map(|c| &c.definition.id))
                        .is_some_and(|id| !selected_ids.contains(id))
                {
                    a.rejection_reason = Some(RejectionReason::SelectionCapacity);
                    a.rejection = Some(
                        "Geometria válida; score abaixo das vagas disponíveis na seleção".into(),
                    );
                }
            }
        }
        self.apply_retention();
        if observer.is_observing() {
            observer.event(live::SearchEvent::Iteration {
                archive: Box::new(self.clone()),
            })?;
        }
        Ok(self.iterations.last().unwrap())
    }
    #[cfg(test)]
    fn construct(&mut self) -> Vec<DecisionSnapshot> {
        self.construct_observed(&mut live::Unobserved, &mut None)
            .unwrap()
    }
    fn construct_observed(
        &mut self,
        observer: &mut dyn live::SearchObserver,
        terminal: &mut Option<procedural::sequential::ConstructionView>,
    ) -> GenerationResult<Vec<DecisionSnapshot>> {
        if self.config.generator.is_sequential() {
            return self.construct_sequential(observer, terminal);
        }
        let mut decisions: Vec<DecisionSnapshot> = Vec::new();
        for position in 0..self.state.pheromones.len() {
            observer.checkpoint()?;
            let log_weights = self.state.pheromones[position]
                .iter()
                .enumerate()
                .map(|(choice, tau)| {
                    if self.config.strategy == SearchStrategy::RandomDifficulty {
                        return 0.;
                    }
                    // Smoothness heuristic; mild enough to preserve sharp/chicane candidates.
                    let eta = if position > 0 && position < self.config.generator.sectors {
                        let previous = decisions[position - 1].choice;
                        1. / (1. + 0.3 * choice.abs_diff(previous) as f64)
                    } else {
                        1.
                    };
                    self.config.alpha * tau.ln() + self.config.beta * eta.ln()
                })
                .collect::<Vec<_>>();
            // Log-sum-exp keeps selection finite for every allowed positive tau.
            let maximum = log_weights.iter().copied().max_by(f64::total_cmp).unwrap();
            let weights: Vec<_> = log_weights.iter().map(|w| (w - maximum).exp()).collect();
            let sum: f64 = weights.iter().sum();
            let mut probabilities = weights.iter().map(|w| w / sum).collect::<Vec<_>>();
            if self.config.strategy == SearchStrategy::AntColony {
                let best = weights
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                for p in &mut probabilities {
                    *p *= 1. - self.config.exploitation_probability;
                }
                probabilities[best] += self.config.exploitation_probability;
            }
            // Record the actual marginal selection distribution, including exploitation.
            let sample = self.state.construction_rng.random::<f64>();
            let mut cumulative = 0.;
            let choice = probabilities
                .iter()
                .position(|p| {
                    cumulative += p;
                    sample < cumulative
                })
                .unwrap_or(probabilities.len() - 1);
            let decision = DecisionSnapshot {
                position,
                choice,
                probabilities,
                construction: None,
            };
            decisions.push(decision.clone());
            observer.event(live::SearchEvent::Decision(decision))?;
        }
        Ok(decisions)
    }
    fn construct_sequential(
        &mut self,
        observer: &mut dyn live::SearchObserver,
        terminal: &mut Option<procedural::sequential::ConstructionView>,
    ) -> GenerationResult<Vec<DecisionSnapshot>> {
        use procedural::sequential::Construction;
        let config = &self.config.generator;
        let mut state = Construction::new(config);
        let mut decisions = vec![];
        while !state.closed() && state.used < config.sectors {
            observer.checkpoint()?;
            let mut options = state.options(config, &mut || observer.checkpoint())?;
            let log_weights: Vec<_> = options
                .iter()
                .enumerate()
                .map(|(choice, a)| {
                    if a.next.is_none() {
                        f64::NEG_INFINITY
                    } else if self.config.strategy == SearchStrategy::RandomDifficulty {
                        0.
                    } else {
                        self.config.alpha * self.state.pheromones[state.used][choice].ln()
                            + self.config.beta * a.eta.ln()
                    }
                })
                .collect();
            let weights = weights_from_logs(&log_weights);
            if weights.iter().sum::<f64>() == 0. {
                *terminal = Some(state.view(
                    state.points().len(),
                    options.iter().map(|a| a.view.clone()).collect(),
                ));
                return Ok(decisions);
            }
            let probabilities = choice_probabilities(
                &weights,
                if self.config.strategy == SearchStrategy::AntColony {
                    self.config.exploitation_probability
                } else {
                    0.
                },
            );
            let sample = self.state.construction_rng.random::<f64>();
            let mut total = 0.;
            let choice = probabilities
                .iter()
                .position(|p| {
                    total += p;
                    sample < total
                })
                .unwrap_or_else(|| probabilities.iter().rposition(|p| *p > 0.).unwrap());
            let start = state.points().len();
            let position = state.used;
            let views = options.iter().map(|a| a.view.clone()).collect();
            state = options[choice].next.take().unwrap();
            let decision = DecisionSnapshot {
                position,
                choice,
                probabilities,
                construction: Some(state.view(start, views)),
            };
            decisions.push(decision.clone());
            observer.event(live::SearchEvent::Decision(decision))?;
        }
        if state.closed() {
            observer.checkpoint()?;
            let row = config.sectors;
            let logs = self.state.pheromones[row]
                .iter()
                .map(|p| {
                    if self.config.strategy == SearchStrategy::RandomDifficulty {
                        0.
                    } else {
                        self.config.alpha * p.ln()
                    }
                })
                .collect::<Vec<_>>();
            let weights = weights_from_logs(&logs);
            let probabilities = choice_probabilities(
                &weights,
                if self.config.strategy == SearchStrategy::AntColony {
                    self.config.exploitation_probability
                } else {
                    0.
                },
            );
            let sample = self.state.construction_rng.random::<f64>();
            let mut total = 0.;
            let choice = probabilities
                .iter()
                .position(|p| {
                    total += p;
                    sample < total
                })
                .unwrap_or(probabilities.len() - 1);
            let decision = DecisionSnapshot {
                position: row,
                choice,
                probabilities,
                construction: Some(state.view(state.points().len(), vec![])),
            };
            decisions.push(decision.clone());
            observer.event(live::SearchEvent::Decision(decision))?;
        }
        Ok(decisions)
    }
    pub fn validate(&self) -> GenerationResult<()> {
        self.config.validate()?;
        if self.inspection_version > 1 {
            return Err("unsupported inspection metadata version".into());
        }
        if self.version != self.config.generator.version()
            || self.state.pheromones.len() != self.config.generator.choice_counts().len()
            || self
                .state
                .pheromones
                .iter()
                .zip(self.config.generator.choice_counts())
                .any(|(r, n)| {
                    r.len() != n
                        || r.iter().any(|p| {
                            !p.is_finite()
                                || *p < self.config.pheromone_min
                                || *p > self.config.pheromone_max
                        })
                })
            || self.state.iteration != self.iterations.len()
            || self.state.iteration > self.config.iterations
            || self.state.candidates > self.config.max_candidates
            || self.state.evaluation_steps > self.config.max_evaluation_steps
        {
            return Err("invalid saved ACO state/budget/pheromone dimensions".into());
        }
        if self.state.selected.len() > self.config.max_selected
            || self.state.decision_archive.len() > 256
            || self.state.geometry_archive.len() > 256
            || self
                .state
                .geometry_archive
                .iter()
                .any(|d| d.len() != 40 || d.iter().any(|v| !v.is_finite()))
        {
            return Err("saved ACO archive exceeds configured bounds".into());
        }
        for selected in &self.state.selected {
            selected.track.validate()?;
            if selected.track.provenance.config != self.config.generator
                || !self
                    .state
                    .seen_hashes
                    .contains(&selected.track.provenance.geometry_hash)
                || selected.fitness.reference_completed == Some(false)
                || [
                    selected.fitness.score,
                    selected.fitness.diversity,
                    selected.fitness.geometric_difficulty,
                ]
                .into_iter()
                .chain(selected.fitness.completion_rate)
                .chain(selected.fitness.mean_progress)
                .chain(selected.fitness.collision_rate)
                .chain(selected.fitness.organic_quality)
                .any(|x| !x.is_finite() || !(0.0..=1.).contains(&x))
            {
                return Err("saved ACO selected-track provenance/fitness is invalid".into());
            }
        }
        if self
            .state
            .decision_archive
            .iter()
            .any(|choices| !self.config.generator.decisions_valid(choices))
        {
            return Err("saved ACO decision archive shape is invalid".into());
        }
        Ok(())
    }
    pub fn save(&self, path: impl AsRef<Path>) -> GenerationResult<()> {
        self.validate()?;
        procedural::write_new_ron(path.as_ref(), self)
    }
    pub fn load(path: impl AsRef<Path>) -> GenerationResult<Self> {
        let search: Self = procedural::read_ron(path.as_ref())?;
        search.validate()?;
        Ok(search)
    }
    pub fn export_selected(&self, directory: impl AsRef<Path>) -> GenerationResult<Vec<String>> {
        self.validate()?;
        std::fs::create_dir_all(directory.as_ref()).map_err(|e| e.to_string())?;
        let mut ids = Vec::new();
        for ranked in &self.state.selected {
            let id = &ranked.track.definition.id;
            ranked
                .track
                .save(directory.as_ref().join(format!("{id}.generated.ron")))?;
            ranked
                .track
                .export_definition(directory.as_ref().join(format!("{id}.ron")))?;
            ids.push(id.clone());
        }
        Ok(ids)
    }
}
fn weights_from_logs(logs: &[f64]) -> Vec<f64> {
    let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    logs.iter()
        .map(|log| {
            if log.is_finite() && max.is_finite() {
                (log - max).exp()
            } else {
                0.
            }
        })
        .collect()
}
fn choice_probabilities(weights: &[f64], exploitation: f64) -> Vec<f64> {
    let max = weights.iter().copied().fold(0., f64::max);
    let scaled = weights.iter().map(|w| w / max).collect::<Vec<_>>();
    let sum = scaled.iter().sum::<f64>();
    let mut p = scaled
        .iter()
        .map(|w| w / sum * (1. - exploitation))
        .collect::<Vec<_>>();
    let best = weights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    p[best] += exploitation;
    p
}
fn useful_difficulty(progress: f64, target: f64) -> f64 {
    if progress <= target {
        progress / target
    } else {
        (1. - progress) / (1. - target)
    }
    .clamp(0., 1.)
}
fn decision_diversity(choices: &[usize], archive: &[Vec<usize>]) -> f64 {
    // Circular shifts merely rotate radial circuits. Count them as near duplicates
    // even though their world-coordinate hashes differ.
    let sectors = choices.len().saturating_sub(1);
    if sectors == 0 {
        return 0.;
    }
    archive
        .iter()
        .filter(|other| other.len() == choices.len())
        .flat_map(|other| {
            (0..sectors).map(move |shift| {
                let radial_differences = (0..sectors)
                    .filter(|i| choices[*i] != other[(i + shift) % sectors])
                    .count();
                (radial_differences + usize::from(choices[sectors] != other[sectors])) as f64
                    / choices.len() as f64
            })
        })
        .min_by(f64::total_cmp)
        .unwrap_or(1.)
}

mod saved_rng {
    use super::*;
    use serde::{Deserializer, Serializer};
    #[derive(Serialize, Deserialize)]
    struct Saved {
        seed: [u8; 32],
        stream: u64,
        position: [u64; 2],
    }
    pub fn serialize<S: Serializer>(rng: &ChaCha12Rng, serializer: S) -> Result<S::Ok, S::Error> {
        let position = rng.get_word_pos();
        Saved {
            seed: rng.get_seed(),
            stream: rng.get_stream(),
            position: [position as u64, (position >> 64) as u64],
        }
        .serialize(serializer)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<ChaCha12Rng, D::Error> {
        let saved = Saved::deserialize(deserializer)?;
        let mut rng = ChaCha12Rng::from_seed(saved.seed);
        rng.set_stream(saved.stream);
        rng.set_word_pos(saved.position[0] as u128 | ((saved.position[1] as u128) << 64));
        Ok(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn decisions(search: &SearchArchive) -> Vec<Vec<usize>> {
        search
            .iterations
            .iter()
            .flat_map(|i| {
                i.ants
                    .iter()
                    .map(|a| a.decisions.iter().map(|d| d.choice).collect())
            })
            .collect()
    }
    #[test]
    fn old_search_configs_retain_legacy_fitness_identity() {
        let old: AcoConfig = ron::from_str("(ants:2,iterations:1)").unwrap();
        assert_eq!(old.fitness_objective, FitnessObjective::LegacyAdversarial);
        assert_eq!(
            AcoConfig::default().fitness_objective,
            FitnessObjective::UsefulProgress
        );
    }
    #[test]
    fn useful_difficulty_rewards_partial_success_and_failed_evaluation_cannot_select() {
        assert!(useful_difficulty(0.7, 0.7) > useful_difficulty(0.01, 0.7));
        assert!(useful_difficulty(0.7, 0.7) > useful_difficulty(1., 0.7));
        let mut search = SearchArchive::new(
            AcoConfig {
                generator: GeneratorConfig::legacy(),
                ants: 4,
                iterations: 1,
                max_candidates: 4,
                max_evaluation_steps: 20,
                policy_label: Some("test-error".into()),
                ..Default::default()
            },
            17,
        )
        .unwrap();
        let mut fail = |_: &GeneratedTrack, _: u64, _: usize| Err("execution failure".to_string());
        search.run_iteration(Some(&mut fail)).unwrap();
        assert!(search.state.selected.is_empty());
        assert_eq!(search.state.evaluation_steps, 20);
        assert!(
            search.iterations[0]
                .ants
                .iter()
                .any(|a| a.rejection_reason == Some(RejectionReason::ExecutionFailure))
        );
    }
    #[test]
    fn probabilistic_deterministic_bounded_real_pheromone_updates() {
        let config = AcoConfig {
            generator: GeneratorConfig::legacy(),
            ants: 4,
            iterations: 2,
            max_candidates: 8,
            ..Default::default()
        };
        let mut a = SearchArchive::new(config.clone(), 17).unwrap();
        let mut b = SearchArchive::new(config, 17).unwrap();
        while !a.is_finished() {
            a.run_iteration(None).unwrap();
            b.run_iteration(None).unwrap();
        }
        assert_eq!(decisions(&a), decisions(&b));
        assert_eq!(a.state.candidates, 8);
        assert!(!a.state.selected.is_empty());
        let s = &a.iterations[0];
        assert!(
            s.pheromones_before
                .iter()
                .flatten()
                .zip(s.pheromones_evaporated.iter().flatten())
                .all(|(x, y)| (*x * 0.8 - y).abs() < 1.0e-10)
        );
        assert!(
            s.pheromones_after
                .iter()
                .flatten()
                .zip(s.pheromones_evaporated.iter().flatten())
                .any(|(x, y)| x > y)
        );
        assert!(
            a.state
                .pheromones
                .iter()
                .flatten()
                .all(|p| p.is_finite() && *p >= 0.05 && *p <= 5.)
        );
        for ant in &s.ants {
            for d in &ant.decisions {
                assert!((d.probabilities.iter().sum::<f64>() - 1.).abs() < 1.0e-10);
                assert!(d.probabilities.iter().all(|p| *p >= 0.));
                assert!(d.probabilities[d.choice] > 0.);
            }
        }
        assert!(a.run_iteration(None).is_err());
    }
    #[test]
    fn saved_rng_resumes_exact_decisions_and_selected_tracks() {
        let config = AcoConfig {
            generator: GeneratorConfig::legacy(),
            ants: 3,
            iterations: 2,
            max_candidates: 6,
            ..Default::default()
        };
        let mut a = SearchArchive::new(config, 42).unwrap();
        a.run_iteration(None).unwrap();
        let mut b: SearchArchive = ron::from_str(&ron::to_string(&a).unwrap()).unwrap();
        b.validate().unwrap();
        a.run_iteration(None).unwrap();
        b.run_iteration(None).unwrap();
        assert_eq!(decisions(&a), decisions(&b));
        assert_eq!(a.state.pheromones, b.state.pheromones);
        assert_eq!(a.state.selected[0].track, b.state.selected[0].track);
    }
    #[test]
    fn unverified_reference_is_distinct_and_evaluation_budget_enforced() {
        let config = AcoConfig {
            generator: GeneratorConfig::legacy(),
            ants: 4,
            iterations: 1,
            max_candidates: 4,
            policy_label: Some("frozen-training-v1".into()),
            max_evaluation_steps: 2,
            ..Default::default()
        };
        let mut search = SearchArchive::new(config, 17).unwrap();
        let mut evaluator = |_: &GeneratedTrack, _: u64, budget: usize| {
            Ok(PilotEvaluation {
                completion_rate: 0.,
                mean_progress: 0.2,
                collision_rate: 1.,
                reference_completed: false,
                reference_collision: true,
                simulation_steps: budget.min(2) as u64,
                reference_steps: 1,
                policy_steps: 1,
                policy_episodes: 1,
                inference_seconds: 0.,
                ..Default::default()
            })
        };
        search.run_iteration(Some(&mut evaluator)).unwrap();
        assert!(search.state.selected.is_empty());
        assert!(search.state.evaluation_steps <= 2);
        assert!(
            search.iterations[0]
                .ants
                .iter()
                .any(|a| a.rejection_reason == Some(RejectionReason::ReferenceCollision))
        );
    }
    #[test]
    fn stagnation_resets_pheromones_after_rejections() {
        let config = AcoConfig {
            ants: 2,
            iterations: 1,
            max_candidates: 2,
            stagnation_iterations: 1,
            generator: GeneratorConfig {
                min_length: 1.,
                max_length: 2.,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut search = SearchArchive::new(config, 7).unwrap();
        let snapshot = search.run_iteration(None).unwrap();
        assert!(snapshot.stagnation_reset);
        assert!(snapshot.ants.iter().all(|a| a.rejection.is_some()));
        assert!(snapshot.pheromones_after.iter().flatten().all(|p| *p == 1.));
    }
    #[test]
    fn construction_and_evaluation_rngs_are_isolated() {
        let config = AcoConfig {
            generator: GeneratorConfig::legacy(),
            ants: 3,
            iterations: 1,
            max_candidates: 3,
            policy_label: Some("frozen-v0".into()),
            ..Default::default()
        };
        let mut with_eval = SearchArchive::new(config.clone(), 27).unwrap();
        let mut without_eval = SearchArchive::new(config, 27).unwrap();
        let mut evaluator = |_: &GeneratedTrack, _: u64, _: usize| {
            Ok(PilotEvaluation {
                completion_rate: 0.5,
                mean_progress: 0.7,
                collision_rate: 0.2,
                reference_completed: true,
                reference_collision: false,
                simulation_steps: 1,
                reference_steps: 0,
                policy_steps: 1,
                policy_episodes: 1,
                inference_seconds: 0.,
                ..Default::default()
            })
        };
        with_eval.run_iteration(Some(&mut evaluator)).unwrap();
        without_eval.run_iteration(None).unwrap();
        assert_eq!(decisions(&with_eval), decisions(&without_eval));
    }
    #[test]
    fn extreme_pheromone_ranges_keep_selection_probabilities_finite() {
        let config = AcoConfig {
            generator: GeneratorConfig::legacy(),
            ants: 1,
            iterations: 1,
            max_candidates: 1,
            alpha: 8.,
            pheromone_min: 1.0e-300,
            pheromone_max: 1.0e300,
            initial_pheromone: 1.0e300,
            ..Default::default()
        };
        let mut search = SearchArchive::new(config, 4).unwrap();
        search.state.pheromones[0][0] = 1.0e-300;
        for d in search.construct() {
            assert!(d.probabilities.iter().all(|p| p.is_finite()));
            assert!((d.probabilities.iter().sum::<f64>() - 1.).abs() < 1.0e-10);
        }
    }
    #[test]
    fn rotational_copies_do_not_inflate_diversity() {
        let original = vec![0, 1, 2, 3, 2, 1, 1];
        let rotated = vec![2, 3, 2, 1, 0, 1, 1];
        assert_eq!(decision_diversity(&rotated, &[original]), 0.);
    }
    #[test]
    fn random_difficulty_has_equal_candidate_budget_without_pheromone_learning() {
        let config = AcoConfig {
            generator: GeneratorConfig::legacy(),
            strategy: SearchStrategy::RandomDifficulty,
            ants: 3,
            iterations: 1,
            max_candidates: 3,
            ..Default::default()
        };
        let mut search = SearchArchive::new(config, 4).unwrap();
        search.run_iteration(None).unwrap();
        assert_eq!(search.state.candidates, 3);
        assert!(search.state.pheromones.iter().flatten().all(|p| *p == 1.));
    }
}
