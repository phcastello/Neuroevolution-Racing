//! Curriculum changes are applied only after collection + PPO update, then all episodes restart.
use super::*;
use crate::{
    aco::{AcoConfig, CandidateEvaluator, PilotEvaluation, SearchArchive, SearchStrategy},
    simulation::procedural::{self, GeneratedTrack, GeneratorConfig},
};
use racing_rl::config::{CurriculumConfig, CurriculumKind};
use serde::{Deserialize, Serialize};
use std::rc::Rc;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CurriculumState {
    pub activations: u64,
    #[serde(default)]
    pub curriculum_rollouts: u64,
    #[serde(default)]
    pub selected_rollouts: u64,
    #[serde(default)]
    pub used_generated_tracks: std::collections::BTreeSet<String>,
    #[serde(default)]
    pub used_selected_tracks: std::collections::BTreeSet<String>,
    #[serde(default)]
    pub search_seconds: f64,
    pub active: Vec<GeneratedTrack>,
    pub history: Vec<GeneratedTrack>,
    pub fresh: Vec<GeneratedTrack>,
    pub generator: GeneratorConfig,
    pub last_search: Option<SearchArchive>,
}
impl CurriculumState {
    pub fn decode(saved: Option<&str>, c: &Config) -> Result<Self> {
        let state: Self = match saved {
            Some(s) => ron::from_str(s).map_err(|e| racing_rl::Error::Invalid(e.to_string()))?,
            None => Self::default(),
        };
        ensure(
            c.curriculum.is_some() || saved.is_none(),
            "curriculum state without curriculum config",
        )?;
        if let Some(options) = &c.curriculum {
            ensure(
                state.active.len() <= options.selected
                    && state.history.len() <= options.archive_limit
                    && state.fresh.len() <= options.selected,
                "curriculum archive limits",
            )?;
        }
        for track in state
            .active
            .iter()
            .chain(&state.history)
            .chain(&state.fresh)
        {
            track.validate().map_err(racing_rl::Error::Invalid)?;
        }
        if let Some(search) = &state.last_search {
            search.validate().map_err(racing_rl::Error::Invalid)?;
            ensure(
                search.config.fitness_objective == crate::aco::FitnessObjective::UsefulProgress
                    || c.curriculum
                        .as_ref()
                        .is_some_and(|o| o.kind == CurriculumKind::Random),
                "legacy adversarial curriculum requires an explicit new scientific run; fitness objective changed",
            )?;
        }
        Ok(state)
    }
    pub fn observe_rollout(
        &mut self,
        tracks: &std::collections::BTreeMap<String, env::TrackMetrics>,
    ) {
        let mut generated = false;
        let mut selected = false;
        for (id, metrics) in tracks {
            if metrics.decisions == 0 {
                continue;
            }
            if id.starts_with("generated_") {
                generated = true;
                self.used_generated_tracks.insert(id.clone());
            }
            if self
                .active
                .iter()
                .chain(&self.history)
                .any(|t| &t.definition.id == id)
            {
                selected = true;
                self.used_selected_tracks.insert(id.clone());
            }
        }
        self.curriculum_rollouts += u64::from(generated);
        self.selected_rollouts += u64::from(selected);
    }
    pub fn encode(&self, c: &Config) -> Result<Option<String>> {
        if c.curriculum.is_none() {
            return Ok(None);
        }
        Ok(Some(
            ron::to_string(self).map_err(|e| racing_rl::Error::Invalid(e.to_string()))?,
        ))
    }
    pub fn scenes(
        &self,
        originals: &[Rc<env::Scene>],
        options: &CurriculumConfig,
    ) -> Result<Vec<Rc<env::Scene>>> {
        let mut scenes = vec![];
        // Weights are explicit multiplicities in the existing uniform scene sampler.
        for _ in 0..options.mix[0] {
            scenes.extend_from_slice(originals);
        }
        for (tracks, weight) in [
            (&self.fresh, options.mix[1]),
            (&self.active, options.mix[2]),
            (&self.history, options.mix[3]),
        ] {
            for track in tracks {
                let scene = Rc::new(env::Scene::new(
                    track.validate().map_err(racing_rl::Error::Invalid)?,
                ));
                for _ in 0..weight {
                    scenes.push(scene.clone());
                }
            }
        }
        Ok(scenes)
    }
    pub fn activate<B: Backend>(
        &mut self,
        network: &Network<B>,
        c: &Config,
        p: &SimulationConfig,
        device: &B::Device,
        completed: u64,
        dir: &Path,
    ) -> Result<()> {
        let options = c.curriculum.as_ref().expect("enabled curriculum");
        ensure(
            options.due(completed),
            "curriculum activation outside configured rollout boundary",
        )?;
        let started = Instant::now();
        let seed = options
            .seed
            .wrapping_add(self.activations.wrapping_mul(0x9e3779b97f4a7c15));
        let n = options.candidate_limit();
        let mut search = SearchArchive::new(
            AcoConfig {
                generator: self.generator.clone(),
                ants: options.ants,
                iterations: options.iterations,
                max_candidates: n,
                max_selected: options.selected,
                max_evaluation_steps_per_candidate: options.evaluation_ticks as usize,
                max_evaluation_steps: options.decision_limit(),
                policy_label: Some(format!(
                    "training policy version {} at rollout {completed}",
                    network.version()
                )),
                strategy: if options.kind == CurriculumKind::Aco {
                    SearchStrategy::AntColony
                } else {
                    SearchStrategy::RandomDifficulty
                },
                ..Default::default()
            },
            seed,
        )
        .map_err(racing_rl::Error::Invalid)?;
        for track in self.active.iter().chain(&self.history).chain(&self.fresh) {
            search
                .state
                .seen_hashes
                .insert(track.provenance.geometry_hash.clone());
            search
                .state
                .decision_archive
                .push(track.provenance.decisions.clone());
        }
        if search.state.decision_archive.len() > 256 {
            search
                .state
                .decision_archive
                .drain(..search.state.decision_archive.len() - 256);
        }
        let mut evaluator = |track: &GeneratedTrack, seed, budget| {
            evaluate_candidate(network, c, p, device, track, seed, budget)
        };
        // Immutable policy borrow prevents optimizer or training action/environment RNG access.
        while !search.is_finished() {
            let evaluator: Option<&mut dyn CandidateEvaluator> =
                if options.kind == CurriculumKind::Random {
                    None
                } else {
                    Some(&mut evaluator)
                };
            search
                .run_iteration(evaluator)
                .map_err(racing_rl::Error::Invalid)?;
        }
        let selected = if options.kind == CurriculumKind::Random {
            // Random curriculum takes first valid candidates, without fitness ranking.
            search
                .iterations
                .iter()
                .flat_map(|s| &s.ants)
                .filter(|a| a.rejection.is_none())
                .filter_map(|a| a.candidate.clone())
                .take(options.selected)
                .collect()
        } else {
            search
                .state
                .selected
                .iter()
                .map(|x| x.track.clone())
                .collect()
        };
        for track in self.active.drain(..) {
            if !self
                .history
                .iter()
                .any(|x| x.provenance.geometry_hash == track.provenance.geometry_hash)
            {
                self.history.push(track);
            }
        }
        if self.history.len() > options.archive_limit {
            self.history
                .drain(..self.history.len() - options.archive_limit);
        }
        self.active = selected;
        self.fresh.clear();
        for i in 0..options.selected {
            if let Ok(track) =
                procedural::generate(&self.generator, seed.wrapping_add(0x4652455348 + i as u64))
            {
                self.fresh.push(track);
            }
        }
        let activation_dir = dir.join(format!("curriculum-{completed:06}"));
        std::fs::create_dir(&activation_dir)?;
        search
            .save(activation_dir.join("search.ron"))
            .map_err(racing_rl::Error::Invalid)?;
        search.config.max_candidates = n.min(24);
        search
            .export_selected(activation_dir.join("search-ranked-tracks"))
            .map_err(racing_rl::Error::Invalid)?;
        let active_dir = activation_dir.join("tracks");
        std::fs::create_dir(&active_dir)?;
        for track in &self.active {
            track
                .save(active_dir.join(format!("{}.generated.ron", track.definition.id)))
                .map_err(racing_rl::Error::Invalid)?;
            track
                .export_definition(active_dir.join(format!("{}.ron", track.definition.id)))
                .map_err(racing_rl::Error::Invalid)?;
        }
        let cost = serde_json::json!({
            "activation_rollout": completed,
            "policy_version": network.version(),
            "search_seconds": started.elapsed().as_secs_f64(),
            "candidates_built": search.iterations.iter().map(|i| i.candidates_built).sum::<usize>(),
            "candidates_geometrically_valid": search.iterations.iter().map(|i| i.candidates_geometrically_valid).sum::<usize>(),
            "candidates_evaluated": search.iterations.iter().map(|i| i.candidates_evaluated).sum::<usize>(),
            "evaluation_decisions": search.state.evaluation_steps,
            "reference_decisions": search.iterations.iter().map(|i| i.reference_decisions).sum::<u64>(),
            "policy_decisions": search.iterations.iter().map(|i| i.policy_decisions).sum::<u64>(),
            "reference_completed": search.iterations.iter().map(|i| i.reference_completed).sum::<usize>(),
            "reference_collisions": search.iterations.iter().map(|i| i.reference_collisions).sum::<usize>(),
            "reference_timeouts": search.iterations.iter().map(|i| i.reference_timeouts).sum::<usize>(),
            "evaluation_failures": search.iterations.iter().map(|i| i.evaluation_failures).sum::<usize>(),
            "generation_seconds": search.iterations.iter().map(|i| i.generation_seconds).sum::<f64>(),
            "evaluation_seconds": search.iterations.iter().map(|i| i.evaluation_seconds).sum::<f64>(),
            "simulation_seconds": search.iterations.iter().map(|i| i.simulation_seconds).sum::<f64>(),
            "inference_seconds": search.iterations.iter().map(|i| i.inference_seconds).sum::<f64>(),
            "pheromone_update_seconds": search.iterations.iter().map(|i| i.pheromone_update_seconds).sum::<f64>(),
            "selected_tracks": search.state.selected.len(),
            "curriculum_rollouts_before_activation": self.curriculum_rollouts,
            "training_tracks_distinct": self.active.iter().map(|t| t.provenance.geometry_hash.as_str()).collect::<std::collections::BTreeSet<_>>().len(),
            "prepared_for_continuation": completed >= c.total_rollouts as u64,
        });
        std::fs::write(
            activation_dir.join("cost.json"),
            serde_json::to_vec_pretty(&cost)
                .map_err(|e| racing_rl::Error::Invalid(e.to_string()))?,
        )?;
        self.search_seconds += started.elapsed().as_secs_f64();
        self.last_search = Some(search);
        self.activations += 1;
        Ok(())
    }
}

pub fn evaluate_candidate<B: Backend>(
    n: &Network<B>,
    c: &Config,
    p: &SimulationConfig,
    device: &B::Device,
    track: &GeneratedTrack,
    seed: u64,
    budget: usize,
) -> std::result::Result<PilotEvaluation, String> {
    evaluate_candidate_controlled(n, c, p, device, track, seed, budget, &mut || Ok(()))
}

fn evaluate_candidate_controlled<B: Backend>(
    n: &Network<B>,
    c: &Config,
    p: &SimulationConfig,
    device: &B::Device,
    track: &GeneratedTrack,
    seed: u64,
    budget: usize,
    checkpoint: &mut dyn FnMut() -> std::result::Result<(), String>,
) -> std::result::Result<PilotEvaluation, String> {
    checkpoint()?;
    let t = track.validate()?;
    let scene = Rc::new(env::Scene::new(t));
    let mut config = c.clone();
    config.num_envs = 1;
    config.curriculum = None;
    // Share the explicit budget between reference and frozen PPO, counting every physics decision.
    let per_pilot = budget / 2;
    if per_pilot == 0 {
        return Err("candidate budget requires at least two decisions".into());
    }
    config.timeout_ticks = c
        .timeout_ticks
        .min((per_pilot as u64).saturating_mul(c.action_repeat as u64));
    let mut reference = env::RacingEnvironment::new(
        0,
        vec![scene.clone()],
        &config,
        p,
        ChaCha12Rng::seed_from_u64(seed),
    )
    .map_err(|e| e.to_string())?;
    let mut result = PilotEvaluation {
        reference_episodes: 1,
        ..Default::default()
    };
    for _ in 0..per_pilot {
        checkpoint()?;
        let started = Instant::now();
        let a = reference.reference_action();
        let step = reference.step(a).map_err(|e| e.to_string())?;
        result.simulation_seconds += started.elapsed().as_secs_f64();
        result.reference_steps += 1;
        result.reference_progress = result.reference_progress.max(step.progress as f64);
        if step.terminated || step.truncated {
            result.reference_completed = step.reason == Some(EndReason::Completed);
            result.reference_collision = step.reason == Some(EndReason::Collision);
            break;
        }
    }
    result.simulation_steps = result.reference_steps;
    result.budget_insufficient = !result.reference_completed
        && !result.reference_collision
        && config.timeout_ticks < c.timeout_ticks;
    // Do not spend frozen PPO inference on a reference failure already sufficient
    // to reject. This is unverified drivability, never physical impossibility.
    if !result.reference_completed {
        return Ok(result);
    }
    let mut policy =
        env::RacingEnvironment::new(0, vec![scene], &config, p, ChaCha12Rng::seed_from_u64(seed))
            .map_err(|e| e.to_string())?;
    result.policy_episodes = 1;
    for _ in 0..per_pilot {
        checkpoint()?;
        let observation = policy.observe().map_err(|e| e.to_string())?;
        let started = Instant::now();
        let output = n
            .infer(matrix(&observation, 1, n.input_size, device).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let action = read(output.mean.tanh()).map_err(|e| e.to_string())?;
        result.inference_seconds += started.elapsed().as_secs_f64();
        let started = Instant::now();
        let step = policy
            .step([action[0], action[1]])
            .map_err(|e| e.to_string())?;
        result.simulation_seconds += started.elapsed().as_secs_f64();
        result.policy_steps += 1;
        result.mean_progress = result.mean_progress.max(step.progress as f64).clamp(0., 1.);
        if step.terminated || step.truncated {
            result.completion_rate = f64::from(step.reason == Some(EndReason::Completed));
            result.collision_rate = f64::from(step.reason == Some(EndReason::Collision));
            break;
        }
    }
    result.simulation_steps += result.policy_steps;
    Ok(result)
}
/// Headless standalone ACO uses one frozen CPU copy throughout the complete search.
pub fn checkpoint_evaluator(path: &Path) -> std::result::Result<impl CandidateEvaluator, String> {
    let saved: Checkpoint = persistence::load(path).map_err(|e| e.to_string())?;
    let device = Default::default();
    let (n, _optimizer, p) = restore::<Cpu>(&saved, &device).map_err(|e| e.to_string())?;
    let c = saved.config;
    Ok(move |track: &GeneratedTrack, seed, budget| {
        evaluate_candidate(&n, &c, &p, &device, track, seed, budget)
    })
}
/// Visual ACO uses the identical frozen evaluation with cooperative tick gates.
pub fn checkpoint_evaluator_controlled(
    path: &Path,
    control: crate::aco::live::SearchControl,
) -> std::result::Result<impl CandidateEvaluator, String> {
    let saved: Checkpoint = persistence::load(path).map_err(|e| e.to_string())?;
    control.checkpoint()?;
    let device = Default::default();
    let (n, _optimizer, p) = restore::<Cpu>(&saved, &device).map_err(|e| e.to_string())?;
    let c = saved.config;
    Ok(move |track: &GeneratedTrack, seed, budget| {
        evaluate_candidate_controlled(&n, &c, &p, &device, track, seed, budget, &mut || {
            control.checkpoint()
        })
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schedule_is_at_safe_boundaries() {
        let c = CurriculumConfig::default();
        for i in 0..100 {
            assert!(!c.due(i));
        }
        assert!(c.due(100));
        assert!(!c.due(101));
        assert!(c.due(150));
    }
    #[test]
    fn curriculum_roundtrip_and_training_only() {
        let mut c = Config::default();
        c.curriculum = Some(Default::default());
        let mut state = CurriculumState::default();
        state
            .active
            .push(procedural::generate(&state.generator, 1).unwrap());
        let text = state.encode(&c).unwrap();
        let rebuilt = CurriculumState::decode(text.as_deref(), &c).unwrap();
        assert_eq!(rebuilt.active, state.active);
        let mut invalid = rebuilt;
        invalid.active[0].definition.role = crate::simulation::TrackRole::Validation;
        assert!(CurriculumState::decode(invalid.encode(&c).unwrap().as_deref(), &c).is_err());
    }
    #[test]
    fn historical_format_fixture_keeps_original_angles_and_rejects_31_input_mismatch() {
        let c = Config {
            hidden: vec![4],
            num_envs: 1,
            rollout_length: 2,
            minibatch_size: 2,
            ..Default::default()
        };
        let network = Network::<Cpu>::new(&c, &Default::default()).unwrap();
        let optimizer = Optimizer::new(c.optimizer.clone(), &network.parameters());
        let saved = checkpoint(
            &network,
            &optimizer,
            SaveContext {
                config: &c,
                physics: &SimulationConfig::default(),
                rngs: &Rngs::new(&c),
                backend: "historical-compatible-fixture-cpu",
                completed_rollouts: 0,
                decisions: 0,
                physical_ticks: 0,
                curriculum_ron: None,
            },
        )
        .unwrap();
        let dir = persistence::run_directory(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/audit-validation/historical-fixture"),
        )
        .unwrap();
        let path = dir.join("checkpoint-25.ron");
        persistence::save(&path, &saved).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(!text.contains("anticipation:"));
        let mut loaded: Checkpoint = persistence::load(&path).unwrap();
        let (restored, adam, _) = restore::<Cpu>(&loaded, &Default::default()).unwrap();
        assert_eq!(restored.input_size, 25);
        assert_eq!(
            loaded.config.observation.angles(),
            racing_rl::config::SensorLayout::Original.angles()
        );
        assert!(loaded.config.observation.anticipation.is_none());
        assert_eq!(restored.snapshot().unwrap(), saved.parameters);
        assert_eq!(adam.snapshot().unwrap().first, saved.optimizer.first);
        let mut gpu_config = loaded.config.clone();
        gpu_config.device = DeviceChoice::Gpu { adapter: 0 };
        super::super::validate_resume(&gpu_config, &loaded.config).unwrap();
        loaded.config.observation = loaded.config.observation.with_anticipation(true);
        assert!(restore::<Cpu>(&loaded, &Default::default()).is_err());
    }
    #[test]
    fn live_worker_loads_frozen_checkpoint_and_runs_real_policy_evaluation() {
        use crate::aco::live::{LiveSearch, RunMode, SearchEvent};
        let c = Config {
            hidden: vec![4],
            num_envs: 1,
            rollout_length: 2,
            minibatch_size: 2,
            timeout_ticks: 10800,
            ..Default::default()
        };
        let network = Network::<Cpu>::new(&c, &Default::default()).unwrap();
        let optimizer = Optimizer::new(c.optimizer.clone(), &network.parameters());
        let saved = checkpoint(
            &network,
            &optimizer,
            SaveContext {
                config: &c,
                physics: &SimulationConfig::default(),
                rngs: &Rngs::new(&c),
                backend: "live-aco-test-cpu",
                completed_rollouts: 0,
                decisions: 0,
                physical_ticks: 0,
                curriculum_ron: None,
            },
        )
        .unwrap();
        let root = persistence::run_directory(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/aco-frozen-worker-test"),
        )
        .unwrap();
        let path = root.join("checkpoint.ron");
        persistence::save(&path, &saved).unwrap();
        let original = std::fs::read(&path).unwrap();
        let live = LiveSearch::start(
            AcoConfig {
                ants: 1,
                iterations: 1,
                max_candidates: 1,
                max_evaluation_steps: 16000,
                max_evaluation_steps_per_candidate: 16000,
                ..Default::default()
            },
            47,
            Some(path.clone()),
            RunMode::Iteration,
            120.,
        )
        .unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(60);
        let archive = loop {
            let mut finished = None;
            for event in live.poll() {
                match event {
                    SearchEvent::Finished { archive } => finished = Some(archive),
                    SearchEvent::Failed(e) => panic!("{e}"),
                    _ => (),
                }
            }
            if let Some(a) = finished {
                break a;
            }
            assert!(Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(2));
        };
        let evaluation = archive.iterations[0].ants[0].evaluation.as_ref().unwrap();
        assert!(evaluation.reference_completed);
        assert!(evaluation.policy_steps > 0);
        assert!(
            archive
                .config
                .policy_label
                .as_ref()
                .unwrap()
                .contains("fnv1a64=")
        );
        assert_eq!(std::fs::read(path).unwrap(), original);
        assert_eq!(network.snapshot().unwrap(), saved.parameters);
        println!(
            "frozen worker: reference={} PPO={} decisions; immutable checkpoint",
            evaluation.reference_steps, evaluation.policy_steps
        );
    }
    #[test]
    fn evaluation_tick_pause_resume_cancel_and_frozen_network() {
        use crate::aco::live::{RunMode, SearchControl};
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };
        use std::time::Duration;
        let ticks = Arc::new(AtomicUsize::new(0));
        let control = SearchControl::new(RunMode::Running, 120.);
        let worker_control = control.clone();
        let worker_ticks = ticks.clone();
        let worker = std::thread::spawn(move || {
            let c = Config {
                hidden: vec![4],
                timeout_ticks: 3000,
                ..Default::default()
            };
            let n = Network::<Cpu>::new(&c, &Default::default()).unwrap();
            let before = n.snapshot().unwrap();
            let t = procedural::generate(&GeneratorConfig::legacy(), 3).unwrap();
            let result = evaluate_candidate_controlled(
                &n,
                &c,
                &Default::default(),
                &Default::default(),
                &t,
                7,
                6000,
                &mut || {
                    if worker_ticks.load(Ordering::SeqCst) == 4 {
                        worker_control.set_mode(RunMode::Paused);
                    }
                    worker_control.checkpoint()?;
                    worker_ticks.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(before, n.snapshot().unwrap());
            assert!(result.reference_completed && result.policy_steps > 0);
            assert_eq!(
                worker_ticks.load(Ordering::SeqCst) as u64,
                result.simulation_steps + 1
            );
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        while !control.is_paused() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(ticks.load(Ordering::SeqCst), 4);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(ticks.load(Ordering::SeqCst), 4);
        control.set_mode(RunMode::Running);
        worker.join().unwrap();
        let c = Config {
            hidden: vec![4],
            ..Default::default()
        };
        let n = Network::<Cpu>::new(&c, &Default::default()).unwrap();
        let t = procedural::generate(&GeneratorConfig::legacy(), 3).unwrap();
        let mut count = 0;
        let error = evaluate_candidate_controlled(
            &n,
            &c,
            &Default::default(),
            &Default::default(),
            &t,
            7,
            6000,
            &mut || {
                count += 1;
                if count == 5 {
                    Err("search cancelled".into())
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert_eq!(error, "search cancelled");
        assert_eq!(count, 5);
    }
    #[test]
    fn evaluator_freezes_parameters_and_bounds_work() {
        let c = Config {
            hidden: vec![8],
            timeout_ticks: 3000,
            ..Default::default()
        };
        let n = Network::<Cpu>::new(&c, &Default::default()).unwrap();
        let before = n.snapshot().unwrap();
        let t = procedural::generate(&GeneratorConfig::legacy(), 3).unwrap();
        let result =
            evaluate_candidate(&n, &c, &Default::default(), &Default::default(), &t, 7, 8).unwrap();
        assert!(result.simulation_steps <= 8);
        assert_eq!(before, n.snapshot().unwrap());
        assert!(result.reference_steps > 0);
        assert_eq!(result.policy_steps, 0);
        assert!(result.budget_insufficient);
        assert!(result.simulation_steps <= 8);
        assert!(result.inference_seconds.is_finite());
        let full = evaluate_candidate(
            &n,
            &c,
            &Default::default(),
            &Default::default(),
            &t,
            7,
            6000,
        )
        .unwrap();
        assert!(full.reference_completed);
        assert!(full.policy_steps > 0);
        assert_eq!(full.policy_episodes, 1);
        assert_eq!(before, n.snapshot().unwrap());
    }
}

#[cfg(test)]
mod training_integration_tests {
    use super::super::{Job, persistence_policy::PersistenceOptions, session_budget};
    use super::*;
    #[test]
    fn training_updates_curriculum_only_between_rollouts_and_resumes_saved_state() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/audit-validation")
            .join(format!(
                "racing-curriculum-test-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        std::fs::create_dir(&root).unwrap();
        let c = Config {
            hidden: vec![4],
            num_envs: 1,
            rollout_length: 2,
            minibatch_size: 2,
            update_epochs: 1,
            total_rollouts: 2,
            timeout_ticks: 10800,
            results_root: root.join("runs").display().to_string(),
            training_tracks: vec!["interlagos".into()],
            curriculum: Some(CurriculumConfig {
                kind: CurriculumKind::Aco,
                warmup: 1,
                interval: 1,
                ants: 3,
                iterations: 1,
                selected: 2,
                archive_limit: 4,
                mix: [1, 0, 32, 0],
                evaluation_ticks: 10800,
                ..Default::default()
            }),
            ..Default::default()
        };
        let c = super::super::resolve_config(c).unwrap();
        let options = PersistenceOptions {
            save_every: std::num::NonZeroU64::new(100),
            save_first: Some(0),
            save_last: Some(0),
            no_save_rollouts: false,
            run_path_file: Some(root.join("receipt1")),
            ..Default::default()
        };
        super::super::execute_with_clock::<Cpu, _>(
            &c,
            Job::Train(None, options),
            &Default::default(),
            "integration CPU",
            session_budget::Monotonic(Instant::now()),
        )
        .unwrap();
        let first = PathBuf::from(
            std::fs::read_to_string(root.join("receipt1"))
                .unwrap()
                .trim(),
        );
        let saved: Checkpoint = persistence::load(&first.join("checkpoint-000002.ron")).unwrap();
        saved.validate().unwrap();
        assert_ne!(
            saved.parameters,
            Network::<Cpu>::new(&c, &Default::default())
                .unwrap()
                .snapshot()
                .unwrap()
        );
        let second_rollout: Rollout = persistence::load(&first.join("rollout-000002.ron")).unwrap();
        second_rollout.validate().unwrap();
        assert_eq!(second_rollout.policy_version + 1, saved.policy_version);
        let (restored_network, restored_adam, _) =
            restore::<Cpu>(&saved, &Default::default()).unwrap();
        assert_eq!(restored_network.snapshot().unwrap(), saved.parameters);
        assert_eq!(
            restored_adam.snapshot().unwrap().first,
            saved.optimizer.first
        );
        assert_eq!(
            restored_adam.snapshot().unwrap().second,
            saved.optimizer.second
        );
        assert_eq!(restored_adam.step, saved.optimizer.step);
        assert_ne!(saved.rngs.action.get_seed(), saved.rngs.shuffle.get_seed());
        assert_ne!(
            saved.rngs.action.get_seed(),
            saved.rngs.environments[0].get_seed()
        );
        assert_eq!(saved.completed_rollouts, 2);
        assert!(saved.policy_version > 0);
        assert!(saved.optimizer.step > 0);
        let state =
            CurriculumState::decode(saved.curriculum_ron.as_deref(), &saved.config).unwrap();
        assert_eq!(state.activations, 2);
        assert!(
            state
                .last_search
                .as_ref()
                .is_some_and(|s| s.state.candidates > 0)
        );
        assert!(
            state
                .last_search
                .as_ref()
                .is_some_and(|s| s.iterations.iter().any(|i| i.candidates_evaluated > 0))
        );
        assert!(first.join("curriculum-000001/search.ron").is_file());
        assert!(first.join("curriculum-000002/search.ron").is_file());
        let exported: Vec<_> = std::fs::read_dir(first.join("curriculum-000002/tracks"))
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".generated.ron"))
            .collect();
        assert_eq!(exported.len(), state.active.len());
        for entry in exported {
            let track = GeneratedTrack::load(entry.path()).unwrap();
            assert!(state.active.contains(&track));
        }

        assert!(!first.join("checkpoint-000001.ron").exists());
        let mut next = c;
        next.total_rollouts = 3;
        let options = PersistenceOptions {
            save_every: std::num::NonZeroU64::new(100),
            save_first: Some(0),
            save_last: Some(0),
            no_save_rollouts: false,
            run_path_file: Some(root.join("receipt2")),
            ..Default::default()
        };
        super::super::execute_with_clock::<Cpu, _>(
            &next,
            Job::Train(Some(first.join("checkpoint-000002.ron")), options),
            &Default::default(),
            "integration CPU",
            session_budget::Monotonic(Instant::now()),
        )
        .unwrap();
        let second = PathBuf::from(
            std::fs::read_to_string(root.join("receipt2"))
                .unwrap()
                .trim(),
        );
        let resumed: Checkpoint = persistence::load(&second.join("checkpoint-000003.ron")).unwrap();
        resumed.validate().unwrap();
        let third_rollout: Rollout = persistence::load(&second.join("rollout-000003.ron")).unwrap();
        third_rollout.validate().unwrap();
        assert_eq!(third_rollout.policy_version, saved.policy_version);
        assert_ne!(second_rollout.policy_version, third_rollout.policy_version);
        assert_eq!(resumed.completed_rollouts, 3);
        assert!(resumed.optimizer.step > saved.optimizer.step);
        let state =
            CurriculumState::decode(resumed.curriculum_ron.as_deref(), &resumed.config).unwrap();
        assert_eq!(state.activations, 3);
        assert!(state.curriculum_rollouts > 0);
        assert!(state.selected_rollouts > 0);
        assert!(!state.used_selected_tracks.is_empty());
        assert_ne!(saved.parameters, resumed.parameters);
        assert_eq!(resumed.decisions - saved.decisions, 2);
        let search = SearchArchive::load(first.join("curriculum-000001/search.ron")).unwrap();
        assert!(
            !search.state.selected.is_empty(),
            "real ACO must select a reference-verified track"
        );
        assert!(search.iterations.iter().any(|i| {
            i.pheromones_after
                .iter()
                .flatten()
                .zip(i.pheromones_evaporated.iter().flatten())
                .any(|(a, b)| a > b)
        }));
        assert!(search.state.evaluation_steps <= search.config.max_evaluation_steps);
        let track_log = std::fs::read_to_string(first.join("track_metrics.csv")).unwrap();
        assert!(
            search
                .state
                .selected
                .iter()
                .any(|t| track_log.contains(&format!("2,\"{}\"", t.track.definition.id)))
        );
        println!(
            "ACO integration artifacts={} resume={} selected_rollouts={} Adam={} -> {} candidates={} selected={}",
            first.display(),
            second.display(),
            state.selected_rollouts,
            saved.optimizer.step,
            resumed.optimizer.step,
            search.state.candidates,
            search.state.selected.len()
        );
        for track in state
            .active
            .iter()
            .chain(&state.history)
            .chain(&state.fresh)
        {
            assert_eq!(
                track.definition.role,
                crate::simulation::TrackRole::Training
            );
        }
        // Small real artifacts retained in target for review.
    }
}

#[cfg(all(test, not(feature = "rl-gpu")))]
mod device_tests {
    use super::*;
    #[test]
    fn explicit_gpu_request_errors_without_cpu_fallback_or_run_creation() {
        let c = Config {
            device: DeviceChoice::Gpu { adapter: 999 },
            results_root: std::env::temp_dir()
                .join(format!("racing-no-gpu-{}", std::process::id()))
                .display()
                .to_string(),
            ..Default::default()
        };
        let error = super::super::dispatch(&c, super::super::Job::Collect).unwrap_err();
        assert!(error.to_string().contains("rl-gpu"));
        assert!(!Path::new(&c.results_root).exists());
    }
}
