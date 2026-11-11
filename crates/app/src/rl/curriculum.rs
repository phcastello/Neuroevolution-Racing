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
        }
        Ok(state)
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
        let n = options.ants * options.iterations;
        let mut search = SearchArchive::new(
            AcoConfig {
                generator: self.generator.clone(),
                ants: options.ants,
                iterations: options.iterations,
                max_candidates: n,
                max_selected: options.selected,
                max_evaluation_steps_per_candidate: options.evaluation_ticks as usize,
                max_evaluation_steps: options.evaluation_ticks * n as u64,
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
        persistence::save(
            &activation_dir.join("cost.ron"),
            &(
                completed,
                network.version(),
                started.elapsed().as_secs_f64(),
                search.state.candidates,
                search.state.evaluation_steps,
            ),
        )?;
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
    let mut reference_completed = false;
    let mut steps = 0;
    for _ in 0..per_pilot {
        let a = reference.reference_action();
        let s = reference.step(a).map_err(|e| e.to_string())?;
        steps += 1;
        if s.terminated || s.truncated {
            reference_completed = s.reason == Some(EndReason::Completed);
            break;
        }
    }
    let mut policy =
        env::RacingEnvironment::new(0, vec![scene], &config, p, ChaCha12Rng::seed_from_u64(seed))
            .map_err(|e| e.to_string())?;
    let mut progress = 0f64;
    let mut completed = false;
    let mut collision = false;
    for _ in 0..per_pilot {
        let observation = policy.observe().map_err(|e| e.to_string())?;
        let output = n
            .infer(matrix(&observation, 1, n.input_size, device).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let action = read(output.mean.tanh()).map_err(|e| e.to_string())?;
        let s = policy
            .step([action[0], action[1]])
            .map_err(|e| e.to_string())?;
        steps += 1;
        progress = progress.max(s.progress as f64);
        if s.terminated || s.truncated {
            completed = s.reason == Some(EndReason::Completed);
            collision = s.reason == Some(EndReason::Collision);
            break;
        }
    }
    Ok(PilotEvaluation {
        completion_rate: f64::from(completed),
        mean_progress: progress.clamp(0., 1.),
        collision_rate: f64::from(collision),
        reference_completed,
        simulation_steps: steps,
    })
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
    fn evaluator_freezes_parameters_and_bounds_work() {
        let c = Config {
            hidden: vec![8],
            ..Default::default()
        };
        let n = Network::<Cpu>::new(&c, &Default::default()).unwrap();
        let before = n.snapshot().unwrap();
        let t = procedural::generate(&Default::default(), 3).unwrap();
        let result =
            evaluate_candidate(&n, &c, &Default::default(), &Default::default(), &t, 7, 8).unwrap();
        assert!(result.simulation_steps <= 8);
        assert_eq!(before, n.snapshot().unwrap());
        assert!(!result.reference_completed);
    }
}

#[cfg(test)]
mod training_integration_tests {
    use super::super::{Job, persistence_policy::PersistenceOptions, session_budget};
    use super::*;
    #[test]
    fn training_updates_curriculum_only_between_rollouts_and_resumes_saved_state() {
        let root = std::env::temp_dir().join(format!(
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
            timeout_ticks: 8,
            results_root: root.join("runs").display().to_string(),
            training_tracks: vec!["interlagos".into()],
            curriculum: Some(CurriculumConfig {
                kind: CurriculumKind::Random,
                warmup: 1,
                interval: 1,
                ants: 4,
                iterations: 1,
                selected: 1,
                archive_limit: 2,
                evaluation_ticks: 8,
                ..Default::default()
            }),
            ..Default::default()
        };
        let c = super::super::resolve_config(c).unwrap();
        let options = PersistenceOptions {
            save_every: std::num::NonZeroU64::new(100),
            save_first: Some(0),
            save_last: Some(0),
            no_save_rollouts: true,
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
        assert_eq!(saved.completed_rollouts, 2);
        assert!(saved.policy_version > 0);
        assert!(saved.optimizer.step > 0);
        let state =
            CurriculumState::decode(saved.curriculum_ron.as_deref(), &saved.config).unwrap();
        assert_eq!(state.activations, 2);
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
            no_save_rollouts: true,
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
        assert_eq!(resumed.completed_rollouts, 3);
        assert!(resumed.optimizer.step > saved.optimizer.step);
        let state =
            CurriculumState::decode(resumed.curriculum_ron.as_deref(), &resumed.config).unwrap();
        assert_eq!(state.activations, 3);
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
        std::fs::remove_dir_all(root).unwrap();
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
