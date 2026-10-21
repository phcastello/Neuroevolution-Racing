use super::*;
use clap::Parser;

#[test]
fn best_selection_includes_source_and_never_replaces_an_exact_tie_with_a_newer_policy() {
    let source = Score {
        path: "source".into(),
        rollout: 200,
        adam_step: 100,
        completed: 4,
        mean_progress: 0.98,
    };
    let mut newer = source.clone();
    newer.rollout = 300;
    newer.path = "new".into();
    assert!(!newer.better_than(&source));
    newer.mean_progress = 0.99;
    assert!(newer.better_than(&source));
    newer.completed = 3;
    assert!(!newer.better_than(&source));
    newer.completed = 5;
    newer.mean_progress = 0.1;
    assert!(newer.better_than(&source));
}

#[derive(Parser)]
struct TestArgs {
    #[command(flatten)]
    options: EvaluationOptions,
}

fn options(arguments: &[&str]) -> EvaluationOptions {
    TestArgs::try_parse_from(std::iter::once("evaluation").chain(arguments.iter().copied()))
        .unwrap()
        .options
}

#[test]
fn evaluation_defaults_remain_deterministic_and_multiple_episodes_per_track() {
    let o = options(&[]);
    assert_eq!(o.mode, EvaluationMode::Deterministic);
    assert_eq!(o.tracks, EvaluationTracks::Validation);
    assert_eq!(o.episodes, 4);
    o.validate().unwrap();
    assert!(options(&["--episodes", "0"]).validate().is_err());
    let o = options(&[
        "--mode",
        "stochastic",
        "--episodes",
        "12",
        "--tracks",
        "both",
        "--action-seed",
        "123",
        "--environment-seed",
        "456",
    ]);
    assert_eq!(o.mode, EvaluationMode::Stochastic);
    assert_eq!(o.episodes, 12);
    assert_eq!(o.action_seed, Some(123));
    assert_eq!(o.environment_seed, Some(456));
}

#[test]
fn evaluation_track_selection_never_mutates_learning_configuration() {
    let lib = library().unwrap();
    let training = lib.training_tracks().next().unwrap().id.clone();
    let validation = lib.validation_tracks().next().unwrap().id.clone();
    let c = Config {
        training_tracks: vec![training.clone()],
        evaluation_tracks: vec![validation.clone()],
        ..Config::default()
    };
    let original = c.clone();
    assert_eq!(
        options(&[]).selected_tracks(&c, &lib).unwrap(),
        [validation.clone()]
    );
    assert_eq!(
        options(&["--tracks", "training"])
            .selected_tracks(&c, &lib)
            .unwrap(),
        [training.clone()]
    );
    assert_eq!(
        options(&["--tracks", "both"])
            .selected_tracks(&c, &lib)
            .unwrap(),
        [training.clone(), validation.clone()]
    );
    assert_eq!(
        options(&["--track", &validation, "--track", &training])
            .selected_tracks(&c, &lib)
            .unwrap(),
        [validation, training]
    );
    assert!(
        options(&["--track", "missing-track"])
            .selected_tracks(&c, &lib)
            .is_err()
    );
    assert_eq!(c, original);
}

#[test]
fn action_seeds_are_reproducible_and_independent_by_track_episode_and_domain() {
    let seed = episode_seed(55, "interlagos", 3, ACTION_DOMAIN);
    assert_eq!(seed, episode_seed(55, "interlagos", 3, ACTION_DOMAIN));
    for other in [
        environment_seed(55, 3),
        episode_seed(55, "interlagos", 3, ACTION_DOMAIN ^ 1),
        episode_seed(55, "interlagos", 4, ACTION_DOMAIN),
        episode_seed(55, "monaco", 3, ACTION_DOMAIN),
        episode_seed(56, "interlagos", 3, ACTION_DOMAIN),
    ] {
        assert_ne!(seed, other);
    }
    assert_eq!(environment_seed(55, 3), 58);
    assert_eq!(environment_seed(u64::MAX, 1), 0);
    // A selected track keeps exactly the same stream when the suite is reordered.
    let ids = ["interlagos", "monaco"];
    let forward: Vec<_> = ids
        .iter()
        .map(|id| (id, episode_seed(55, id, 3, ACTION_DOMAIN)))
        .collect();
    let mut reverse: Vec<_> = ids
        .iter()
        .rev()
        .map(|id| (id, episode_seed(55, id, 3, ACTION_DOMAIN)))
        .collect();
    reverse.reverse();
    assert_eq!(forward, reverse);
}

#[test]
fn stochastic_actions_use_saved_gaussian_tanh_and_leave_policy_unchanged() {
    // Synthetic small policy and observations only: no checkpoint or episode is run.
    let c = Config {
        hidden: vec![4],
        ..Config::default()
    };
    let device = Default::default();
    let n = Network::<Cpu>::new(&c, &device).unwrap();
    let original = n.snapshot().unwrap();
    let version = n.version();
    let observation = vec![0.25; c.observation.dimension()];
    let seed = episode_seed(123, "synthetic", 0, ACTION_DOMAIN);
    let mut first = ChaCha12Rng::seed_from_u64(seed);
    let mut second = first.clone();
    let mut expected_rng = first.clone();
    let mut other = ChaCha12Rng::seed_from_u64(episode_seed(123, "synthetic", 1, ACTION_DOMAIN));
    let mut different = false;
    for _ in 0..16 {
        let a = action(
            &n,
            &observation,
            EvaluationMode::Stochastic,
            &mut first,
            &device,
        )
        .unwrap();
        let b = action(
            &n,
            &observation,
            EvaluationMode::Stochastic,
            &mut second,
            &device,
        )
        .unwrap();
        let mean = n
            .infer(matrix(&observation, 1, n.input_size, &device).unwrap())
            .unwrap()
            .mean;
        let expected = read(
            distribution::sample(mean, n.log_std.clone(), &mut expected_rng)
                .unwrap()
                .action,
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(a, [expected[0], expected[1]]);
        assert!(a.iter().all(|x| x.is_finite() && (-1.0..=1.0).contains(x)));
        different |= a
            != action(
                &n,
                &observation,
                EvaluationMode::Stochastic,
                &mut other,
                &device,
            )
            .unwrap();
    }
    assert!(different);
    let before = first.get_word_pos();
    let a = action(
        &n,
        &observation,
        EvaluationMode::Deterministic,
        &mut first,
        &device,
    )
    .unwrap();
    let expected = read(
        n.infer(matrix(&observation, 1, n.input_size, &device).unwrap())
            .unwrap()
            .mean
            .tanh(),
    )
    .unwrap();
    assert_eq!(a, [expected[0], expected[1]]);
    assert_eq!(first.get_word_pos(), before);
    assert_eq!(n.snapshot().unwrap(), original);
    assert_eq!(n.version(), version);
}

#[test]
fn track_summary_counts_episode_outcomes_progress_and_reward() {
    let mut summary = TrackSummary::default();
    for (i, (completed, collision, timeout, progress, reward)) in [
        (true, false, false, 1.0, 8.0),
        (false, true, false, 0.5, -2.0),
        (false, false, true, 0.0, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        summary.record(&EvaluationEpisode {
            track: "synthetic".into(),
            track_role: "Training".into(),
            episode: i,
            mode: EvaluationMode::Stochastic,
            seed: i as u64,
            action_seed: 100 + i as u64,
            reward,
            completed,
            collision,
            timeout,
            progress,
            decisions: 2,
            physical_ticks: 3,
            collision_events: CollisionEvents {
                physical_ticks: 3,
                translation: u64::from(collision),
                ..CollisionEvents::default()
            },
            seconds: 0.,
            simulated_seconds: 0.,
            trained: false,
        });
    }
    assert_eq!(
        (
            summary.episodes,
            summary.completed,
            summary.collision,
            summary.timeout
        ),
        (3, 1, 1, 1)
    );
    assert_eq!(summary.reward_sum, 6.);
    assert_eq!(summary.reward_mean, 2.);
    assert_eq!(summary.progress_mean, 0.5);
    assert_eq!(summary.progress_max, 1.);
    assert_eq!(summary.decisions, 6);
    assert_eq!(summary.physical_ticks, 9);
    assert_eq!(summary.collision_events.translation, 1);
}

#[test]
fn periodic_evaluation_leaves_training_state_rngs_and_next_trajectory_identical() {
    use racing_rl::config::SpeedNormalization;
    let mut c = Config {
        hidden: vec![4],
        num_envs: 1,
        rollout_length: 2,
        minibatch_size: 2,
        update_epochs: 1,
        total_rollouts: 2,
        timeout_ticks: 5,
        training_tracks: vec!["monza".into()],
        ..Config::default()
    };
    c.observation.speed_normalization = SpeedNormalization::SignedAsymptotic;
    let device = Default::default();
    let p = physics();
    let lib = library().unwrap();
    let mut rngs = Rngs::new(&c);
    let mut n = Network::<Cpu>::new(&c, &device).unwrap();
    let mut o = Optimizer::new(c.optimizer.clone(), &n.parameters());
    let scenes = env::scenes(&lib, &c.training_tracks, false).unwrap();
    let mut training = env::environments(scenes.clone(), &c, &p, &rngs.environments).unwrap();
    let mut control = env::environments(scenes, &c, &p, &rngs.environments).unwrap();
    let mut control_rngs = rngs.clone();
    let first = rollout::collect(&n, &mut training, &c, &mut rngs.action, &device).unwrap();
    let control_first =
        rollout::collect(&n, &mut control, &c, &mut control_rngs.action, &device).unwrap();
    assert_eq!(
        ron::to_string(&first.transitions).unwrap(),
        ron::to_string(&control_first.transitions).unwrap()
    );
    racing_rl::trainer::update(&mut n, &mut o, &first, &c, &mut rngs.shuffle, &device).unwrap();
    rngs.environments = training.iter().map(|e| e.rng.clone()).collect();
    let snapshot = || {
        checkpoint(
            &n,
            &o,
            SaveContext {
                config: &c,
                physics: &p,
                rngs: &rngs,
                backend: "synthetic CPU",
                completed_rollouts: 1,
                decisions: 2,
                physical_ticks: 2,
            },
        )
        .unwrap()
    };
    let saved = snapshot();
    let before = ron::to_string(&saved).unwrap();
    let dir =
        persistence::run_directory(&std::env::temp_dir().join("ppo-periodic-fixtures")).unwrap();
    let path = dir.join("checkpoint-000001.ron");
    persistence::save(&path, &saved).unwrap();
    let mut rows = Vec::new();
    write_periodic_header(&mut rows).unwrap();
    periodic(&saved, &path, "fixture", &mut rows).unwrap();
    assert_eq!(ron::to_string(&saved).unwrap(), before);
    assert_eq!(ron::to_string(&snapshot()).unwrap(), before);
    assert_eq!(rows.iter().filter(|b| **b == b'\n').count(), 2);
    let csv = String::from_utf8(rows).unwrap();
    assert!(csv.contains("simulated_seconds,evaluation_wall_seconds"));
    // Same updated policy acts on the uninterrupted control environments. The
    // comparison catches resets, evaluation transitions, shared RNGs or mutations.
    let next = rollout::collect(&n, &mut training, &c, &mut rngs.action, &device).unwrap();
    let expected =
        rollout::collect(&n, &mut control, &c, &mut control_rngs.action, &device).unwrap();
    assert_eq!(
        ron::to_string(&next.transitions).unwrap(),
        ron::to_string(&expected.transitions).unwrap()
    );
    assert_eq!(
        rngs.action.get_word_pos(),
        control_rngs.action.get_word_pos()
    );
    for (a, b) in training.iter_mut().zip(&mut control) {
        assert_eq!(a.rng.get_word_pos(), b.rng.get_word_pos());
        assert_eq!(a.observe().unwrap(), b.observe().unwrap());
    }
    std::fs::remove_file(&path).unwrap();
    std::fs::remove_dir(&dir).unwrap();
}
