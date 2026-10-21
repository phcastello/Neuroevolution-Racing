use super::*;

#[test]
fn ga_reward_preset_only_changes_reward_and_output_root_from_b() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/rl");
    let b = Config::load(&root.join("ppo_update_b_lr.ron")).unwrap();
    let actual = Config::load(&root.join("ppo_reward_ga_inspired.ron")).unwrap();
    assert_eq!(actual.parameter_count(), 204_805);
    assert_eq!(actual.learning_rate, 0.0001);
    assert_eq!(actual.latent_entropy_coef, 0.);
    assert_eq!(actual.reward.progress_weight, 100.);
    assert_eq!(actual.reward.completion_bonus, 10.);
    assert_eq!(actual.reward.collision_penalty, 2.);
    assert_eq!(
        actual.reward.mode,
        racing_rl::config::RewardMode::GaInspired {
            speed_weight: 25.,
            progress_speed_half_saturation: 120.,
        }
    );
    assert_eq!(b.reward.mode, racing_rl::config::RewardMode::Baseline);
    let mut expected = b;
    expected.reward = actual.reward.clone();
    expected.results_root = "results_ppo/reward_ablation/ga_inspired".into();
    assert_eq!(actual, expected);
}

#[test]
fn controlled_presets_only_change_requested_variables_duration_and_output_root() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/rl");
    let base = Config::load(&root.join("large_200k.ron")).unwrap();
    for (name, rate, entropy, output) in [
        (
            "ppo_update_a_control.ron",
            0.0003,
            0.,
            "results_ppo/update_ablation/A_control",
        ),
        (
            "ppo_update_b_lr.ron",
            0.0001,
            0.,
            "results_ppo/update_ablation/B_lr",
        ),
        (
            "ppo_update_c_lr_entropy.ron",
            0.0001,
            0.001,
            "results_ppo/update_ablation/C_lr_entropy",
        ),
    ] {
        let actual = Config::load(&root.join(name)).unwrap();
        let mut expected = base.clone();
        expected.total_rollouts = 1_000;
        expected.learning_rate = rate;
        expected.latent_entropy_coef = entropy;
        expected.results_root = output.into();
        assert_eq!(
            actual, expected,
            "unexpected experimental difference: {name}"
        );
        actual.validate().unwrap();
        assert_eq!(actual.observation.dimension(), 25);
        assert_eq!(actual.hidden, [256, 256, 256, 256]);
        assert_eq!(actual.parameter_count(), 204_805);
        assert_eq!(actual.rollout_size(), 8_192);
        assert_eq!(
            actual.update_epochs * actual.rollout_size() / actual.minibatch_size,
            64
        );
        assert_eq!(actual.device, DeviceChoice::Gpu { adapter: 0 });
    }
}

#[test]
fn speed_horizon_presets_only_change_authorized_fields() {
    use racing_rl::config::SpeedNormalization;
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/rl");
    let base = Config::load(&root.join("ppo_reward_ga_inspired.ron")).unwrap();
    assert_eq!(
        base.observation.speed_normalization,
        SpeedNormalization::LinearClamp
    );
    for (name, gamma, lambda, folder) in [
        ("ppo_speed_d.ron", 0.99, 0.95, "D_speed"),
        ("ppo_speed_horizon_e.ron", 0.997, 0.98, "E_speed_horizon"),
    ] {
        let c = Config::load(&root.join(name)).unwrap();
        let mut expected = base.clone();
        expected.total_rollouts = 500;
        expected.observation.speed_normalization = SpeedNormalization::SignedAsymptotic;
        expected.gamma = gamma;
        expected.lambda = lambda;
        expected.results_root = format!("results_ppo/speed_horizon_ablation/{folder}");
        assert_eq!(c, expected, "{name}");
        assert_eq!(c.parameter_count(), 204_805);
        assert_eq!(c.observation.dimension(), 25);
        assert_eq!(c.observation.speed_scale, 250.);
        let lib = library().unwrap();
        let ids: std::collections::BTreeSet<_> =
            lib.training_tracks().map(|d| d.id.as_str()).collect();
        assert_eq!(
            ids,
            [
                "monza",
                "red_bull_ring",
                "interlagos",
                "catalunya",
                "silverstone"
            ]
            .into_iter()
            .collect()
        );
    }
}

#[test]
fn resume_requires_same_observation_semantics_but_allows_more_rollouts() {
    use racing_rl::config::SpeedNormalization;
    let old = Config::default();
    let mut new = old.clone();
    new.observation.speed_normalization = SpeedNormalization::SignedAsymptotic;
    assert_eq!(old.observation.dimension(), new.observation.dimension());
    assert!(validate_resume(&new, &old).is_err());
    assert!(validate_resume(&old, &new).is_err());
    let mut longer = new.clone();
    longer.total_rollouts += 500;
    validate_resume(&longer, &new).unwrap();
    longer.results_root = "a-new-branch".into();
    validate_resume(&longer, &new).unwrap();
    longer.observation.speed_scale += 1.;
    assert!(validate_resume(&longer, &new).is_err());
    let mut longer_old = old.clone();
    longer_old.total_rollouts += 500;
    validate_resume(&longer_old, &old).unwrap();
}

#[test]
fn continuation_preset_changes_only_accumulated_cap_and_output_destination() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/rl");
    let mut expected = Config::load(&root.join("ppo_speed_horizon_e.ron")).unwrap();
    let actual = Config::load(&root.join("ppo_e_continue_8h.ron")).unwrap();
    expected.total_rollouts = 4000;
    expected.results_root = "results_ppo/e_continuation_8h/runs".into();
    assert_eq!(actual, expected);
    assert_eq!(actual.parameter_count(), 204_805);
}

#[test]
#[ignore = "explicit read-only compatibility check of the chosen historical E200 source"]
fn validate_prepared_e_source_without_training_or_evaluation() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let requested =
        resolve_config(Config::load(&root.join("configs/rl/ppo_e_continue_8h.ron")).unwrap())
            .unwrap();
    let path=root.join("results_ppo/speed_horizon_ablation/E_speed_horizon/ppo-1792524131440514300-32024/checkpoint-000200.ron");
    let saved: Checkpoint = persistence::load(&path).unwrap();
    saved.validate().unwrap();
    validate_resume(&requested, &saved.config).unwrap();
    let (network, optimizer, _) = restore::<Cpu>(&saved, &Default::default()).unwrap();
    assert_eq!(network.snapshot().unwrap(), saved.parameters);
    assert_eq!(optimizer.step, 12781);
    assert_eq!(optimizer.snapshot().unwrap().first, saved.optimizer.first);
    assert_eq!(optimizer.snapshot().unwrap().second, saved.optimizer.second);
    assert_eq!(saved.completed_rollouts, 200);
    assert_eq!(requested.total_rollouts, 4000);
    assert_eq!(requested.parameter_count(), 204805);
}

#[test]
fn bounded_true_resume_saves_off_interval_final_and_expired_session_preserves_state() {
    use std::{cell::Cell, num::NonZeroU64};
    struct Fake {
        calls: Cell<usize>,
        expired: bool,
    }
    impl session_budget::Clock for Fake {
        fn seconds(&self) -> f64 {
            let calls = self.calls.get();
            self.calls.set(calls + 1);
            if self.expired || calls >= 2 { 950. } else { 0. }
        }
    }
    let parent = std::env::temp_dir().join("ppo-bounded-fixtures");
    let root = persistence::run_directory(&parent).unwrap();
    let mut c = Config {
        hidden: vec![4],
        num_envs: 1,
        rollout_length: 2,
        minibatch_size: 2,
        update_epochs: 1,
        total_rollouts: 2,
        timeout_ticks: 4,
        training_tracks: vec!["monza".into()],
        results_root: root.join("source").to_string_lossy().into_owned(),
        ..Config::default()
    };
    let receipt = root.join("source.txt");
    let options = PersistenceOptions {
        no_save_rollouts: true,
        save_every: NonZeroU64::new(1),
        save_first: Some(0),
        save_last: Some(0),
        run_path_file: Some(receipt.clone()),
        ..Default::default()
    };
    execute::<Cpu>(
        &c,
        Job::Train(None, options),
        &Default::default(),
        "tiny source fixture",
    )
    .unwrap();
    let source_dir = PathBuf::from(std::fs::read_to_string(receipt).unwrap().trim());
    let source_path = source_dir.join("checkpoint-000002.ron");
    let source_bytes = std::fs::read(&source_path).unwrap();
    let source: Checkpoint = persistence::load(&source_path).unwrap();
    assert_eq!(source.optimizer.step, 2);
    c.total_rollouts = 9;
    c.results_root = root.join("branches").to_string_lossy().into_owned();
    for expired in [false, true] {
        let receipt = root.join(format!("resume-{expired}.txt"));
        let options = PersistenceOptions {
            no_save_rollouts: true,
            save_every: NonZeroU64::new(100),
            save_first: Some(0),
            save_last: Some(0),
            evaluate_every: NonZeroU64::new(100),
            run_path_file: Some(receipt.clone()),
            wall_time_seconds: Some(if expired { 0. } else { 1000. }),
            ..Default::default()
        };
        validate_resume(&c, &source.config).unwrap();
        execute_with_clock::<Cpu, _>(
            &c,
            Job::Train(Some(source_path.clone()), options),
            &Default::default(),
            "tiny bounded fixture",
            Fake {
                calls: Cell::new(0),
                expired,
            },
        )
        .unwrap();
        let dir = PathBuf::from(std::fs::read_to_string(receipt).unwrap().trim());
        let n = if expired { 2 } else { 3 };
        let saved: Checkpoint =
            persistence::load(&dir.join(format!("checkpoint-{n:06}.ron"))).unwrap();
        saved.validate().unwrap();
        assert_eq!(saved.completed_rollouts, n);
        assert_eq!(saved.optimizer.step, n);
        assert_eq!(
            saved.decisions,
            source.decisions + if expired { 0 } else { 2 }
        );
        if expired {
            assert_eq!(saved.parameters, source.parameters);
            assert_eq!(saved.optimizer.first, source.optimizer.first);
            assert_eq!(saved.optimizer.second, source.optimizer.second);
            assert_eq!(
                ron::to_string(&saved.rngs).unwrap(),
                ron::to_string(&source.rngs).unwrap()
            );
        } else {
            assert!(dir.join("losses-000003.ron").is_file());
        }
        let state = std::fs::read_to_string(dir.join("continuation_state.csv")).unwrap();
        assert!(state.contains("wall_time_budget"));
        assert!(state.contains(",false,wall_time_budget,"));
        assert!(!dir.join("rollout-000003.ron").exists());
        assert_eq!(std::fs::read(&source_path).unwrap(), source_bytes);
        assert!(persistence::save(&source_path, &saved).is_err());
        assert_eq!(std::fs::read(&source_path).unwrap(), source_bytes);
    }
    struct Plenty;
    impl session_budget::Clock for Plenty {
        fn seconds(&self) -> f64 {
            0.
        }
    }
    c.total_rollouts = 3;
    let receipt = root.join("cap.txt");
    let options = PersistenceOptions {
        no_save_rollouts: true,
        save_every: NonZeroU64::new(100),
        save_first: Some(0),
        save_last: Some(0),
        evaluate_every: NonZeroU64::new(100),
        run_path_file: Some(receipt.clone()),
        wall_time_seconds: Some(1000.),
        ..Default::default()
    };
    execute_with_clock::<Cpu, _>(
        &c,
        Job::Train(Some(source_path.clone()), options),
        &Default::default(),
        "tiny rollout cap fixture",
        Plenty,
    )
    .unwrap();
    let dir = PathBuf::from(std::fs::read_to_string(receipt).unwrap().trim());
    let state = std::fs::read_to_string(dir.join("continuation_state.csv")).unwrap();
    assert!(state.contains(",true,rollout_limit,"));
    assert!(dir.join("checkpoint-000003.ron").is_file());
    let evaluations = std::fs::read_to_string(dir.join("periodic_evaluations.csv")).unwrap();
    assert_eq!(evaluations.lines().count(), 3); // exactly one source + one final episode
    let resolved = std::fs::canonicalize(&root).unwrap();
    assert_eq!(
        resolved.parent(),
        Some(std::fs::canonicalize(parent).unwrap().as_path())
    );
    std::fs::remove_dir_all(resolved).unwrap();
}

#[test]
fn tiny_orchestration_persists_exact_receipts_checkpoints_losses_and_separate_evaluations() {
    use std::num::NonZeroU64;
    let root = persistence::run_directory(&std::env::temp_dir().join("ppo-orchestration-fixtures"))
        .unwrap();
    let c = Config {
        hidden: vec![4],
        num_envs: 1,
        rollout_length: 2,
        minibatch_size: 2,
        update_epochs: 1,
        total_rollouts: 2,
        timeout_ticks: 4,
        training_tracks: vec!["monza".into()],
        results_root: root.join("runs").to_string_lossy().into_owned(),
        ..Config::default()
    };
    let mut paths = Vec::new();
    for index in 0..2 {
        let receipt = root.join(format!("receipt-{index}.txt"));
        let options = PersistenceOptions {
            save_every: NonZeroU64::new(1),
            save_first: Some(0),
            save_last: Some(0),
            no_save_rollouts: true,
            evaluate_every: NonZeroU64::new(1),
            experiment: format!("fixture-{index}"),
            run_path_file: Some(receipt.clone()),
            ..Default::default()
        };
        options.validate_evaluation(c.total_rollouts).unwrap();
        execute::<Cpu>(
            &c,
            Job::Train(None, options),
            &Default::default(),
            "synthetic CPU fixture",
        )
        .unwrap();
        let dir = PathBuf::from(std::fs::read_to_string(receipt).unwrap().trim());
        assert!(dir.starts_with(std::fs::canonicalize(&root).unwrap()));
        assert!(!paths.contains(&dir));
        for rollout in [1, 2] {
            assert!(dir.join(format!("checkpoint-{rollout:06}.ron")).is_file());
            assert!(dir.join(format!("losses-{rollout:06}.ron")).is_file());
            assert!(!dir.join(format!("rollout-{rollout:06}.ron")).exists());
        }
        for name in [
            "metrics.csv",
            "update_metrics.csv",
            "periodic_evaluations.csv",
        ] {
            let text = std::fs::read_to_string(dir.join(name)).unwrap();
            assert_eq!(text.lines().count(), 3, "{name}");
        }
        let log = std::fs::read_to_string(dir.join("run.log")).unwrap();
        assert_eq!(log.matches("evaluation_wall_seconds=").count(), 2);
        assert!(dir.join("track_metrics.csv").is_file());
        let saved: Checkpoint = persistence::load(&dir.join("checkpoint-000002.ron")).unwrap();
        saved.validate().unwrap();
        assert_eq!(saved.completed_rollouts, 2);
        paths.push(dir);
    }
    // Only remove this newly created fixture, verified inside its dedicated temp root.
    let resolved = std::fs::canonicalize(&root).unwrap();
    let parent =
        std::fs::canonicalize(std::env::temp_dir().join("ppo-orchestration-fixtures")).unwrap();
    assert_eq!(resolved.parent(), Some(parent.as_path()));
    std::fs::remove_dir_all(resolved).unwrap();
}
