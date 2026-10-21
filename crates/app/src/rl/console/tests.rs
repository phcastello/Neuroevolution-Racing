use super::*;
use racing_rl::{
    config::Config,
    ppo::LossMetrics,
    rollout::{EndReason, RewardParts, Timings, Transition},
    trainer::UpdateMetrics,
};

fn fixture() -> (Rollout, UpdateTelemetry) {
    let config = Config {
        num_envs: 2,
        rollout_length: 2,
        minibatch_size: 2,
        update_epochs: 2,
        ..Default::default()
    };
    let transitions = [
        (0, 10, Some(EndReason::Completed), 1.25, 10., 0.),
        (1, 20, None, 2.5, 0., 0.),
        (0, 11, Some(EndReason::Collision), 0., 0., -2.),
        (1, 20, Some(EndReason::Timeout), 0., 0., 0.),
    ]
    .into_iter()
    .enumerate()
    .map(
        |(i, (env_id, episode_id, reason, progress, completion, collision))| Transition {
            env_id,
            episode_id,
            observation: vec![],
            action: [0.; 2],
            u: [0.; 2],
            old_log_prob: 0.,
            old_value: 0.,
            next_observation: vec![],
            next_value: 0.,
            reward: RewardParts {
                progress,
                completion,
                collision,
            },
            terminated: matches!(reason, Some(EndReason::Completed | EndReason::Collision)),
            truncated: reason == Some(EndReason::Timeout),
            reason,
            physical_ticks: i as u64 + 1,
            progress: 0.,
        },
    )
    .collect();
    let rollout = Rollout {
        version: 1,
        policy_version: 99,
        num_envs: 2,
        rollout_length: 2,
        observation_spec: config.observation.clone(),
        transitions,
        raw_advantages: vec![],
        policy_advantages: vec![],
        value_targets: vec![],
        timings: Timings {
            observations_seconds: 0.002,
            upload_seconds: 0.041,
            inference_seconds: 2.02,
            download_seconds: 0.311,
            simulation_seconds: 0.168,
            preparation_seconds: 0.0001,
            total_seconds: 2.6,
        },
    };
    let metrics: Vec<_> = [false, true]
        .into_iter()
        .map(|stop| UpdateMetrics {
            loss: LossMetrics {
                policy_loss: -0.125,
                value_loss: 0.5,
                latent_entropy: 1.,
                total_loss: 0.125,
                mean_ratio: 1.,
                clip_fraction: 0.25,
                approximate_kl: if stop { 0.04 } else { 0.01 },
            },
            gradient_norm_before_clip: (!stop).then_some(1.25),
            optimizer_step: 101,
            stopped_by_kl: stop,
        })
        .collect();
    let update = UpdateTelemetry::from_metrics(&metrics, &config, 100, 101, [-0.5; 2]).unwrap();
    (rollout, update)
}

fn value<'a>(output: &'a str, label: &str) -> &'a str {
    let prefix = format!("   {label}:");
    output
        .lines()
        .find(|line| line.starts_with(&prefix))
        .unwrap_or_else(|| panic!("missing metric: {label}"))
        .get(37..)
        .unwrap()
        .trim_end()
}

#[test]
fn blocks_preserve_every_legacy_rollout_metric_and_all_timing_fields() {
    let (rollout, update) = fixture();
    let text = training_rollout(500, 1000, &rollout, &update, 3.2);
    assert!(text.contains("PPO TRAINING | Rollout 500 / 1000 (50.0%)"));
    for (label, expected) in [
        ("Observed", "3"),
        ("Terminated", "2"),
        ("Truncated", "1"),
        ("Completed", "1"),
        ("Collisions", "1"),
        ("Timeouts", "1"),
        ("Total", "+11.7500"),
        ("Progress", "+3.7500"),
        ("Completion", "+10.0000"),
        ("Collision", "-2.0000"),
        ("Transitions", "4"),
        ("Decisions", "4"),
        ("Physical ticks", "10"),
        ("Rendered frames", "0"),
        ("Collection throughput", "1.54 transitions/s"),
        ("Observation", "0.0020s"),
        ("Upload", "0.0410s"),
        ("Inference", "2.0200s"),
        ("Download", "0.3110s"),
        ("Simulation", "0.1680s"),
        ("Preparation", "0.0001s"),
        ("Collection total", "2.6000s"),
        ("PPO update total", "3.2000s"),
        ("Overall (collection + update)", "5.8000s"),
    ] {
        assert_eq!(value(&text, label), expected, "{label}");
    }
    // No existing PPO diagnostic row is filtered out by the console layout.
    for (label, expected) in update.console_lines() {
        assert_eq!(value(&text, label), expected, "{label}");
    }
}

#[test]
fn blocks_are_ascii_aligned_and_at_most_eighty_columns() {
    let (rollout, update) = fixture();
    for (number, target) in [(1, 1000), (500, 1000), (1000, 1000), (20_000, 20_000)] {
        let text = training_rollout(number, target, &rollout, &update, 3.2);
        assert!(text.is_ascii());
        assert!(text.lines().all(|line| line.len() <= 80), "{text}");
        assert!(text.starts_with(&format!("\n{DIVIDER}\n PPO TRAINING")));
        assert!(text.ends_with(&format!("{FOOTER}\n")));
        for title in ["Episodes", "Rewards", "PPO Update", "Simulation", "Timings"] {
            assert_eq!(
                text.lines()
                    .filter(|line| *line == format!(" {title}"))
                    .count(),
                1
            );
        }
    }
}

#[test]
fn zero_duration_and_missing_diagnostics_are_explicit_and_never_nan() {
    let (mut rollout, _) = fixture();
    rollout.transitions.clear();
    rollout.timings = Timings::default();
    let update = UpdateTelemetry::from_metrics(&[], &Config::default(), 0, 0, [-0.5; 2]).unwrap();
    let text = training_rollout(1, 1, &rollout, &update, 0.);
    assert_eq!(value(&text, "Collection throughput"), "N/A transitions/s");
    assert_eq!(value(&text, "Observed"), "0");
    assert_eq!(value(&text, "Total"), "+0.0000");
    assert!(!text.contains("NaN"));
    assert!(!text.contains("inf"));
}

#[test]
fn presentation_does_not_modify_rollout_or_legacy_log_summary() {
    let (mut rollout, update) = fixture();
    rollout.timings.observations_seconds = 0.0020123456789;
    let serialized = ron::to_string(&rollout).unwrap();
    let legacy = super::super::summary(&rollout);
    let csv_before = {
        let mut bytes = vec![];
        update.write_row(500, &mut bytes).unwrap();
        bytes
    };
    let report = training_rollout(500, 1000, &rollout, &update, 3.2);
    assert_eq!(ron::to_string(&rollout).unwrap(), serialized);
    assert_eq!(super::super::summary(&rollout), legacy);
    assert!(legacy.contains("0.0020123456789"));
    assert!(!report.contains("0.0020123456789"));
    let mut csv_after = vec![];
    update.write_row(500, &mut csv_after).unwrap();
    assert_eq!(csv_before, csv_after);
}
