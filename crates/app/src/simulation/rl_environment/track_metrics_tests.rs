use super::*;

fn fixture(config: &Config, track_count: usize) -> RacingEnvironment {
    let library = TrackLibrary::load_default().unwrap();
    let ids = library
        .training_tracks()
        .take(track_count)
        .map(|track| track.id.clone())
        .collect::<Vec<_>>();
    RacingEnvironment::new(
        0,
        scenes(&library, &ids, false).unwrap(),
        config,
        &SimulationConfig::default(),
        ChaCha12Rng::seed_from_u64(17),
    )
    .unwrap()
}

#[test]
fn decision_metrics_match_step_and_survive_reset_until_drained() {
    let config = Config {
        action_repeat: 5,
        timeout_ticks: 2,
        ..Default::default()
    };
    let mut env = fixture(&config, 1);
    let track = env.scenes[env.scene_index].track.definition.id.clone();
    let step = env.step([0., 1.]).unwrap();
    let metrics = &env.track_metrics[&track];
    assert_eq!(metrics.decisions, 1);
    assert_eq!(metrics.physical_ticks, 2);
    assert_eq!(metrics.reward_sum, f64::from(step.reward.total()));
    assert_eq!(metrics.progress_reward_sum, f64::from(step.reward.progress));
    assert!(
        (metrics.progress_gain * f64::from(config.reward.progress_weight)
            - metrics.progress_reward_sum)
            .abs()
            < 1e-5
    );
    assert_eq!(metrics.progress_fraction_sum, f64::from(step.progress));
    assert_eq!(metrics.progress_fraction_max, step.progress);
    assert_eq!(metrics.episodes_timeout, 1);
    assert_eq!(metrics.ended_episode_progress_sum, f64::from(step.progress));
    assert_eq!(metrics.collision_events, env.collision_events);
    assert!(env.step([0., 1.]).is_err());
    assert_eq!(env.track_metrics[&track].decisions, 1);

    env.reset().unwrap();
    assert_eq!(env.track_metrics[&track].decisions, 1);
    let second = env.step([0., 1.]).unwrap();
    let state = env.state;
    let events = env.collision_events;
    let drained = env.take_track_metrics();
    assert!(env.track_metrics.is_empty());
    assert_eq!(env.state, state);
    assert_eq!(env.collision_events, events);
    assert_eq!(drained[&track].decisions, 2);
    assert_eq!(drained[&track].physical_ticks, 4);
    assert_eq!(drained[&track].episodes_timeout, 2);
    assert_eq!(
        drained[&track].reward_sum,
        f64::from(step.reward.total()) + f64::from(second.reward.total())
    );
    env.reset().unwrap();
    env.step([0., 0.]).unwrap();
    assert_eq!(env.track_metrics[&track].decisions, 1);
    assert_eq!(env.track_metrics[&track].collision_events.physical_ticks, 2);
}

#[test]
fn terminal_metrics_stay_with_the_track_of_the_decision() {
    let config = Config {
        timeout_ticks: 1,
        ..Default::default()
    };
    let mut env = fixture(&config, 2);
    let mut expected = BTreeMap::<String, u64>::new();
    for _ in 0..32 {
        let track = env.scenes[env.scene_index].track.definition.id.clone();
        *expected.entry(track).or_default() += 1;
        let step = env.step([0., 0.]).unwrap();
        assert_eq!(step.reason, Some(EndReason::Timeout));
        env.reset().unwrap();
    }
    assert_eq!(expected.len(), 2);
    let metrics = env.take_track_metrics();
    assert_eq!(metrics.len(), expected.len());
    for (track, decisions) in expected {
        assert_eq!(metrics[&track].decisions, decisions);
        assert_eq!(metrics[&track].physical_ticks, decisions);
        assert_eq!(metrics[&track].episodes_timeout, decisions);
        assert_eq!(metrics[&track].episodes_completed, 0);
        assert_eq!(metrics[&track].episodes_collision, 0);
    }
}

#[test]
fn collision_completion_and_merge_preserve_all_accumulators() {
    let config = Config::default();
    let mut env = fixture(&config, 1);
    let track = env.scenes[0].track.definition.id.clone();
    env.state.position = env.scenes[0].track.left_border[0];
    let collision = env.step([0., 1.]).unwrap();
    assert_eq!(collision.reason, Some(EndReason::Collision));
    let collision_metrics = env.take_track_metrics().remove(&track).unwrap();
    assert_eq!(collision_metrics.episodes_collision, 1);
    assert_eq!(collision_metrics.collision_events.translation, 1);
    assert_eq!(
        collision_metrics.reward_sum,
        f64::from(collision.reward.total())
    );

    env.reset().unwrap();
    let length = env.scenes[0].track.total_length;
    env.progress.best = length - 0.1;
    env.progress.cumulative = length - 0.1;
    let completed = env.step([0., 1.]).unwrap();
    assert_eq!(completed.reason, Some(EndReason::Completed));
    let completed_metrics = env.take_track_metrics().remove(&track).unwrap();
    assert_eq!(completed_metrics.episodes_completed, 1);
    assert_eq!(completed_metrics.collision_events.translation, 0);

    let mut merged = collision_metrics.clone();
    merged.merge(&completed_metrics);
    assert_eq!(merged.decisions, 2);
    assert_eq!(merged.physical_ticks, 2);
    assert_eq!(merged.episodes_collision, 1);
    assert_eq!(merged.episodes_completed, 1);
    assert_eq!(merged.episodes_timeout, 0);
    assert_eq!(merged.collision_events.physical_ticks, 2);
    assert_eq!(merged.collision_events.translation, 1);
    assert_eq!(
        merged.reward_sum,
        collision_metrics.reward_sum + completed_metrics.reward_sum
    );
    assert_eq!(
        merged.progress_reward_sum,
        collision_metrics.progress_reward_sum + completed_metrics.progress_reward_sum
    );
    assert_eq!(
        merged.progress_gain,
        collision_metrics.progress_gain + completed_metrics.progress_gain
    );
    assert_eq!(
        merged.progress_fraction_sum,
        f64::from(collision.progress) + f64::from(completed.progress)
    );
    assert_eq!(merged.progress_fraction_max, completed.progress);
    assert_eq!(
        merged.ended_episode_progress_sum,
        f64::from(collision.progress) + f64::from(completed.progress)
    );
}
