use super::*;
use racing_rl::config::{RewardConfig, RewardMode};

fn config() -> Config {
    Config {
        num_envs: 1,
        rollout_length: 4,
        minibatch_size: 4,
        hidden: vec![8],
        reward: RewardConfig {
            mode: RewardMode::GaInspired {
                speed_weight: 25.,
                progress_speed_half_saturation: 120.,
            },
            progress_weight: 100.,
            completion_bonus: 10.,
            collision_penalty: 2.,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn environment(config: &Config) -> RacingEnvironment {
    let library = TrackLibrary::load_default().unwrap();
    let scenes = scenes(&library, &["interlagos".into()], false).unwrap();
    RacingEnvironment::new(
        0,
        scenes,
        config,
        &SimulationConfig::default(),
        ChaCha12Rng::seed_from_u64(7),
    )
    .unwrap()
}

#[test]
fn ga_reward_zero_for_idle_reverse_oscillation_and_rejected_jumps() {
    let c = config();
    let mut p = Progress {
        previous: 10.,
        ..Default::default()
    };
    for projected in [10., 9., 10.] {
        let gain = p.advance(projected, 100., 1.);
        assert_eq!(
            reward::progress_reward(&c.reward, gain, 100., c.physical_dt),
            0.
        );
    }
    let gain = p.advance(12., 100., 2.);
    assert!(reward::progress_reward(&c.reward, gain, 100., c.physical_dt) > 0.);
    for projected in [10., 12., 10., 12., 50., 51., 10., 11., 50., 51.] {
        let gain = p.advance(projected, 100., 1.);
        assert_eq!(
            reward::progress_reward(&c.reward, gain, 100., c.physical_dt),
            0.
        );
    }
    // Actual environment transitions: no raw vehicle speed can bypass the gain gate.
    let mut e = environment(&c);
    e.state.speed = 0.;
    assert_eq!(e.step([0., 0.]).unwrap().reward.total(), 0.);
    e.state.speed = -45.;
    for _ in 0..3 {
        assert_eq!(e.step([0., -0.5]).unwrap().reward.total(), 0.);
    }
}

#[test]
fn ga_reward_normalizes_lap_fraction_and_bounds_the_speed_contribution() {
    let mut c = config();
    // Same useful speed (120 units/s) gives the same speed factor; doubling L
    // halves reward per unit distance. This is not a raw-distance reward.
    let short = reward::progress_reward(&c.reward, 2., 6_000., c.physical_dt);
    let long = reward::progress_reward(&c.reward, 2., 12_000., c.physical_dt);
    assert_eq!(short, 2. * long);
    assert!((short - 112.5 * 2. / 6_000.).abs() < 1e-7);
    // Speed is physical units/s as in GA, so isolate the progress component when
    // checking identical fractions at different physical speeds/track lengths.
    c.reward.mode = RewardMode::GaInspired {
        speed_weight: 0.,
        progress_speed_half_saturation: 120.,
    };
    assert_eq!(
        reward::progress_reward(&c.reward, 2., 6_000., c.physical_dt),
        reward::progress_reward(&c.reward, 4., 12_000., c.physical_dt),
    );
    c = config();
    for gain in [0.001, 1., 2., 10., 100.] {
        let r = reward::progress_reward(&c.reward, gain, 6_000., c.physical_dt);
        assert!(r >= 100. * gain / 6_000.);
        assert!(r < 125. * gain / 6_000.);
    }
    // Accumulated dense reward is bounded by 125 per lap, independent of L.
    for length in [6_000., 12_000.] {
        let total =
            1000. * reward::progress_reward(&c.reward, length / 1000., length, c.physical_dt);
        assert!((100. ..125.).contains(&total));
    }
}

#[test]
fn ga_reward_faster_useful_progress_has_larger_discounted_return() {
    let c = config();
    let discounted_lap = |ticks: usize, wait: usize| -> f64 {
        let r = reward::progress_reward(&c.reward, 6_000. / ticks as f32, 6_000., c.physical_dt);
        let dense: f64 = (0..ticks)
            .map(|t| f64::from(c.gamma).powi((t + wait) as i32) * f64::from(r))
            .sum();
        dense
            + f64::from(c.gamma).powi((ticks + wait - 1) as i32)
                * f64::from(c.reward.completion_bonus)
    };
    assert!(discounted_lap(3_000, 0) > discounted_lap(6_000, 0));
    assert!(discounted_lap(3_000, 0) > discounted_lap(3_000, 60));
    let typical = reward::progress_reward(&c.reward, 2., 6_000., c.physical_dt);
    assert!((typical - 0.0375).abs() < 1e-7);
    assert!((typical / (1. - c.gamma) - 3.75).abs() < 1e-4);
}

#[test]
fn ga_reward_completion_once_collision_penalty_and_reset() {
    let c = config();
    let mut e = environment(&c);
    e.state.position = e.scenes[0].track.left_border[0];
    let hit = e.step([0., 1.]).unwrap();
    assert_eq!(hit.reason, Some(EndReason::Collision));
    assert_eq!(hit.reward.collision, -2.);
    assert_eq!(hit.reward.completion, 0.);
    assert!(e.step([0., 1.]).is_err());
    e.reset().unwrap();
    assert_eq!(e.progress.best, 0.);
    let length = e.scenes[0].track.total_length;
    e.progress.best = length - 0.1;
    e.progress.cumulative = length - 0.1;
    let completed = e.step([0., 1.]).unwrap();
    assert_eq!(completed.reason, Some(EndReason::Completed));
    assert_eq!(completed.reward.completion, 10.);
    assert_eq!(completed.reward.collision, 0.);
    assert!(e.step([0., 1.]).is_err());
    e.reset().unwrap();
    e.state.speed = 0.;
    assert_eq!(e.step([0., 0.]).unwrap().reward.total(), 0.);
}

#[test]
fn baseline_reward_keeps_original_arithmetic_and_serialized_shape() {
    let c = RewardConfig::default();
    for gain in [0., 0.01, 2., 100.] {
        assert_eq!(
            reward::progress_reward(&c, gain, 6_000., 1. / 60.),
            gain * 0.01
        );
    }
    let legacy = "(version:1,progress_weight:0.01,completion_bonus:10.0,collision_penalty:1.0)";
    let restored: RewardConfig = ron::from_str(legacy).unwrap();
    assert_eq!(restored, c);
    assert_eq!(ron::to_string(&restored).unwrap(), legacy);
}
