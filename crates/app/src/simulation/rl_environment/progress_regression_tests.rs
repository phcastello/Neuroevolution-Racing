use super::*;

#[test]
fn real_interlagos_corner_step_no_longer_erases_continuous_progress() {
    // Frozen physical state/action from E400 diagnostic decisions 1968 -> 1969.
    // Exercise the real integrator, walls, projection and production step path.
    let c = Config {
        num_envs: 1,
        training_tracks: vec!["interlagos".into()],
        ..Config::default()
    };
    let scenes = scenes(
        &TrackLibrary::load_default().unwrap(),
        &c.training_tracks,
        false,
    )
    .unwrap();
    let mut envs = seeded_environments(scenes, &c, &SimulationConfig::default(), 55).unwrap();
    let e = &mut envs[0];
    e.state = VehicleState {
        position: Vec2::new(-1112.6224, 485.7824),
        heading: 0.877625,
        speed: 143.905,
    };
    e.progress_position = e.state.position;
    e.nearest = 323;
    e.progress = Progress {
        previous: 9844.126,
        cumulative: 9828.138,
        best: 9828.138,
    };
    let step = e.step([-0.9459784, -0.8434218]).unwrap();
    assert!(e.state.position.distance(Vec2::new(-1111.0686, 487.55524)) < 0.01);
    assert_eq!(e.nearest, 324);
    assert!((e.progress.cumulative - 9843.592).abs() < 0.02);
    assert!(step.reward.progress > 0.);
    assert!(!step.terminated && !step.truncated);
}

#[test]
fn adjacent_corner_projection_discontinuity_keeps_signed_progress_and_high_water() {
    // Actual E400 Interlagos corner: 15.454 projected units in 2.357 physical
    // units, switching segments 323 -> 324. The old guard erased 9828 units.
    assert!(movement_is_connected(Vec2::ZERO, Vec2::ZERO, 323, 324, 640));
    assert!(!movement_is_connected(Vec2::ZERO, Vec2::ONE, 323, 324, 640));
    assert!(!movement_is_connected(
        Vec2::ZERO,
        Vec2::ZERO,
        323,
        325,
        640
    ));
    assert!(movement_is_connected(Vec2::ZERO, Vec2::ZERO, 639, 0, 640));
    let mut p = Progress {
        previous: 9844.126,
        cumulative: 9828.138,
        best: 9828.138,
    };
    let gain = p.advance_connected(9859.58, 18340.092, 2.3573954, true);
    assert!((gain - 15.454102).abs() < 0.01);
    let best = p.best;
    for _ in 0..10 {
        assert_eq!(
            p.advance_connected(9844.126, 18340.092, 2.3573954, true),
            0.
        );
        assert_eq!(p.advance_connected(9859.58, 18340.092, 2.3573954, true), 0.);
    }
    assert_eq!(p.best, best);
    assert_eq!(p.advance_connected(15000., 18340.092, 0.01, false), 0.);
    assert_eq!(p.cumulative, 0.);
}

#[test]
fn full_lap_is_still_required_and_relocation_is_not_progress() {
    let c = Config {
        num_envs: 1,
        training_tracks: vec!["interlagos".into()],
        ..Config::default()
    };
    let scenes = scenes(
        &TrackLibrary::load_default().unwrap(),
        &c.training_tracks,
        false,
    )
    .unwrap();
    let mut envs = seeded_environments(scenes, &c, &SimulationConfig::default(), 9).unwrap();
    let e = &mut envs[0];
    let length = e.scenes[0].track.total_length;
    e.progress.best = length * 0.9999;
    e.progress.cumulative = e.progress.best;
    e.state.position = e.scenes[0].track.point_at_distance(500.);
    let result = e.step([0., 0.]).unwrap();
    assert!(!result.terminated && !result.truncated);
    assert_eq!(result.reward.progress, 0.);
    assert_eq!(e.progress.cumulative, 0.);
}

/// The rejected-projection branch before the fix did not advance its reference.
fn legacy_advance(previous: &mut f32, projected: f32, length: f32, movement: f32) -> bool {
    let delta = wrapped_distance_delta(*previous, projected, length);
    if delta.abs() > 4. * movement + 2. {
        return false;
    }
    *previous = projected;
    true
}

#[test]
fn rejected_projection_does_not_leave_the_reference_permanently_stale() {
    let mut legacy_previous = 10.;
    for projected in 50..=90 {
        assert!(!legacy_advance(
            &mut legacy_previous,
            projected as f32,
            200.,
            1.
        ));
    }
    assert_eq!(legacy_previous, 10.);

    let mut progress = Progress {
        previous: 10.,
        ..Default::default()
    };
    assert_eq!(progress.advance(50., 200., 1.), 0.);
    assert_eq!(progress.previous, 50.);
    assert_eq!(progress.cumulative, 0.);
    assert_eq!(progress.best, 0.);
    for projected in 51..=90 {
        assert_eq!(progress.advance(projected as f32, 200., 1.), 1.);
    }
    assert_eq!(progress.best, 40.);
}

#[test]
fn ga_and_ppo_references_follow_the_same_real_projection_after_a_rejected_jump() {
    use crate::simulation::{
        components::{Car, CarProgress},
        systems::update_track_progress,
    };
    use bevy::ecs::system::RunSystemOnce;

    let library = TrackLibrary::load_default().unwrap();
    let track = Track::from_definition(library.definition("interlagos").unwrap()).unwrap();
    let length = track.total_length;
    let initial_position = track.point_at_distance(10.);
    let relocated_position = track.point_at_distance(50.);
    let next_position = track.point_at_distance(51.);
    let initial_projection = track.project(initial_position);
    let mut progress = Progress {
        previous: initial_projection.track_distance,
        ..Default::default()
    };
    let mut world = World::new();
    world.insert_resource(track);
    world.insert_resource(SimulationConfig::default());
    let car = world
        .spawn((
            Car { id: 0 },
            Transform::from_translation(initial_position.extend(2.)),
            CarProgress::new(
                initial_projection.track_distance,
                length,
                initial_projection.segment_index,
                initial_projection.point,
            ),
        ))
        .id();

    // Exercise GA's actual ECS system, not a copy of its reference-update logic.
    world.get_mut::<Transform>(car).unwrap().translation = relocated_position.extend(2.);
    world.run_system_once(update_track_progress).unwrap();
    let relocated_reference = world.get::<CarProgress>(car).unwrap().centerline_distance;
    assert!((relocated_reference - 50.).abs() < 0.01);
    assert!(wrapped_distance_delta(progress.previous, relocated_reference, length).abs() > 2.);
    assert_eq!(progress.advance(relocated_reference, length, 0.), 0.);
    assert_eq!(progress.previous, relocated_reference);
    assert_eq!(progress.best, 0.);

    world.get_mut::<Transform>(car).unwrap().translation = next_position.extend(2.);
    world.run_system_once(update_track_progress).unwrap();
    let next_reference = world.get::<CarProgress>(car).unwrap().centerline_distance;
    let movement = relocated_position.distance(next_position);
    let expected_delta = wrapped_distance_delta(relocated_reference, next_reference, length);
    assert!(expected_delta > 0. && expected_delta <= 4. * movement + 2.);
    let gain = progress.advance(next_reference, length, movement);
    assert_eq!(progress.previous, next_reference);
    assert!((gain - expected_delta).abs() < 0.0001);
}

#[test]
fn repeated_segment_jumps_are_never_credited() {
    let mut progress = Progress {
        previous: 10.,
        ..Default::default()
    };
    for projected in [50., 90., 130., 170., 10., 50., 90.] {
        assert_eq!(progress.advance(projected, 200., 0.01), 0.);
        assert_eq!(progress.previous, projected);
        assert_eq!(progress.best, 0.);
    }
}

#[test]
fn repeated_forward_backward_branch_jumps_cannot_farm_previous_progress() {
    let mut progress = Progress {
        previous: 10.,
        ..Default::default()
    };
    assert_eq!(progress.advance(50., 200., 0.01), 0.);
    assert_eq!(progress.advance(51., 200., 1.), 1.);
    for _ in 0..10 {
        assert_eq!(progress.advance(10., 200., 0.01), 0.);
        assert_eq!(progress.advance(11., 200., 1.), 0.);
        assert_eq!(progress.advance(50., 200., 0.01), 0.);
        assert_eq!(progress.advance(51., 200., 1.), 0.);
    }
    assert_eq!(progress.best, 1.);
}

#[test]
fn successive_forward_branch_jumps_across_the_seam_cannot_farm_progress() {
    let mut progress = Progress {
        previous: 10.,
        ..Default::default()
    };
    // Every rejected jump has a positive wrapped delta, including 171 -> 10.
    // Debiting only rejected negative deltas therefore cannot protect this cycle.
    assert_eq!(progress.advance(50., 200., 0.01), 0.);
    assert_eq!(progress.advance(51., 200., 1.), 1.);
    for _ in 0..10 {
        for projected in [90., 130., 170., 10., 50.] {
            assert_eq!(progress.advance(projected, 200., 0.01), 0.);
            assert_eq!(progress.advance(projected + 1., 200., 1.), 0.);
        }
    }
    assert_eq!(progress.best, 1.);
}

#[test]
fn reanchored_stationary_reverse_and_oscillating_motion_is_not_rewarded() {
    let mut progress = Progress {
        previous: 10.,
        ..Default::default()
    };
    assert_eq!(progress.advance(50., 200., 0.), 0.);
    assert_eq!(progress.advance(50., 200., 0.), 0.);
    assert_eq!(progress.advance(49., 200., 1.), 0.);
    assert_eq!(progress.advance(50., 200., 1.), 0.);
    assert_eq!(progress.advance(51., 200., 1.), 1.);
    for _ in 0..10 {
        assert_eq!(progress.advance(49., 200., 2.), 0.);
        assert_eq!(progress.advance(51., 200., 2.), 0.);
    }
    assert_eq!(progress.best, 1.);
}

#[test]
fn real_seam_crossings_still_advance_but_discontinuous_crossings_do_not() {
    let mut progress = Progress {
        previous: 198.,
        ..Default::default()
    };
    assert_eq!(progress.advance(199., 200., 1.), 1.);
    assert_eq!(progress.advance(0., 200., 1.), 1.);
    assert_eq!(progress.advance(1., 200., 1.), 1.);
    assert_eq!(progress.advance(199., 200., 2.), 0.);
    assert_eq!(progress.advance(1., 200., 2.), 0.);
    assert_eq!(progress.best, 3.);

    let mut discontinuous = Progress {
        previous: 190.,
        ..Default::default()
    };
    assert_eq!(discontinuous.advance(10., 200., 0.01), 0.);
    assert_eq!(discontinuous.previous, 10.);
    assert_eq!(discontinuous.advance(11., 200., 1.), 1.);
    assert_eq!(discontinuous.advance(190., 200., 0.01), 0.);
    assert_eq!(discontinuous.advance(191., 200., 1.), 0.);
    assert_eq!(discontinuous.best, 1.);
}

#[test]
fn existing_reverse_debt_survives_a_rejected_projection() {
    let mut progress = Progress {
        previous: 50.,
        cumulative: -10.,
        best: 10.,
    };
    assert_eq!(progress.advance(20., 200., 0.01), 0.);
    assert_eq!(progress.previous, 20.);
    assert_eq!(progress.cumulative, -10.);
    for projected in 21..=40 {
        assert_eq!(progress.advance(projected as f32, 200., 1.), 0.);
    }
    assert_eq!(progress.best, 10.);
    assert_eq!(progress.advance(41., 200., 1.), 1.);
}
