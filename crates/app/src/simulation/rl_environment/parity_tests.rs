//! Exercise the actual GA ECS physics system against the RL environment, using
//! independent Avian scenes with the same walls and the same physical actions.
use super::*;
use crate::simulation::{
    Car, CarControls, EvaluationState, FinishReason, KinematicCar,
    systems::{apply_vehicle_physics, ga_canonical_start_for_test},
};
use bevy::ecs::system::RunSystemOnce;

struct GaPhysics {
    scene: Scene,
    car: Entity,
}

impl GaPhysics {
    fn new(track: Track, config: &SimulationConfig, state: VehicleState, dt: f32) -> Self {
        let mut scene = Scene::new(track);
        scene.app.insert_resource(config.clone());
        scene
            .app
            .world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(Duration::from_secs_f32(dt));
        let car = scene
            .app
            .world_mut()
            .spawn((
                Car { id: 0 },
                Transform::from_translation(state.position.extend(2.))
                    .with_rotation(Quat::from_rotation_z(state.heading)),
                KinematicCar {
                    heading: state.heading,
                    speed: state.speed,
                },
                CarControls::NEUTRAL,
                EvaluationState::new(0.),
            ))
            .id();
        Self { scene, car }
    }

    fn step(&mut self, action: [f32; 2]) -> (VehicleState, Option<FinishReason>) {
        self.scene
            .app
            .world_mut()
            .entity_mut(self.car)
            .insert(controls_from_network_outputs(action));
        // This calls GA's apply_car_physics_tick, including its real Avian query
        // and EvaluationState termination path; no parallel physics copy.
        self.scene
            .app
            .world_mut()
            .run_system_once(apply_vehicle_physics)
            .unwrap();
        let car = self.scene.app.world().entity(self.car);
        let pose = car.get::<Transform>().unwrap();
        let kinematic = car.get::<KinematicCar>().unwrap();
        (
            VehicleState {
                position: pose.translation.truncate(),
                heading: kinematic.heading,
                speed: kinematic.speed,
            },
            car.get::<EvaluationState>().unwrap().finish_reason,
        )
    }
}

fn environment(track: Track, config: &Config) -> RacingEnvironment {
    RacingEnvironment::new(
        0,
        vec![Rc::new(Scene::new(track))],
        config,
        &SimulationConfig::default(),
        ChaCha12Rng::seed_from_u64(22),
    )
    .unwrap()
}

#[test]
fn canonical_initial_pose_speed_and_reset_match_ga_on_every_track() {
    let library = TrackLibrary::load_default().unwrap();
    let config = Config::default();
    for definition in library.all_tracks() {
        let track = Track::from_definition(definition).unwrap();
        let expected = ga_canonical_start_for_test(&track);
        let mut env = environment(track, &config);
        assert_eq!(env.collision_events, CollisionEvents::default());
        assert_eq!(env.state, expected, "initial state: {}", definition.id);
        let observation = env.observe().unwrap();
        assert_eq!(&observation[22..], &[0., 0., 0.]);
        env.step([0.5, 1.]).unwrap();
        let events = env.collision_events;
        assert_eq!(events.physical_ticks, 1);
        env.reset().unwrap();
        assert_eq!(env.collision_events, events);
        assert_eq!(env.state, expected, "reset state: {}", definition.id);
        assert_eq!(env.observe().unwrap(), observation);
    }
}

#[test]
fn identical_action_sequences_match_real_ga_physics_on_every_track() {
    let library = TrackLibrary::load_default().unwrap();
    let config = Config::default();
    let actions = [[0., 0.], [0.3, 0.8], [-0.8, 1.], [1., -1.]];
    for definition in library.all_tracks() {
        let track = Track::from_definition(definition).unwrap();
        let mut env = environment(track.clone(), &config);
        let mut ga = GaPhysics::new(track, &env.physics, env.state, config.physical_dt);
        for tick in 0..120 {
            let action = actions[(tick / 15) % actions.len()];
            let (ga_state, ga_reason) = ga.step(action);
            let step = env.step(action).unwrap();
            assert_eq!(env.collision_events.physical_ticks, tick as u64 + 1);
            assert_eq!(
                env.collision_events.translation,
                u64::from(step.reason == Some(EndReason::Collision))
            );
            assert_eq!(
                step.reason == Some(EndReason::Collision),
                ga_reason == Some(FinishReason::Collision),
                "collision on {} at tick {tick}",
                definition.id
            );
            assert_eq!(
                env.state.position, ga_state.position,
                "{} tick {tick}",
                definition.id
            );
            assert_eq!(
                env.state.heading, ga_state.heading,
                "{} tick {tick}",
                definition.id
            );
            assert_eq!(
                env.state.speed, ga_state.speed,
                "{} tick {tick}",
                definition.id
            );
            if step.terminated || step.truncated {
                break;
            }
        }
    }
}

/// A real Avian wall, with its near face at y=7.6 (14.6 - 14/2).
/// The parallel car has half-width 7.5. A normal one-tick steering action
/// intersects that wall, while translation along the rejected heading is safe.
fn wall_fixture() -> Track {
    let library = TrackLibrary::load_default().unwrap();
    let mut definition = library.definition("interlagos").unwrap().clone();
    definition.control_points = vec![[-400., -200.], [400., -200.], [400., 200.], [-400., 200.]];
    definition.start_index = 0;
    let mut track = Track::from_definition(&definition).unwrap();
    track.left_border = vec![
        Vec2::new(-1000., 14.6),
        Vec2::new(1000., 14.6),
        Vec2::new(1000., 1000.),
        Vec2::new(-1000., 1000.),
    ];
    track.right_border = vec![
        Vec2::new(-2000., -2000.),
        Vec2::new(2000., -2000.),
        Vec2::new(2000., -1500.),
        Vec2::new(-2000., -1500.),
    ];
    track
}

#[test]
fn isolated_rotation_collision_rejects_heading_but_does_not_end_ga_or_ppo() {
    let track = wall_fixture();
    let config = Config::default();
    let mut env = environment(track.clone(), &config);
    env.state = VehicleState {
        position: Vec2::ZERO,
        heading: 0.,
        speed: 20.,
    };
    let before = env.state;
    let action = [1., 0.];
    let integrated = integrate_vehicle(
        before,
        controls_from_network_outputs(action),
        &env.physics,
        config.physical_dt,
    );
    let scene = &env.scenes[0];
    assert!(!scene.intersects(before.position, before.heading));
    let resolved = resolve_vehicle_motion(before, integrated, |p, h| scene.intersects(p, h));
    assert!(resolved.rotation_collided);
    assert!(!resolved.translation_collided);
    // This explicitly reproduces the old PPO predicate's disagreement with GA.
    assert!(resolved.rotation_collided || resolved.translation_collided);
    let mut ga = GaPhysics::new(track, &env.physics, before, config.physical_dt);
    let (ga_state, ga_reason) = ga.step(action);
    let step = env.step(action).unwrap();
    assert_eq!(ga_reason, None);
    assert_eq!(step.reason, None);
    assert!(!step.terminated && !step.truncated);
    assert_eq!(step.reward.collision, 0.);
    assert_eq!(
        env.collision_events,
        CollisionEvents {
            physical_ticks: 1,
            translation: 0,
            rotation: 1,
            rotation_only: 1,
        }
    );
    assert_eq!(env.state, ga_state);
    assert_eq!(env.state.heading, before.heading);
    assert!(env.state.position.x > before.position.x);
    assert!(!env.scenes[0].intersects(env.state.position, env.state.heading));
    // The rejected turn must not leave the environment unusable after the tick.
    env.step([0., 0.]).unwrap();
    assert_eq!(
        env.collision_events,
        CollisionEvents {
            physical_ticks: 2,
            translation: 0,
            rotation: 1,
            rotation_only: 1,
        }
    );
    let events = env.collision_events;
    env.reset().unwrap();
    assert_eq!(env.collision_events, events);
}

#[test]
fn translation_collision_ends_both_simulations_at_the_last_safe_pose() {
    let track = wall_fixture();
    let config = Config::default();
    let mut env = environment(track.clone(), &config);
    // Safely below the wall, pointing toward it. A normal translation reaches it.
    env.state = VehicleState {
        position: Vec2::new(0., -0.1),
        heading: std::f32::consts::FRAC_PI_2,
        speed: 45.,
    };
    // Vertical car length is 28; use a center sufficiently far below y=7.6.
    env.state.position.y = -7.;
    let before = env.state;
    assert!(!env.scenes[0].intersects(before.position, before.heading));
    let action = [0., 1.];
    let integrated = integrate_vehicle(
        before,
        controls_from_network_outputs(action),
        &env.physics,
        config.physical_dt,
    );
    let scene = &env.scenes[0];
    let resolved = resolve_vehicle_motion(before, integrated, |p, h| scene.intersects(p, h));
    assert!(!resolved.rotation_collided);
    assert!(resolved.translation_collided);
    let mut ga = GaPhysics::new(track, &env.physics, before, config.physical_dt);
    let (ga_state, ga_reason) = ga.step(action);
    let step = env.step(action).unwrap();
    assert_eq!(ga_reason, Some(FinishReason::Collision));
    assert_eq!(step.reason, Some(EndReason::Collision));
    assert!(step.terminated && !step.truncated);
    assert_eq!(step.reward.collision, -config.reward.collision_penalty);
    assert_eq!(
        env.collision_events,
        CollisionEvents {
            physical_ticks: 1,
            translation: 1,
            rotation: 0,
            rotation_only: 0,
        }
    );
    assert_eq!(env.state.position, ga_state.position);
    assert_eq!(env.state.heading, ga_state.heading);
    assert_eq!(env.state.speed, ga_state.speed);
    assert_eq!(env.state.speed, 0.);
    assert_eq!(env.state.position, before.position);
    assert!(!env.scenes[0].intersects(env.state.position, env.state.heading));
    let events = env.collision_events;
    assert!(env.step(action).is_err());
    assert_eq!(env.collision_events, events);
    env.reset().unwrap();
    assert_eq!(env.collision_events, events);
    assert_eq!(env.state, ga_canonical_start_for_test(&env.scenes[0].track));
    env.step([0., 0.]).unwrap();
    assert_eq!(
        env.collision_events,
        CollisionEvents {
            physical_ticks: 2,
            ..events
        }
    );
}

#[test]
fn simultaneous_rotation_and_translation_events_are_not_classified_as_rotation_only() {
    let track = wall_fixture();
    let config = Config::default();
    let mut env = environment(track.clone(), &config);
    // An intentionally invalid overlap exercises both query flags, independently
    // of the safe-spawn guarantees covered by the canonical-state test.
    env.state = VehicleState {
        position: Vec2::new(0., 10.),
        heading: 0.,
        speed: 0.,
    };
    let before = env.state;
    let action = [0., 0.];
    let scene = &env.scenes[0];
    let integrated = integrate_vehicle(
        before,
        controls_from_network_outputs(action),
        &env.physics,
        config.physical_dt,
    );
    let resolved = resolve_vehicle_motion(before, integrated, |p, h| scene.intersects(p, h));
    assert!(resolved.rotation_collided && resolved.translation_collided);
    let mut ga = GaPhysics::new(track, &env.physics, before, config.physical_dt);
    let (ga_state, ga_reason) = ga.step(action);
    let step = env.step(action).unwrap();
    assert_eq!(ga_reason, Some(FinishReason::Collision));
    assert_eq!(step.reason, Some(EndReason::Collision));
    assert_eq!(env.state, ga_state);
    assert_eq!(
        env.collision_events,
        CollisionEvents {
            physical_ticks: 1,
            translation: 1,
            rotation: 1,
            rotation_only: 0,
        }
    );
}
