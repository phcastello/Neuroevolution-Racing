use std::f32::consts::{FRAC_PI_3, FRAC_PI_6};

use avian2d::prelude::*;
use bevy::prelude::*;
use bevy_egui::input::EguiWantsInput;

use super::{
    CAR_LENGTH, CAR_WIDTH, ManualControlMode, PlaybackState, SimulationConfig, SimulationMode,
    TestDriveEnvironment, TestDriveSettings, TrackLibrary, TrackSelection,
    components::{
        Car, CarProgress, ControllerTuning, KinematicCar, ManualCar, SelectedCar, SensorReadings,
        SimulationLayer, TemporaryControlled, TrackWall,
    },
    controller::{
        CarController, CarControls, CarObservation, TemporaryController,
        TemporaryNavigationContext, signed_angle_to,
    },
    track::{Track, closed_segments, normalized_progress, wrapped_distance_delta},
};

// Array order is left -> right, matching the future MLP input contract.
const SENSOR_ANGLES: [f32; 5] = [FRAC_PI_3, FRAC_PI_6, 0.0, -FRAC_PI_6, -FRAC_PI_3];
const WALL_THICKNESS: f32 = 14.0;
const MIN_GRIP_SPEED: f32 = 10.0;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimulationSet {
    Sense,
    ControlSource,
    Physics,
    Progress,
}

type ReplaceableTrackEntities<'w, 's> = Query<'w, 's, Entity, Or<(With<Car>, With<TrackWall>)>>;
type TemporaryCars<'w, 's> = Query<
    'w,
    's,
    (
        &'static Transform,
        &'static KinematicCar,
        &'static CarObservation,
        &'static CarProgress,
        &'static ControllerTuning,
        &'static mut CarControls,
    ),
    With<TemporaryControlled>,
>;
type ManualCars<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static mut Transform,
        &'static mut KinematicCar,
        &'static mut CarControls,
        &'static mut SensorReadings,
        &'static mut CarObservation,
    ),
    With<ManualCar>,
>;

#[derive(Default)]
pub(super) struct SimulationLifecycleState {
    mode: Option<SimulationMode>,
    environment: TestDriveEnvironment,
}

pub fn apply_track_selection(
    library: Res<TrackLibrary>,
    mut selection: ResMut<TrackSelection>,
    mut track: ResMut<Track>,
) {
    let Some(requested_id) = selection.requested_id.take() else {
        return;
    };
    let Some(definition) = library.definition(&requested_id) else {
        selection.status = format!("Unknown track: {requested_id}");
        return;
    };
    match Track::from_definition(definition) {
        Ok(new_track) => {
            *track = new_track;
            selection.active_id = requested_id;
            selection.status = format!("Loaded {}", definition.name);
        }
        Err(error) => selection.status = format!("Track switch failed: {error}"),
    }
}

pub fn rebuild_simulation(
    mut commands: Commands,
    track: Res<Track>,
    config: Res<SimulationConfig>,
    mode: Res<SimulationMode>,
    test_drive: Res<TestDriveSettings>,
    old_entities: ReplaceableTrackEntities,
    mut lifecycle: Local<SimulationLifecycleState>,
) {
    if !track.is_changed()
        && lifecycle.mode == Some(*mode)
        && lifecycle.environment == test_drive.environment
    {
        return;
    }
    lifecycle.mode = Some(*mode);
    lifecycle.environment = test_drive.environment;
    for entity in &old_entities {
        commands.entity(entity).despawn();
    }
    let open_field = *mode == SimulationMode::TestDrive
        && test_drive.environment == TestDriveEnvironment::OpenField;
    if !open_field {
        spawn_track_colliders(&mut commands, &track);
    }
    if *mode == SimulationMode::TestDrive {
        spawn_manual_car(&mut commands, &track, test_drive.environment);
    } else {
        spawn_temporary_cars(&mut commands, &track, &config);
    }
}

fn spawn_track_colliders(commands: &mut Commands, track: &Track) {
    for (start, end) in
        closed_segments(&track.left_border).chain(closed_segments(&track.right_border))
    {
        let delta = end - start;
        commands.spawn((
            TrackWall,
            RigidBody::Static,
            Collider::rectangle(delta.length(), WALL_THICKNESS),
            CollisionLayers::new([SimulationLayer::TrackWall], [SimulationLayer::Car]),
            Transform::from_translation(((start + end) * 0.5).extend(0.0))
                .with_rotation(Quat::from_rotation_z(delta.y.atan2(delta.x))),
        ));
    }
}

fn spawn_temporary_cars(commands: &mut Commands, track: &Track, config: &SimulationConfig) {
    let usable_width = (track.width - CAR_WIDTH - 12.0).max(0.0);
    let columns = ((usable_width / (CAR_WIDTH + 5.0)).floor() as usize + 1)
        .clamp(1, 5)
        .min(config.population_size.max(1));
    let column_spacing = if columns > 1 {
        usable_width.min((columns - 1) as f32 * 17.0) / (columns - 1) as f32
    } else {
        0.0
    };
    for id in 0..config.population_size {
        let row = (id / columns) as f32;
        let column = id % columns;
        let lateral_offset = (column as f32 - (columns - 1) as f32 * 0.5) * column_spacing;
        let grid_distance = 16.0 + row * 34.0;
        let center = track.point_at_distance(grid_distance);
        let center_projection = track.project(center);
        let forward = track.samples[center_projection.segment_index].tangent;
        let position = center + forward.perp() * lateral_offset;
        let heading = forward.y.atan2(forward.x);
        let projection = track.project(position);
        let mut entity = commands.spawn((
            Car { id },
            KinematicCar {
                heading,
                speed: 45.0 + id as f32 * 0.7,
            },
            SensorReadings::default(),
            CarObservation {
                sensors: [1.0; 5],
                normalized_speed: 0.0,
            },
            CarControls::NEUTRAL,
            CarProgress::new(
                projection.track_distance,
                track.total_length,
                projection.segment_index,
                projection.point,
            ),
            ControllerTuning {
                steering_bias: (lateral_offset * 0.00025).clamp(-0.012, 0.012),
            },
            TemporaryControlled,
            CollisionLayers::new([SimulationLayer::Car], [SimulationLayer::TrackWall]),
            Transform::from_translation(position.extend(2.0))
                .with_rotation(Quat::from_rotation_z(heading)),
        ));
        if id == 0 {
            entity.insert(SelectedCar);
        }
    }
}

fn spawn_manual_car(commands: &mut Commands, track: &Track, environment: TestDriveEnvironment) {
    let (position, heading, progress) = manual_spawn_state(track, environment);
    let mut entity = commands.spawn((
        Car { id: 0 },
        ManualCar,
        SelectedCar,
        KinematicCar {
            heading,
            speed: 0.0,
        },
        SensorReadings::default(),
        CarObservation {
            sensors: [1.0; 5],
            normalized_speed: 0.0,
        },
        CarControls::NEUTRAL,
        CollisionLayers::new([SimulationLayer::Car], [SimulationLayer::TrackWall]),
        Transform::from_translation(position.extend(2.0))
            .with_rotation(Quat::from_rotation_z(heading)),
    ));
    if let Some(progress) = progress {
        entity.insert(progress);
    }
}

fn manual_spawn_state(
    track: &Track,
    environment: TestDriveEnvironment,
) -> (Vec2, f32, Option<CarProgress>) {
    if environment == TestDriveEnvironment::OpenField {
        return (Vec2::ZERO, 0.0, None);
    }
    let position = track.point_at_distance(16.0);
    let projection = track.project(position);
    let tangent = track.samples[projection.segment_index].tangent;
    (
        position,
        tangent.y.atan2(tangent.x),
        Some(CarProgress::new(
            projection.track_distance,
            track.total_length,
            projection.segment_index,
            projection.point,
        )),
    )
}

pub fn sample_sensors(
    spatial_query: SpatialQuery,
    config: Res<SimulationConfig>,
    mut cars: Query<
        (
            &Transform,
            &KinematicCar,
            &mut SensorReadings,
            &mut CarObservation,
        ),
        With<Car>,
    >,
) {
    let filter = SpatialQueryFilter::from_mask([SimulationLayer::TrackWall]);
    for (transform, state, mut sensors, mut observation) in &mut cars {
        let origin = transform.translation.truncate();
        for (index, relative_angle) in SENSOR_ANGLES.iter().enumerate() {
            let direction = Vec2::from_angle(state.heading + relative_angle);
            let direction = Dir2::new(direction).expect("sensor direction is non-zero");
            let hit_distance = spatial_query
                .cast_ray(origin, direction, config.sensor_max_distance, true, &filter)
                .map_or(config.sensor_max_distance, |hit| hit.distance);
            sensors.normalized[index] = (hit_distance / config.sensor_max_distance).clamp(0.0, 1.0);
            sensors.endpoints[index] = origin + direction.as_vec2() * hit_distance;
        }
        observation.sensors = sensors.normalized;
        observation.normalized_speed =
            normalize_speed(state.speed, config.speed_normalization_scale);
    }
}

pub fn produce_temporary_controls(
    config: Res<SimulationConfig>,
    track: Res<Track>,
    mut cars: TemporaryCars,
) {
    let mut controller = TemporaryController::default();
    for (transform, state, observation, progress, tuning, mut controls) in &mut cars {
        let position = transform.translation.truncate();
        let target = track.point_at_distance(
            progress.centerline_distance + config.temporary_controller_look_ahead,
        );
        let target_direction = target - position;
        let bearing = signed_angle_to(state.heading, target_direction) + tuning.steering_bias;
        controller.set_navigation_context(TemporaryNavigationContext {
            target_bearing: bearing,
        });
        *controls = controller.control(observation);
    }
}

pub fn manual_controls_from_keys(
    accelerate: bool,
    reverse: bool,
    left: bool,
    right: bool,
) -> CarControls {
    let acceleration = match (accelerate, reverse) {
        (true, false) => 1.0,
        (false, true) => -1.0,
        _ => 0.0,
    };
    // Positive steering rotates counter-clockwise in world space, so +1 is left.
    let steering = match (left, right) {
        (true, false) => 1.0,
        (false, true) => -1.0,
        _ => 0.0,
    };
    CarControls::new(acceleration, steering)
}

pub fn produce_manual_controls(
    keyboard: Res<ButtonInput<KeyCode>>,
    egui_input: Res<EguiWantsInput>,
    test_drive: Res<TestDriveSettings>,
    mut cars: Query<&mut CarControls, With<ManualCar>>,
) {
    let controls = match test_drive.control_mode {
        ManualControlMode::Keyboard if !egui_input.wants_any_keyboard_input() => {
            manual_controls_from_keys(
                keyboard.pressed(KeyCode::KeyW),
                keyboard.pressed(KeyCode::KeyS),
                keyboard.pressed(KeyCode::KeyA),
                keyboard.pressed(KeyCode::KeyD),
            )
        }
        ManualControlMode::Keyboard => CarControls::NEUTRAL,
        ManualControlMode::Sliders => CarControls::new(
            test_drive.slider_controls.acceleration,
            test_drive.slider_controls.steering,
        ),
    };
    for mut car_controls in &mut cars {
        *car_controls = controls;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct VehicleState {
    position: Vec2,
    heading: f32,
    speed: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ResolvedVehicleMotion {
    state: VehicleState,
    rotation_collided: bool,
    translation_collided: bool,
}

fn normalize_speed(speed: f32, normalization_scale: f32) -> f32 {
    let scale = normalization_scale.max(f32::EPSILON);
    (1.0 - (-speed.abs() / scale).exp()).clamp(0.0, 1.0)
}

fn propulsion_efficiency(speed: f32, falloff_speed: f32) -> f32 {
    let scale = falloff_speed.max(f32::EPSILON);
    1.0 / (1.0 + speed.abs() / scale)
}

fn coast_toward_rest(speed: f32, deceleration: f32, dt: f32) -> f32 {
    let speed_loss = deceleration.max(0.0) * dt.max(0.0);
    if speed > 0.0 {
        (speed - speed_loss).max(0.0)
    } else {
        (speed + speed_loss).min(0.0)
    }
}

pub(crate) fn desired_yaw_rate(speed: f32, steering: f32, config: &SimulationConfig) -> f32 {
    let speed_ratio = normalize_speed(speed, config.speed_normalization_scale).clamp(0.15, 1.0);
    steering * config.turn_rate * speed_ratio
}

pub(crate) fn max_grip_yaw_rate(speed: f32, config: &SimulationConfig) -> f32 {
    let effective_speed = speed.abs().max(MIN_GRIP_SPEED);
    config.max_lateral_acceleration.max(0.0) / effective_speed
}

pub(crate) fn limited_yaw_rate(speed: f32, steering: f32, config: &SimulationConfig) -> f32 {
    let desired = desired_yaw_rate(speed, steering, config);
    let grip_limit = max_grip_yaw_rate(speed, config);
    desired.clamp(-grip_limit, grip_limit)
}

fn integrate_vehicle(
    state: VehicleState,
    controls: CarControls,
    config: &SimulationConfig,
    dt: f32,
) -> VehicleState {
    let controls = CarControls::new(controls.acceleration, controls.steering);
    let speed = if controls.acceleration.abs() <= f32::EPSILON {
        coast_toward_rest(state.speed, config.coasting_deceleration, dt)
    } else {
        let acceleration_rate = if controls.acceleration > 0.0 {
            config.acceleration_rate
        } else {
            config.braking_rate
        };
        // Propulsion becomes progressively weaker while increasing speed magnitude.
        // An opposing command keeps the full rate so braking remains responsive.
        let gaining_speed = controls.acceleration * state.speed >= 0.0;
        let effective_rate = if gaining_speed {
            acceleration_rate
                * propulsion_efficiency(state.speed, config.acceleration_falloff_speed)
        } else {
            acceleration_rate
        };
        state.speed + controls.acceleration * effective_rate * dt
    };
    let yaw_rate = limited_yaw_rate(speed, controls.steering, config);
    let heading = state.heading + yaw_rate * dt;
    let position = state.position + Vec2::from_angle(heading) * speed * dt;
    VehicleState {
        position,
        heading,
        speed,
    }
}

/// Resolves rotation and translation independently so that rejecting a movement
/// can never leave the car rotated into a wall.
fn resolve_vehicle_motion(
    current: VehicleState,
    integrated: VehicleState,
    mut intersects_wall: impl FnMut(Vec2, f32) -> bool,
) -> ResolvedVehicleMotion {
    let rotation_collided = intersects_wall(current.position, integrated.heading);
    let heading = if rotation_collided {
        current.heading
    } else {
        integrated.heading
    };

    // If steering was rejected, keep moving along the last valid heading. This
    // lets the car continue parallel to a wall instead of snagging on it.
    let intended_displacement = integrated.position - current.position;
    let translation_target = if rotation_collided {
        current.position
            + Vec2::from_angle(heading) * intended_displacement.length() * integrated.speed.signum()
    } else {
        integrated.position
    };
    let translation_collided = intersects_wall(translation_target, heading);
    let position = if translation_collided {
        current.position
    } else {
        translation_target
    };

    ResolvedVehicleMotion {
        state: VehicleState {
            position,
            heading,
            speed: integrated.speed,
        },
        rotation_collided,
        translation_collided,
    }
}

pub fn apply_vehicle_physics(
    time: Res<Time<Fixed>>,
    config: Res<SimulationConfig>,
    spatial_query: SpatialQuery,
    mut cars: Query<(&mut Transform, &mut KinematicCar, &CarControls), With<Car>>,
) {
    let dt = time.delta_secs();
    let car_shape = Collider::rectangle(CAR_LENGTH, CAR_WIDTH);
    let wall_filter = SpatialQueryFilter::from_mask([SimulationLayer::TrackWall]);

    for (mut transform, mut state, controls) in &mut cars {
        let current = VehicleState {
            position: transform.translation.truncate(),
            heading: state.heading,
            speed: state.speed,
        };
        let integrated = integrate_vehicle(current, *controls, &config, dt);
        let resolved = resolve_vehicle_motion(current, integrated, |position, heading| {
            !spatial_query
                .shape_intersections(&car_shape, position, heading, &wall_filter)
                .is_empty()
        });

        if resolved.translation_collided {
            state.speed = -integrated.speed.abs().min(18.0) * 0.35;
        } else {
            state.speed = integrated.speed;
        }
        transform.translation.x = resolved.state.position.x;
        transform.translation.y = resolved.state.position.y;
        state.heading = resolved.state.heading;
        transform.rotation = Quat::from_rotation_z(state.heading);
    }
}

pub fn update_track_progress(
    track: Res<Track>,
    config: Res<SimulationConfig>,
    mut cars: Query<(&Transform, &mut CarProgress), With<Car>>,
) {
    for (transform, mut progress) in &mut cars {
        let position = transform.translation.truncate();
        let projection = track.project_near(
            position,
            progress.nearest_segment,
            config.progress_search_radius,
        );
        let delta = wrapped_distance_delta(
            progress.centerline_distance,
            projection.track_distance,
            track.total_length,
        );

        progress.track_distance = (progress.track_distance + delta).clamp(0.0, track.total_length);
        progress.best_track_distance = progress.best_track_distance.max(progress.track_distance);
        progress.normalized_progress =
            normalized_progress(progress.track_distance, track.total_length);
        progress.nearest_segment = projection.segment_index;
        progress.projected_point = projection.point;
        progress.centerline_distance = projection.track_distance;
    }
}

pub fn handle_test_drive_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    egui_input: Res<EguiWantsInput>,
    mode: Res<SimulationMode>,
    mut test_drive: ResMut<TestDriveSettings>,
) {
    if *mode == SimulationMode::TestDrive
        && !egui_input.wants_any_keyboard_input()
        && keyboard.just_pressed(KeyCode::KeyR)
    {
        test_drive.reset_requested = true;
    }
}

pub fn reset_manual_car(
    mut commands: Commands,
    mode: Res<SimulationMode>,
    track: Res<Track>,
    mut test_drive: ResMut<TestDriveSettings>,
    mut cars: ManualCars,
) {
    if *mode != SimulationMode::TestDrive || !test_drive.reset_requested {
        return;
    }
    test_drive.reset_requested = false;
    test_drive.slider_controls = CarControls::NEUTRAL;
    let (position, heading, progress) = manual_spawn_state(&track, test_drive.environment);
    for (entity, mut transform, mut state, mut controls, mut sensors, mut observation) in &mut cars
    {
        transform.translation.x = position.x;
        transform.translation.y = position.y;
        transform.rotation = Quat::from_rotation_z(heading);
        state.heading = heading;
        state.speed = 0.0;
        *controls = CarControls::NEUTRAL;
        *sensors = SensorReadings::default();
        observation.sensors = [1.0; 5];
        observation.normalized_speed = 0.0;
        if let Some(progress) = progress.clone() {
            commands.entity(entity).insert(progress);
        } else {
            commands.entity(entity).remove::<CarProgress>();
        }
    }
}

pub fn toggle_pause_from_keyboard(
    keyboard: Res<ButtonInput<KeyCode>>,
    egui_input: Res<EguiWantsInput>,
    mut playback: ResMut<PlaybackState>,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    if !egui_input.wants_any_keyboard_input() && keyboard.just_pressed(KeyCode::Space) {
        playback.paused = !playback.paused;
        if playback.paused {
            virtual_time.pause();
        } else {
            virtual_time.unpause();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switching_tracks_replaces_walls_cars_and_progress_state() {
        let library = TrackLibrary::load_default().unwrap();
        let initial = Track::from_definition(library.definition("interlagos").unwrap()).unwrap();
        let mut selection = TrackSelection {
            active_id: "interlagos".into(),
            status: String::new(),
            requested_id: None,
        };
        selection.request("monza");

        let mut app = App::new();
        app.insert_resource(library)
            .insert_resource(initial)
            .insert_resource(selection)
            .insert_resource(SimulationMode::Training)
            .insert_resource(TestDriveSettings::default())
            .insert_resource(SimulationConfig {
                population_size: 3,
                ..default()
            })
            .add_systems(Update, (apply_track_selection, rebuild_simulation).chain());
        app.update();

        let monza_samples = app.world().resource::<Track>().samples.len();
        let first_cars = app
            .world_mut()
            .query_filtered::<Entity, With<Car>>()
            .iter(app.world())
            .collect::<Vec<_>>();
        assert_eq!(first_cars.len(), 3);
        assert_eq!(
            app.world_mut()
                .query_filtered::<Entity, With<TrackWall>>()
                .iter(app.world())
                .count(),
            monza_samples * 2
        );

        app.world_mut()
            .resource_mut::<TrackSelection>()
            .request("spa");
        app.update();

        let spa_track = app.world().resource::<Track>();
        assert_eq!(spa_track.definition.id, "spa");
        let spa_samples = spa_track.samples.len();
        let spa_total_length = spa_track.total_length;
        let second_cars = app
            .world_mut()
            .query_filtered::<(Entity, &CarProgress), With<Car>>()
            .iter(app.world())
            .map(|(entity, progress)| {
                assert!(progress.track_distance < spa_total_length);
                assert!(progress.best_track_distance < spa_total_length);
                assert!((0.0..=1.0).contains(&progress.normalized_progress));
                entity
            })
            .collect::<Vec<_>>();
        assert_eq!(second_cars.len(), 3);
        assert!(first_cars.iter().all(|first| !second_cars.contains(first)));
        assert_eq!(
            app.world_mut()
                .query_filtered::<Entity, With<TrackWall>>()
                .iter(app.world())
                .count(),
            spa_samples * 2
        );
    }

    #[test]
    fn neutral_and_opposing_manual_keys_cancel() {
        assert_eq!(
            manual_controls_from_keys(false, false, false, false),
            CarControls::NEUTRAL
        );
        assert_eq!(
            manual_controls_from_keys(true, true, false, false),
            CarControls::NEUTRAL
        );
        assert_eq!(
            manual_controls_from_keys(false, false, true, true),
            CarControls::NEUTRAL
        );
        assert_eq!(
            manual_controls_from_keys(true, false, true, false),
            CarControls::new(1.0, 1.0)
        );
    }

    #[test]
    fn identical_controls_have_identical_deterministic_integration() {
        let initial = VehicleState {
            position: Vec2::new(12.0, -5.0),
            heading: 0.7,
            speed: 82.0,
        };
        let controls = CarControls::new(0.8, 0.35);
        let config = SimulationConfig::default();
        let manual_result = integrate_vehicle(initial, controls, &config, 1.0 / 60.0);
        let future_mlp_result = integrate_vehicle(initial, controls, &config, 1.0 / 60.0);
        assert_eq!(manual_result, future_mlp_result);
    }

    #[test]
    fn high_speed_reduces_yaw_rate_and_increases_turning_radius() {
        let config = SimulationConfig::default();
        let controls = CarControls::new(0.0, 1.0);
        let dt = 1.0 / 60.0;
        let low_speed = VehicleState {
            position: Vec2::ZERO,
            heading: 0.0,
            speed: 20.0,
        };
        let high_speed = VehicleState {
            speed: 500.0,
            ..low_speed
        };

        let low_result = integrate_vehicle(low_speed, controls, &config, dt);
        let high_result = integrate_vehicle(high_speed, controls, &config, dt);
        let low_yaw_rate = (low_result.heading - low_speed.heading) / dt;
        let high_yaw_rate = (high_result.heading - high_speed.heading) / dt;
        let low_radius = low_result.speed.abs() / low_yaw_rate.abs();
        let high_radius = high_result.speed.abs() / high_yaw_rate.abs();

        assert!(high_yaw_rate.abs() < low_yaw_rate.abs());
        assert!(high_radius > low_radius);
    }

    #[test]
    fn integrated_yaw_rate_respects_lateral_acceleration_limit() {
        let config = SimulationConfig::default();
        let initial = VehicleState {
            position: Vec2::ZERO,
            heading: 0.0,
            speed: 500.0,
        };
        let dt = 1.0 / 60.0;
        let result = integrate_vehicle(initial, CarControls::new(0.0, 1.0), &config, dt);
        let actual_yaw_rate = (result.heading - initial.heading) / dt;
        let lateral_acceleration = (result.speed * actual_yaw_rate).abs();

        assert!(lateral_acceleration <= config.max_lateral_acceleration + 1.0e-3);
        assert!((lateral_acceleration - config.max_lateral_acceleration).abs() < 1.0e-2);
    }

    #[test]
    fn low_speed_steering_remains_limited_by_normal_turn_rate() {
        let config = SimulationConfig::default();
        let speed = 5.0;
        let requested = desired_yaw_rate(speed, 1.0, &config);
        let actual = limited_yaw_rate(speed, 1.0, &config);

        assert!((actual - requested).abs() < 1.0e-6);
        assert!(actual > 0.0);
    }

    #[test]
    fn reverse_is_finite_and_uses_the_same_grip_magnitude() {
        let config = SimulationConfig::default();
        let forward_yaw_rate = limited_yaw_rate(500.0, 1.0, &config);
        let reverse_yaw_rate = limited_yaw_rate(-500.0, 1.0, &config);
        let reverse = VehicleState {
            position: Vec2::ZERO,
            heading: 0.0,
            speed: -500.0,
        };
        let result = integrate_vehicle(reverse, CarControls::new(0.0, 1.0), &config, 1.0 / 60.0);

        assert_eq!(forward_yaw_rate, reverse_yaw_rate);
        assert!(result.position.is_finite());
        assert!(result.heading.is_finite());
        assert!(result.speed.is_finite());
        assert!(
            (reverse.speed.abs() * reverse_yaw_rate.abs())
                <= config.max_lateral_acceleration + 1.0e-3
        );
    }

    #[test]
    fn rotation_into_a_wall_is_rejected_without_discarding_safe_translation() {
        let current = VehicleState {
            position: Vec2::ZERO,
            heading: 0.0,
            speed: 20.0,
        };
        let integrated = VehicleState {
            position: Vec2::new(0.0, 1.0),
            heading: std::f32::consts::FRAC_PI_2,
            speed: 20.0,
        };

        // A horizontal wall whose near edge is 9 units above the car center.
        // The parallel car reaches y=7.5; after a 90-degree turn it reaches y=14.
        let resolved = resolve_vehicle_motion(current, integrated, |position, heading| {
            let half_extent_y =
                heading.sin().abs() * (CAR_LENGTH * 0.5) + heading.cos().abs() * (CAR_WIDTH * 0.5);
            position.y + half_extent_y >= 9.0
        });

        assert!(resolved.rotation_collided);
        assert!(!resolved.translation_collided);
        assert_eq!(resolved.state.heading, current.heading);
        assert_eq!(resolved.state.position, Vec2::new(1.0, 0.0));
    }

    #[test]
    fn translation_into_a_wall_does_not_change_the_last_safe_pose() {
        let current = VehicleState {
            position: Vec2::ZERO,
            heading: 0.0,
            speed: 20.0,
        };
        let integrated = VehicleState {
            position: Vec2::new(0.0, 2.0),
            ..current
        };

        let resolved = resolve_vehicle_motion(current, integrated, |position, _| {
            position.y + CAR_WIDTH * 0.5 >= 9.0
        });

        assert!(!resolved.rotation_collided);
        assert!(resolved.translation_collided);
        assert_eq!(resolved.state.heading, current.heading);
        assert_eq!(resolved.state.position, current.position);
    }

    #[test]
    fn acceleration_gain_decreases_as_speed_increases_without_a_hard_limit() {
        let config = SimulationConfig::default();
        let controls = CarControls::new(1.0, 0.0);
        let from_rest = VehicleState {
            position: Vec2::ZERO,
            heading: 0.0,
            speed: 0.0,
        };
        let already_fast = VehicleState {
            speed: 500.0,
            ..from_rest
        };
        let rest_result = integrate_vehicle(from_rest, controls, &config, 1.0);
        let fast_result = integrate_vehicle(already_fast, controls, &config, 1.0);
        let rest_gain = rest_result.speed - from_rest.speed;
        let fast_gain = fast_result.speed - already_fast.speed;

        assert!(rest_gain > fast_gain);
        assert!(fast_gain > 0.0);
        assert!(fast_result.speed > already_fast.speed);
    }

    #[test]
    fn neutral_acceleration_coasts_toward_rest_without_reversing() {
        let config = SimulationConfig::default();
        let moving_forward = VehicleState {
            position: Vec2::ZERO,
            heading: 0.0,
            speed: 30.0,
        };
        let moving_slowly_in_reverse = VehicleState {
            speed: -5.0,
            ..moving_forward
        };

        let forward_result = integrate_vehicle(moving_forward, CarControls::NEUTRAL, &config, 1.0);
        let reverse_result =
            integrate_vehicle(moving_slowly_in_reverse, CarControls::NEUTRAL, &config, 1.0);

        assert_eq!(
            forward_result.speed,
            moving_forward.speed - config.coasting_deceleration
        );
        assert_eq!(reverse_result.speed, 0.0);
    }

    #[test]
    fn opposing_acceleration_keeps_full_braking_rate() {
        let config = SimulationConfig::default();
        let initial = VehicleState {
            position: Vec2::ZERO,
            heading: 0.0,
            speed: 500.0,
        };
        let result = integrate_vehicle(initial, CarControls::new(-1.0, 0.0), &config, 1.0);
        assert!((initial.speed - result.speed - config.braking_rate).abs() < 1.0e-5);
    }

    #[test]
    fn speed_observation_is_asymptotic_and_does_not_clamp_physics() {
        let scale = SimulationConfig::default().speed_normalization_scale;
        assert_eq!(normalize_speed(0.0, scale), 0.0);
        assert!((normalize_speed(scale, scale) - (1.0 - (-1.0_f32).exp())).abs() < 1.0e-6);
        assert!(normalize_speed(scale * 10.0, scale) < 1.0);
    }

    #[test]
    fn switching_test_drive_environment_replaces_player_and_progress() {
        let library = TrackLibrary::load_default().unwrap();
        let track = Track::from_definition(library.definition("interlagos").unwrap()).unwrap();
        let mut app = App::new();
        app.insert_resource(track)
            .insert_resource(SimulationConfig::default())
            .insert_resource(SimulationMode::TestDrive)
            .insert_resource(TestDriveSettings::default())
            .add_systems(Update, rebuild_simulation);
        app.update();

        let first = app
            .world_mut()
            .query_filtered::<Entity, With<ManualCar>>()
            .single(app.world())
            .unwrap();
        assert!(app.world().entity(first).contains::<CarProgress>());

        app.world_mut()
            .resource_mut::<TestDriveSettings>()
            .environment = TestDriveEnvironment::OpenField;
        app.update();

        let second = app
            .world_mut()
            .query_filtered::<Entity, With<ManualCar>>()
            .single(app.world())
            .unwrap();
        assert_ne!(first, second);
        assert!(!app.world().entity(second).contains::<CarProgress>());
        assert_eq!(
            app.world_mut()
                .query_filtered::<Entity, With<ManualCar>>()
                .iter(app.world())
                .count(),
            1
        );
        assert_eq!(
            app.world_mut()
                .query_filtered::<Entity, With<TrackWall>>()
                .iter(app.world())
                .count(),
            0
        );
    }
}
