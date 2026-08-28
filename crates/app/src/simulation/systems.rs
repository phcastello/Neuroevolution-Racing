use std::f32::consts::{FRAC_PI_3, FRAC_PI_6};

use avian2d::prelude::*;
use bevy::prelude::*;
use bevy_egui::input::EguiWantsInput;
use neuroevolution::neural::Mlp;

use crate::simulation::training::{
    EvaluationState, FinishReason, TrackAdvance, TrainingPhase, TrainingState, episode_score,
};

use super::{
    CAR_LENGTH, CAR_WIDTH, ManualControlMode, PlaybackState, SimulationConfig, SimulationMode,
    TestDriveEnvironment, TestDriveSettings, TrackLibrary, TrackSelection,
    components::{
        Car, CarProgress, KinematicCar, ManualCar, SelectedCar, SensorReadings, SimulationLayer,
        TemporaryControlled, TrackWall,
    },
    controller::{CarController, CarControls, CarObservation, MlpController},
    track::{Track, closed_segments, normalized_progress, wrapped_distance_delta},
};

// Array order is left -> right, matching the future MLP input contract.
const SENSOR_ANGLES: [f32; 5] = [FRAC_PI_3, FRAC_PI_6, 0.0, -FRAC_PI_6, -FRAC_PI_3];
const WALL_THICKNESS: f32 = 14.0;
const MIN_GRIP_SPEED: f32 = 10.0;
const CANONICAL_START_DISTANCE: f32 = 16.0;
const CANONICAL_INITIAL_SPEED: f32 = 45.0;
// Preserve the pre-cleanup low-speed steering response independently of the
// controller observation's normalization scale.
const STEERING_RESPONSE_SPEED_SCALE: f32 = 65.0;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimulationSet {
    Sense,
    ControlSource,
    Physics,
    Progress,
    Leader,
    Evaluation,
}

type ReplaceableTrackEntities<'w, 's> = Query<'w, 's, Entity, Or<(With<Car>, With<TrackWall>)>>;
type TemporaryCars<'w, 's> = Query<
    'w,
    's,
    (
        &'static CarObservation,
        &'static mut MlpController,
        &'static mut CarControls,
        Has<SelectedCar>,
        &'static EvaluationState,
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
    mode: Res<SimulationMode>,
) {
    let Some(requested_id) = selection.requested_id.take() else {
        return;
    };
    if *mode == SimulationMode::Training {
        selection.status = "Track selection is managed by the training cycle".into();
        return;
    }
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
    training: Res<TrainingState>,
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
        spawn_temporary_cars(&mut commands, &track, &training, *mode);
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

fn spawn_temporary_cars(
    commands: &mut Commands,
    track: &Track,
    training: &TrainingState,
    mode: SimulationMode,
) {
    let start = canonical_population_start(track);
    match mode {
        SimulationMode::Training => match training.phase() {
            TrainingPhase::TrainingTrack { .. } => {
                for (id, individual) in training.population().individuals().iter().enumerate() {
                    spawn_evaluated_car(
                        commands,
                        training,
                        &start,
                        id,
                        individual.genome().genes(),
                        id == 0,
                    );
                }
            }
            TrainingPhase::Validation { .. } => {
                if let (Some(id), Some(genome)) = (
                    training.champion_population_index(),
                    training.champion_genome(),
                ) {
                    spawn_evaluated_car(commands, training, &start, id, genome, true);
                }
            }
            TrainingPhase::Evolving => {}
        },
        SimulationMode::Champion => {
            if let Some(genome) = training.champion_genome() {
                spawn_evaluated_car(commands, training, &start, 0, genome, true);
            }
        }
        SimulationMode::Race => {
            for (id, individual) in training.population().individuals().iter().enumerate() {
                spawn_evaluated_car(
                    commands,
                    training,
                    &start,
                    id,
                    individual.genome().genes(),
                    id == 0,
                );
            }
        }
        SimulationMode::TestDrive => unreachable!(),
    }
}

fn spawn_evaluated_car(
    commands: &mut Commands,
    training: &TrainingState,
    start: &CanonicalPopulationStart,
    id: usize,
    genome: &[f32],
    selected: bool,
) {
    let mlp = Mlp::from_parameters(training.architecture(), genome).unwrap();
    let controller = MlpController::new(mlp, genome).unwrap();
    let mut entity = commands.spawn((
        Car { id },
        controller,
        KinematicCar {
            heading: start.heading,
            speed: start.speed,
        },
        SensorReadings::default(),
        start.observation,
        start.controls,
        start.progress.clone(),
        EvaluationState::new(start.progress.best_track_distance),
        TemporaryControlled,
        CollisionLayers::new([SimulationLayer::Car], [SimulationLayer::TrackWall]),
        Transform::from_translation(start.position.extend(2.0))
            .with_rotation(Quat::from_rotation_z(start.heading)),
    ));
    if selected {
        entity.insert(SelectedCar);
    }
}

#[derive(Clone, Debug)]
struct CanonicalPopulationStart {
    position: Vec2,
    heading: f32,
    speed: f32,
    controls: CarControls,
    observation: CarObservation,
    progress: CarProgress,
}

fn canonical_population_start(track: &Track) -> CanonicalPopulationStart {
    let position = track.point_at_distance(CANONICAL_START_DISTANCE);
    let projection = track.project(position);
    let tangent = track.samples[projection.segment_index].tangent;
    CanonicalPopulationStart {
        position,
        heading: tangent.y.atan2(tangent.x),
        speed: CANONICAL_INITIAL_SPEED,
        controls: CarControls::NEUTRAL,
        observation: CarObservation::INITIAL,
        progress: CarProgress::new(
            projection.track_distance,
            track.total_length,
            projection.segment_index,
            projection.point,
        ),
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
        CarObservation::INITIAL,
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
    let position = track.point_at_distance(CANONICAL_START_DISTANCE);
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
            Option<&EvaluationState>,
        ),
        With<Car>,
    >,
) {
    let filter = SpatialQueryFilter::from_mask([SimulationLayer::TrackWall]);
    for (transform, state, mut sensors, mut observation, evaluation) in &mut cars {
        if evaluation.is_some_and(EvaluationState::is_finished) {
            continue;
        }
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

pub fn produce_temporary_controls(mut cars: TemporaryCars) {
    for (observation, mut controller, mut controls, selected, evaluation) in &mut cars {
        if evaluation.is_finished() {
            *controls = CarControls::NEUTRAL;
            continue;
        }
        *controls = if selected {
            controller.control_with_telemetry(observation)
        } else {
            controller.control(observation)
        };
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
    if speed.is_nan() {
        return 0.0;
    }
    if speed.is_infinite() {
        return 1.0;
    }

    let magnitude = speed.abs();
    if magnitude == 0.0 {
        return 0.0;
    }
    let scale = if normalization_scale.is_finite() && normalization_scale > 0.0 {
        normalization_scale
    } else {
        f32::EPSILON
    };

    // Algebraically equivalent to magnitude / (magnitude + scale), written
    // this way to avoid overflow for extreme finite inputs.
    (1.0 / (1.0 + scale / magnitude)).clamp(0.0, 1.0)
}

fn steering_speed_ratio(speed: f32) -> f32 {
    (1.0 - (-speed.abs() / STEERING_RESPONSE_SPEED_SCALE).exp()).clamp(0.0, 1.0)
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
    let speed_ratio = steering_speed_ratio(speed).clamp(0.15, 1.0);
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
    mut cars: Query<
        (
            &mut Transform,
            &mut KinematicCar,
            &mut CarControls,
            Option<&mut EvaluationState>,
        ),
        With<Car>,
    >,
) {
    let dt = time.delta_secs();
    let car_shape = Collider::rectangle(CAR_LENGTH, CAR_WIDTH);
    let wall_filter = SpatialQueryFilter::from_mask([SimulationLayer::TrackWall]);

    for (mut transform, mut state, mut controls, mut evaluation) in &mut cars {
        if evaluation
            .as_ref()
            .is_some_and(|evaluation| evaluation.is_finished())
        {
            *controls = CarControls::NEUTRAL;
            state.speed = 0.0;
            continue;
        }
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
            if let Some(evaluation) = evaluation.as_deref_mut() {
                evaluation.finish(FinishReason::Collision);
                *controls = CarControls::NEUTRAL;
                state.speed = 0.0;
            } else {
                state.speed = -integrated.speed.abs().min(18.0) * 0.35;
            }
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
    mut cars: Query<(&Transform, &mut CarProgress, Option<&EvaluationState>), With<Car>>,
) {
    for (transform, mut progress, evaluation) in &mut cars {
        if evaluation.is_some_and(EvaluationState::is_finished) {
            continue;
        }
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

pub fn select_current_leader(
    mut commands: Commands,
    cars: Query<
        (
            Entity,
            &Car,
            &CarProgress,
            &EvaluationState,
            Has<SelectedCar>,
        ),
        With<TemporaryControlled>,
    >,
) {
    let leader = cars
        .iter()
        .max_by(
            |(_, car_a, progress_a, evaluation_a, _), (_, car_b, progress_b, evaluation_b, _)| {
                (!evaluation_a.is_finished())
                    .cmp(&(!evaluation_b.is_finished()))
                    .then_with(|| {
                        progress_a
                            .best_track_distance
                            .total_cmp(&progress_b.best_track_distance)
                    })
                    .then_with(|| car_b.id.cmp(&car_a.id))
            },
        )
        .map(|(entity, _, _, _, _)| entity);

    for (entity, _, _, _, selected) in &cars {
        if Some(entity) == leader && !selected {
            commands.entity(entity).insert(SelectedCar);
        } else if Some(entity) != leader && selected {
            commands.entity(entity).remove::<SelectedCar>();
        }
    }
}

pub fn finish_generation_evaluation(
    time: Res<Time<Fixed>>,
    mut training: ResMut<TrainingState>,
    mut cars: Query<(Entity, &Car, &CarProgress, &mut EvaluationState), With<TemporaryControlled>>,
    mut commands: Commands,
    mut track: ResMut<Track>,
    library: Res<TrackLibrary>,
    mut selection: ResMut<TrackSelection>,
    mode: Res<SimulationMode>,
) {
    if *mode != SimulationMode::Training {
        return;
    }
    let evaluation_config = training.evaluation_config().clone();
    let delta_seconds = time.delta_secs();
    for (_, _, progress, mut evaluation) in &mut cars {
        evaluation.update(
            delta_seconds,
            progress.best_track_distance,
            track.total_length,
            &evaluation_config,
        );
    }
    if cars.is_empty()
        || cars
            .iter()
            .any(|(_, _, _, evaluation)| !evaluation.is_finished())
    {
        return;
    }

    let phase = training.phase().clone();
    let mut scored = cars
        .iter()
        .map(|(entity, car, progress, evaluation)| {
            (
                entity,
                car.id,
                episode_score(
                    &evaluation,
                    progress.best_track_distance,
                    track.total_length,
                    &evaluation_config,
                ),
            )
        })
        .collect::<Vec<_>>();
    scored.sort_by_key(|(_, id, _)| *id);
    for (entity, _, _) in &scored {
        commands.entity(*entity).despawn();
    }
    let advance = match phase {
        TrainingPhase::TrainingTrack { .. } => training
            .record_training_results(
                &scored
                    .iter()
                    .map(|(_, _, result)| *result)
                    .collect::<Vec<_>>(),
            )
            .expect("failed to record training episode scores"),
        TrainingPhase::Validation { .. } => training
            .record_validation_result(scored[0].2)
            .expect("failed to record held-out validation score"),
        TrainingPhase::Evolving => return,
    };

    let next_track_id = match advance {
        TrackAdvance::Training(track_id) | TrackAdvance::Validation(track_id) => track_id,
        TrackAdvance::ReadyToEvolve => {
            let validation = training
                .latest_validation()
                .cloned()
                .expect("validation was just recorded");
            let stats = training
                .evolve_generation()
                .expect("failed to evolve population");
            let counts = training.last_finish_counts();
            println!(
                "Generation {} | Training fitness: best={:.5} average={:.5} | Validation: track={} score={:.5} reason={} | Finish reasons: completed={} collision={} stalled={} timeout={}",
                stats.generation,
                stats.best_fitness,
                stats.average_fitness,
                validation.track_id,
                validation.score,
                validation.finish_reason.label(),
                counts.completed,
                counts.collision,
                counts.stalled,
                counts.timeout,
            );
            training
                .current_track_id()
                .expect("new generation starts on a training track")
                .to_string()
        }
    };
    let definition = library
        .definition(&next_track_id)
        .unwrap_or_else(|| panic!("training requested missing track {next_track_id:?}"));
    *track = Track::from_definition(definition).expect("configured track should build");
    selection.active_id = next_track_id;
    selection.requested_id = None;
    selection.status = format!("Training cycle loaded {}", definition.name);
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

    fn training_state(population_size: usize) -> TrainingState {
        let library = TrackLibrary::load_default().unwrap();
        TrainingState::with_config(
            population_size,
            &library,
            crate::simulation::training::EvaluationConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn selected_car_tracks_the_current_generation_leader() {
        let mut app = App::new();
        app.add_systems(Update, select_current_leader);

        for (id, fitness) in [(0, 12.0), (1, 48.0), (2, 31.0)] {
            let mut progress = CarProgress::new(0.0, 100.0, 0, Vec2::ZERO);
            progress.best_track_distance = fitness;
            let mut entity = app.world_mut().spawn((
                Car { id },
                progress,
                EvaluationState::new(0.0),
                TemporaryControlled,
            ));
            if id == 0 {
                entity.insert(SelectedCar);
            }
        }

        app.update();

        let selected_ids = app
            .world_mut()
            .query_filtered::<&Car, With<SelectedCar>>()
            .iter(app.world())
            .map(|car| car.id)
            .collect::<Vec<_>>();
        assert_eq!(selected_ids, vec![1]);
    }

    #[test]
    fn population_members_share_the_complete_canonical_start_state() {
        let library = TrackLibrary::load_default().unwrap();
        let track = Track::from_definition(library.definition("interlagos").unwrap()).unwrap();
        let mut app = App::new();
        app.insert_resource(track)
            .insert_resource(training_state(6))
            .insert_resource(SimulationConfig {
                population_size: 6,
                ..default()
            })
            .insert_resource(SimulationMode::Training)
            .insert_resource(TestDriveSettings::default())
            .add_systems(Update, rebuild_simulation);
        app.update();

        let expected = canonical_population_start(app.world().resource::<Track>());
        let track_length = app.world().resource::<Track>().total_length;
        let mut ids = Vec::new();
        let mut cars = app.world_mut().query::<(
            &Car,
            &Transform,
            &KinematicCar,
            &CarControls,
            &CarObservation,
            &CarProgress,
        )>();
        for (car, transform, state, controls, observation, progress) in cars.iter(app.world()) {
            ids.push(car.id);
            assert_eq!(transform.translation.truncate(), expected.position);
            assert_eq!(transform.rotation, Quat::from_rotation_z(expected.heading));
            assert_eq!(state.heading, expected.heading);
            assert_eq!(state.speed, CANONICAL_INITIAL_SPEED);
            assert_eq!(state.speed, expected.speed);
            assert_eq!(*controls, expected.controls);
            assert_eq!(*observation, expected.observation);
            assert_eq!(progress.track_distance, expected.progress.track_distance);
            assert_eq!(
                progress.best_track_distance,
                expected.progress.best_track_distance
            );
            assert_eq!(
                progress.normalized_progress,
                expected.progress.normalized_progress
            );
            assert_eq!(progress.nearest_segment, expected.progress.nearest_segment);
            assert_eq!(progress.projected_point, expected.progress.projected_point);
            assert_eq!(
                progress.centerline_distance,
                expected.progress.centerline_distance
            );
            assert!((0.0..=track_length).contains(&progress.track_distance));
        }
        ids.sort_unstable();
        assert_eq!(ids, (0..6).collect::<Vec<_>>());
    }

    #[test]
    fn sensor_angles_and_default_range_match_the_frozen_input_contract() {
        for (actual, expected) in SENSOR_ANGLES
            .map(f32::to_degrees)
            .into_iter()
            .zip([60.0, 30.0, 0.0, -30.0, -60.0])
        {
            assert!((actual - expected).abs() < 1.0e-4);
        }
        assert_eq!(SimulationConfig::default().sensor_max_distance, 750.0);
    }

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
            .insert_resource(training_state(3))
            .insert_resource(selection)
            .insert_resource(SimulationMode::Race)
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
    fn speed_observation_is_zero_and_symmetric() {
        let scale = SimulationConfig::default().speed_normalization_scale;
        assert_eq!(normalize_speed(0.0, scale), 0.0);
        for speed in [1.0, 100.0, 250.0, 519.0] {
            assert_eq!(
                normalize_speed(speed, scale),
                normalize_speed(-speed, scale)
            );
        }
    }

    #[test]
    fn speed_observation_is_monotonic_finite_and_bounded() {
        let scale = SimulationConfig::default().speed_normalization_scale;
        let speeds = [0.0, 1.0, 50.0, 173.0, 259.0, 346.0, 432.0, 519.0, f32::MAX];
        let normalized = speeds.map(|speed| normalize_speed(speed, scale));

        assert!(normalized.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(
            normalized
                .iter()
                .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
        );
        for speed in [f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
            let value = normalize_speed(speed, scale);
            assert!(value.is_finite() && (0.0..=1.0).contains(&value));
        }
        for invalid_scale in [0.0, -1.0, f32::INFINITY, f32::NAN] {
            let value = normalize_speed(100.0, invalid_scale);
            assert!(value.is_finite() && (0.0..=1.0).contains(&value));
        }
    }

    #[test]
    fn racing_speeds_retain_meaningful_normalized_separation() {
        const UNITS_PER_SECOND_TO_KM_H: f32 = 0.57852;
        let scale = SimulationConfig::default().speed_normalization_scale;
        assert_eq!(scale, 250.0);
        let normalized = [150.0, 200.0, 250.0, 300.0]
            .map(|km_h| normalize_speed(km_h / UNITS_PER_SECOND_TO_KM_H, scale));

        assert!(normalized.windows(2).all(|pair| pair[1] - pair[0] > 0.03));
        assert!(normalized[0] > 0.45);
        assert!(normalized[3] < 0.75);
    }

    #[test]
    fn observation_scale_does_not_change_vehicle_physics() {
        let initial = VehicleState {
            position: Vec2::ZERO,
            heading: 0.3,
            speed: 120.0,
        };
        let controls = CarControls::new(0.8, 0.7);
        let default_config = SimulationConfig::default();
        let different_observation_scale = SimulationConfig {
            speed_normalization_scale: 1.0,
            ..default_config.clone()
        };

        assert_eq!(
            integrate_vehicle(initial, controls, &default_config, 1.0 / 60.0),
            integrate_vehicle(initial, controls, &different_observation_scale, 1.0 / 60.0)
        );
    }

    #[test]
    fn switching_test_drive_environment_replaces_player_and_progress() {
        let library = TrackLibrary::load_default().unwrap();
        let track = Track::from_definition(library.definition("interlagos").unwrap()).unwrap();
        let mut app = App::new();
        app.insert_resource(track)
            .insert_resource(training_state(1))
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

    #[test]
    fn finished_population_advances_track_without_evolving_early() {
        let library = TrackLibrary::load_default().unwrap();
        let training = training_state(2);
        let initial_track_id = training.current_track_id().unwrap().to_string();
        let track = Track::from_definition(library.definition(&initial_track_id).unwrap()).unwrap();
        let track_length = track.total_length;
        let selection = TrackSelection {
            active_id: initial_track_id,
            status: String::new(),
            requested_id: None,
        };
        let mut first_progress = CarProgress::new(0.0, track_length, 0, Vec2::ZERO);
        first_progress.best_track_distance = 42.0;
        let mut second_progress = CarProgress::new(0.0, track_length, 0, Vec2::ZERO);
        second_progress.best_track_distance = 73.0;
        let first_evaluation = EvaluationState {
            finish_reason: Some(FinishReason::Collision),
            elapsed: 1.0,
            time_without_progress: 0.0,
            last_significant_progress: 42.0,
            initial_progress: 0.0,
        };
        let second_evaluation = EvaluationState {
            finish_reason: Some(FinishReason::Stalled),
            elapsed: 1.0,
            time_without_progress: 1.0,
            last_significant_progress: 73.0,
            initial_progress: 0.0,
        };

        let mut app = App::new();
        app.insert_resource(training)
            .insert_resource(library)
            .insert_resource(track)
            .insert_resource(selection)
            .insert_resource(SimulationMode::Training)
            .insert_resource(Time::<Fixed>::from_seconds(0.5))
            .add_systems(FixedUpdate, finish_generation_evaluation);
        app.world_mut().spawn((
            Car { id: 0 },
            first_progress,
            first_evaluation,
            TemporaryControlled,
        ));
        app.world_mut().spawn((
            Car { id: 1 },
            second_progress,
            second_evaluation,
            TemporaryControlled,
        ));

        app.world_mut()
            .resource_mut::<Time<Fixed>>()
            .advance_by(std::time::Duration::from_secs_f32(0.5));
        app.world_mut().run_schedule(FixedUpdate);
        let training = app.world().resource::<TrainingState>();
        assert_eq!(training.generation(), 0);
        assert!(training.history().is_empty());
        assert!(
            training
                .population()
                .individuals()
                .iter()
                .all(|individual| individual.fitness().is_none())
        );
        assert!(matches!(
            training.phase(),
            TrainingPhase::TrainingTrack { index: 1, .. }
        ));
    }
}
