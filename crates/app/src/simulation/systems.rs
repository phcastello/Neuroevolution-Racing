use std::f32::consts::{FRAC_PI_2, FRAC_PI_4};

use avian2d::prelude::*;
use bevy::prelude::*;

use super::{
    CAR_LENGTH, CAR_WIDTH, PlaybackState, SimulationConfig,
    components::{
        Car, CarProgress, ControllerTuning, KinematicCar, SelectedCar, SensorReadings, TrackWall,
    },
    controller::{CarController, ControllerInputs, TemporaryController, signed_angle_to},
    track::{Track, closed_segments, crossed_gate},
};

// Array order is left -> right, matching the future MLP input contract.
const SENSOR_ANGLES: [f32; 5] = [FRAC_PI_2, FRAC_PI_4, 0.0, -FRAC_PI_4, -FRAC_PI_2];
const WALL_THICKNESS: f32 = 14.0;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SimulationSet {
    Sense,
    Control,
    Progress,
}

pub fn spawn_track_colliders(mut commands: Commands, track: Res<Track>) {
    for (start, end) in closed_segments(&track.outer_wall).chain(closed_segments(&track.inner_wall))
    {
        let delta = end - start;
        commands.spawn((
            TrackWall,
            RigidBody::Static,
            Collider::rectangle(delta.length(), WALL_THICKNESS),
            Transform::from_translation(((start + end) * 0.5).extend(0.0))
                .with_rotation(Quat::from_rotation_z(delta.y.atan2(delta.x))),
        ));
    }
}

pub fn spawn_cars(mut commands: Commands, track: Res<Track>, config: Res<SimulationConfig>) {
    let start = track.checkpoints[0].center;
    let forward = (track.checkpoints[1].center - start).normalize();
    let heading = forward.y.atan2(forward.x);
    let normal = forward.perp();

    for id in 0..config.population_size {
        let row = (id / 5) as f32;
        let column = (id % 5) as f32 - 2.0;
        let position = start - forward * (20.0 + row * 34.0) + normal * column * 17.0;
        let mut entity = commands.spawn((
            Car { id },
            KinematicCar {
                heading,
                speed: 45.0 + id as f32 * 0.7,
                previous_position: position,
            },
            SensorReadings::default(),
            CarProgress::default(),
            ControllerTuning {
                lane_offset: column * 1.5,
                steering_bias: (column * 0.004).clamp(-0.012, 0.012),
            },
            Transform::from_translation(position.extend(2.0))
                .with_rotation(Quat::from_rotation_z(heading)),
        ));
        if id == 0 {
            entity.insert(SelectedCar);
        }
    }
}

pub fn sample_sensors(
    spatial_query: SpatialQuery,
    config: Res<SimulationConfig>,
    mut cars: Query<(&Transform, &KinematicCar, &mut SensorReadings), With<Car>>,
) {
    let filter = SpatialQueryFilter::default();
    for (transform, state, mut sensors) in &mut cars {
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
    }
}

pub fn drive_cars(
    time: Res<Time<Fixed>>,
    config: Res<SimulationConfig>,
    track: Res<Track>,
    spatial_query: SpatialQuery,
    mut cars: Query<(
        &mut Transform,
        &mut KinematicCar,
        &SensorReadings,
        &CarProgress,
        &ControllerTuning,
    )>,
) {
    let dt = time.delta_secs();
    let car_shape = Collider::rectangle(CAR_LENGTH, CAR_WIDTH);
    let wall_filter = SpatialQueryFilter::default();
    let mut controller = TemporaryController;

    for (mut transform, mut state, sensors, progress, tuning) in &mut cars {
        let position = transform.translation.truncate();
        state.previous_position = position;

        let target = track.checkpoints[progress.expected_checkpoint].center;
        let target_direction = target - position;
        let bearing = signed_angle_to(state.heading, target_direction) + tuning.steering_bias;
        let controls = controller.control(&ControllerInputs {
            sensors: sensors.normalized,
            normalized_speed: state.speed / config.max_speed,
            target_bearing: bearing,
        });

        let acceleration_rate = if controls.acceleration >= 0.0 {
            config.acceleration_rate
        } else {
            config.braking_rate
        };
        state.speed = (state.speed + controls.acceleration * acceleration_rate * dt)
            .clamp(-25.0, config.max_speed);
        let speed_ratio = (state.speed.abs() / config.max_speed).clamp(0.15, 1.0);
        state.heading += controls.steering * config.turn_rate * speed_ratio * dt;

        let forward = Vec2::from_angle(state.heading);
        let proposed = position + forward * state.speed * dt;
        let collided = !spatial_query
            .shape_intersections(&car_shape, proposed, state.heading, &wall_filter)
            .is_empty();

        if collided {
            state.speed = -state.speed.abs().min(18.0) * 0.35;
        } else {
            transform.translation.x = proposed.x;
            transform.translation.y = proposed.y;
        }
        transform.rotation = Quat::from_rotation_z(state.heading);

        // A tiny deterministic lane variation keeps the population visually distinct.
        state.heading += tuning.lane_offset * 0.00002;
    }
}

pub fn update_checkpoint_progress(
    track: Res<Track>,
    mut cars: Query<(&Transform, &KinematicCar, &mut CarProgress), With<Car>>,
) {
    for (transform, state, mut progress) in &mut cars {
        let position = transform.translation.truncate();
        let expected = &track.checkpoints[progress.expected_checkpoint];

        if crossed_gate(state.previous_position, position, expected) {
            let crossed_index = progress.expected_checkpoint;
            progress.completed_checkpoints += 1;
            progress.expected_checkpoint =
                (progress.expected_checkpoint + 1) % track.checkpoints.len();
            if crossed_index == 0 {
                progress.laps += 1;
            }
        }

        let previous_index = if progress.expected_checkpoint == 0 {
            track.checkpoints.len() - 1
        } else {
            progress.expected_checkpoint - 1
        };
        let from = track.checkpoints[previous_index].center;
        let to = track.checkpoints[progress.expected_checkpoint].center;
        let leg_length = from.distance(to).max(1.0);
        progress.toward_next = (1.0 - position.distance(to) / leg_length).clamp(0.0, 1.0);
    }
}

pub fn toggle_pause_from_keyboard(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut playback: ResMut<PlaybackState>,
    mut virtual_time: ResMut<Time<Virtual>>,
) {
    if keyboard.just_pressed(KeyCode::Space) {
        playback.paused = !playback.paused;
        if playback.paused {
            virtual_time.pause();
        } else {
            virtual_time.unpause();
        }
    }
}
