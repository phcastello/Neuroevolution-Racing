use bevy::prelude::*;

use crate::simulation::{
    CAR_LENGTH, CAR_WIDTH, Car, Checkpoint, SelectedCar, SensorReadings, Track,
};

pub struct RenderingPlugin;

impl Plugin for RenderingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (setup_camera, spawn_track_visuals))
            .add_systems(Update, (decorate_new_cars, draw_selected_sensors));
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((Camera2d, Transform::from_xyz(175.0, 0.0, 1000.0)));
}

fn spawn_track_visuals(mut commands: Commands, track: Res<Track>) {
    let centers: Vec<Vec2> = track.checkpoints.iter().map(|gate| gate.center).collect();

    for (start, end) in closed_segments(&centers) {
        spawn_segment(
            &mut commands,
            start,
            end,
            142.0,
            Color::srgb(0.12, 0.14, 0.16),
            -2.0,
        );
        spawn_segment(
            &mut commands,
            start,
            end,
            2.0,
            Color::srgba(0.8, 0.82, 0.7, 0.28),
            -1.0,
        );
    }

    for (index, checkpoint) in track.checkpoints.iter().enumerate() {
        let color = if index == 0 {
            Color::srgb(0.95, 0.95, 0.95)
        } else {
            Color::srgba(0.2, 0.78, 0.95, 0.32)
        };
        spawn_checkpoint(&mut commands, checkpoint, color);
    }

    for (start, end) in closed_segments(&track.outer_wall).chain(closed_segments(&track.inner_wall))
    {
        spawn_segment(
            &mut commands,
            start,
            end,
            14.0,
            Color::srgb(0.78, 0.22, 0.18),
            1.0,
        );
        spawn_segment(
            &mut commands,
            start,
            end,
            5.0,
            Color::srgb(0.98, 0.88, 0.72),
            1.2,
        );
    }
}

fn decorate_new_cars(
    mut commands: Commands,
    cars: Query<(Entity, &Car, Option<&SelectedCar>), Added<Car>>,
) {
    let palette = [
        Color::srgb(0.15, 0.85, 1.0),
        Color::srgb(1.0, 0.68, 0.18),
        Color::srgb(0.72, 0.38, 1.0),
        Color::srgb(0.3, 0.92, 0.42),
        Color::srgb(1.0, 0.35, 0.55),
    ];
    for (entity, car, selected) in &cars {
        let color = if selected.is_some() {
            Color::srgb(0.1, 0.9, 1.0)
        } else {
            palette[car.id % palette.len()]
        };
        commands.entity(entity).insert(Sprite {
            color,
            custom_size: Some(Vec2::new(CAR_LENGTH, CAR_WIDTH)),
            ..default()
        });
    }
}

fn draw_selected_sensors(
    mut gizmos: Gizmos,
    selected: Query<(&Transform, &SensorReadings), With<SelectedCar>>,
) {
    for (transform, sensors) in &selected {
        let origin = transform.translation.truncate();
        for (index, endpoint) in sensors.endpoints.iter().enumerate() {
            let distance = sensors.normalized[index];
            let color = Color::srgb(1.0 - distance * 0.65, 0.3 + distance * 0.7, 0.18);
            gizmos.line_2d(origin, *endpoint, color);
            gizmos.circle_2d(*endpoint, 3.0, color);
        }
        gizmos.rect_2d(
            Isometry2d::new(origin, Rot2::radians(0.0)),
            Vec2::splat(38.0),
            Color::srgba(0.1, 0.9, 1.0, 0.55),
        );
    }
}

fn spawn_checkpoint(commands: &mut Commands, checkpoint: &Checkpoint, color: Color) {
    spawn_segment(
        commands,
        checkpoint.inner,
        checkpoint.outer,
        4.0,
        color,
        0.0,
    );
}

fn spawn_segment(
    commands: &mut Commands,
    start: Vec2,
    end: Vec2,
    thickness: f32,
    color: Color,
    z: f32,
) {
    let delta = end - start;
    commands.spawn((
        Sprite {
            color,
            custom_size: Some(Vec2::new(delta.length(), thickness)),
            ..default()
        },
        Transform::from_translation(((start + end) * 0.5).extend(z))
            .with_rotation(Quat::from_rotation_z(delta.y.atan2(delta.x))),
    ));
}

fn closed_segments(points: &[Vec2]) -> impl Iterator<Item = (Vec2, Vec2)> + '_ {
    points
        .iter()
        .copied()
        .zip(points.iter().copied().cycle().skip(1))
        .take(points.len())
}
