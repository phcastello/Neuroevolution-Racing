use bevy::{
    asset::RenderAssetUsages, ecs::system::SystemParam, mesh::Indices, prelude::*,
    render::render_resource::PrimitiveTopology,
};

use crate::simulation::{
    CAR_LENGTH, CAR_WIDTH, Car, CarProgress, SelectedCar, SensorReadings, SimulationMode,
    TestDriveEnvironment, TestDriveSettings, Track, TrackDebug,
};

const WALL_THICKNESS: f32 = 14.0;
const OPEN_FIELD_HALF_SIZE: f32 = 3000.0;
const OPEN_FIELD_GRID_SPACING: f32 = 100.0;
const OPEN_FIELD_GRID_LINE_COUNT: i32 = 30;

pub struct RenderingPlugin;

impl Plugin for RenderingPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            Update,
            (rebuild_track_visuals, decorate_new_cars, draw_debug_visuals).chain(),
        );
    }
}

#[derive(Component)]
struct TrackVisual;

type TrackVisualAssets<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        Option<&'static Mesh2d>,
        Option<&'static MeshMaterial2d<ColorMaterial>>,
    ),
    With<TrackVisual>,
>;

#[derive(SystemParam)]
struct TrackVisualBuildData<'w, 's> {
    commands: Commands<'w, 's>,
    track: Res<'w, Track>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<ColorMaterial>>,
    old_visuals: TrackVisualAssets<'w, 's>,
    mode: Res<'w, SimulationMode>,
    test_drive: Res<'w, TestDriveSettings>,
    last_environment: Local<'s, Option<(SimulationMode, TestDriveEnvironment)>>,
}

fn rebuild_track_visuals(data: TrackVisualBuildData) {
    let TrackVisualBuildData {
        mut commands,
        track,
        mut meshes,
        mut materials,
        old_visuals,
        mode,
        test_drive,
        mut last_environment,
    } = data;
    let current_environment = (*mode, test_drive.environment);
    if !track.is_changed() && *last_environment == Some(current_environment) {
        return;
    }
    *last_environment = Some(current_environment);
    for (entity, mesh, material) in &old_visuals {
        if let Some(mesh) = mesh {
            meshes.remove(mesh.0.id());
        }
        if let Some(material) = material {
            materials.remove(material.0.id());
        }
        commands.entity(entity).despawn();
    }

    if *mode == SimulationMode::TestDrive
        && test_drive.environment == TestDriveEnvironment::OpenField
    {
        spawn_open_field_grid(&mut commands);
        return;
    }

    commands.spawn((
        TrackVisual,
        Mesh2d(meshes.add(road_ribbon_mesh(&track))),
        MeshMaterial2d(materials.add(Color::srgb(0.12, 0.14, 0.16))),
        Transform::from_xyz(0.0, 0.0, -2.0),
    ));

    for (start, end) in
        closed_segments(&track.left_border).chain(closed_segments(&track.right_border))
    {
        spawn_segment(
            &mut commands,
            start,
            end,
            WALL_THICKNESS,
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

    // The first sample is the single start/finish location for the one-lap task.
    spawn_segment(
        &mut commands,
        track.left_border[0],
        track.right_border[0],
        6.0,
        Color::srgb(0.95, 0.95, 0.95),
        0.0,
    );
}

fn road_ribbon_mesh(track: &Track) -> Mesh {
    let sample_count = track.samples.len();
    let mut positions = Vec::with_capacity(sample_count * 2);
    let mut normals = Vec::with_capacity(sample_count * 2);
    let mut uvs = Vec::with_capacity(sample_count * 2);
    let mut indices = Vec::with_capacity(sample_count * 6);

    for index in 0..sample_count {
        positions.push(track.left_border[index].extend(0.0).to_array());
        positions.push(track.right_border[index].extend(0.0).to_array());
        normals.extend([[0.0, 0.0, 1.0]; 2]);
        let along = track.samples[index].cumulative_distance / track.total_length;
        uvs.push([along, 0.0]);
        uvs.push([along, 1.0]);

        let next = (index + 1) % sample_count;
        let left = (index * 2) as u32;
        let right = left + 1;
        let next_left = (next * 2) as u32;
        let next_right = next_left + 1;
        indices.extend([left, right, next_left, next_left, right, next_right]);
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices))
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

fn draw_debug_visuals(
    mut gizmos: Gizmos,
    track: Res<Track>,
    debug: Res<TrackDebug>,
    mode: Res<SimulationMode>,
    test_drive: Res<TestDriveSettings>,
    selected: Query<(&Transform, &SensorReadings, Option<&CarProgress>), With<SelectedCar>>,
) {
    let track_visible =
        *mode != SimulationMode::TestDrive || test_drive.environment == TestDriveEnvironment::Track;
    if !track_visible {
        draw_open_field_grid(&mut gizmos);
    }
    if debug.enabled && track_visible {
        gizmos.linestrip_2d(
            track
                .samples
                .iter()
                .map(|sample| sample.position)
                .chain(std::iter::once(track.samples[0].position)),
            Color::srgba(0.2, 0.9, 1.0, 0.75),
        );
        for sample in track.samples.iter().step_by(8) {
            gizmos.circle_2d(sample.position, 2.5, Color::srgb(0.95, 0.85, 0.2));
        }
        for control_point in &track.control_points {
            gizmos.circle_2d(*control_point, 8.0, Color::srgb(1.0, 0.35, 0.08));
            gizmos.circle_2d(*control_point, 3.0, Color::srgb(1.0, 0.92, 0.2));
        }
    }

    for (transform, sensors, progress) in &selected {
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

        if debug.enabled
            && let Some(progress) = progress
        {
            gizmos.line_2d(origin, progress.projected_point, Color::srgb(1.0, 0.2, 0.9));
            gizmos.circle_2d(progress.projected_point, 5.0, Color::srgb(1.0, 0.2, 0.9));
        }
    }
}

fn draw_open_field_grid(gizmos: &mut Gizmos) {
    for index in -OPEN_FIELD_GRID_LINE_COUNT..=OPEN_FIELD_GRID_LINE_COUNT {
        let coordinate = index as f32 * OPEN_FIELD_GRID_SPACING;
        let color = if index == 0 {
            Color::srgba(0.35, 1.0, 0.78, 1.0)
        } else if index % 5 == 0 {
            Color::srgba(0.48, 0.78, 0.68, 0.95)
        } else {
            Color::srgba(0.30, 0.55, 0.48, 0.78)
        };
        gizmos.line_2d(
            Vec2::new(-OPEN_FIELD_HALF_SIZE, coordinate),
            Vec2::new(OPEN_FIELD_HALF_SIZE, coordinate),
            color,
        );
        gizmos.line_2d(
            Vec2::new(coordinate, -OPEN_FIELD_HALF_SIZE),
            Vec2::new(coordinate, OPEN_FIELD_HALF_SIZE),
            color,
        );
    }

    gizmos.circle_2d(Vec2::ZERO, 18.0, Color::srgb(1.0, 0.72, 0.2));
}

fn spawn_open_field_grid(commands: &mut Commands) {
    commands.spawn((
        TrackVisual,
        Sprite {
            color: Color::srgb(0.045, 0.09, 0.075),
            custom_size: Some(Vec2::splat(OPEN_FIELD_HALF_SIZE * 2.0)),
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, -3.0),
    ));
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
        TrackVisual,
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
