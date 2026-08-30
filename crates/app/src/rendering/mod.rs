use bevy::{
    asset::RenderAssetUsages, ecs::system::SystemParam, mesh::Indices, prelude::*,
    render::render_resource::PrimitiveTopology,
};

use crate::simulation::{
    CAR_LENGTH, CAR_WIDTH, Car, CarProgress, EvaluationState, FinishReason, KinematicCar,
    LaserState, ManualCar, SelectedCar, SensorReadings, SimulationMode, TestDriveEnvironment,
    TestDriveSettings, Track, TrackDebug, TrainingFastForward, TrainingState,
};

const WALL_THICKNESS: f32 = 14.0;
const CURB_TARGET_STRIPE_LENGTH: f32 = 18.0;
const EDGE_MARKER_SPACING: f32 = 44.0;
const EDGE_MARKER_LENGTH: f32 = 10.0;
const EDGE_MARKER_WIDTH: f32 = 3.0;
const EDGE_MARKER_INSET: f32 = 14.0;
const OPEN_FIELD_HALF_SIZE: f32 = 3000.0;
const OPEN_FIELD_GRID_SPACING: f32 = 100.0;
const OPEN_FIELD_GRID_LINE_COUNT: i32 = 30;
const NORMAL_CAR_TINT: Color = Color::srgb(1.0, 1.0, 1.0);
const COLLIDED_CAR_TINT: Color = Color::srgb(0.3, 0.3, 0.3);
// Car 06 was removed from the asset set. Car 03 takes its former slot in the
// ten-car training population, while Car 12 remains Test Drive-only.
const TRAINING_SPRITE_INDICES: [usize; 10] = [0, 1, 3, 4, 2, 5, 6, 7, 8, 9];
const CAR_SPRITE_PATHS: [&str; 11] = [
    "cars/car_01.png",
    "cars/car_02.png",
    "cars/car_03.png",
    "cars/car_04.png",
    "cars/car_05.png",
    "cars/car_07.png",
    "cars/car_08.png",
    "cars/car_09.png",
    "cars/car_10.png",
    "cars/car_11.png",
    "cars/car_12.png",
];

pub const CAR_SPRITE_LABELS: [&str; 11] = [
    "Car 01 - Lavender Sedan",
    "Car 02 - Blue Sedan",
    "Car 03 - Dark SUV",
    "Car 04 - Green Sedan",
    "Car 05 - Lime Sedan",
    "Car 07 - Yellow Sedan",
    "Car 08 - Silver Coupe",
    "Car 09 - Purple Sedan",
    "Car 10 - Slate Coupe",
    "Car 11 - Red Sedan",
    "Car 12 - Orange Van",
];

pub struct RenderingPlugin;

impl Plugin for RenderingPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CarVisualSettings>()
            .add_systems(Startup, load_car_sprite_assets)
            .add_systems(Update, apply_turbo_visibility)
            .add_systems(
                Update,
                (
                    rebuild_track_visuals,
                    decorate_new_cars,
                    update_manual_car_sprite,
                    update_car_collision_tint,
                    draw_debug_visuals,
                )
                    .chain()
                    .run_if(rendering_enabled),
            );
    }
}

#[derive(Resource)]
struct CarSpriteAssets {
    all: [Handle<Image>; CAR_SPRITE_PATHS.len()],
}

impl CarSpriteAssets {
    fn training_sprite(&self, car_id: usize) -> Handle<Image> {
        self.all[TRAINING_SPRITE_INDICES[car_id % TRAINING_SPRITE_INDICES.len()]].clone()
    }

    fn sprite(&self, index: usize) -> Handle<Image> {
        self.all[index.min(self.all.len() - 1)].clone()
    }
}

#[derive(Resource, Clone, Debug)]
pub struct CarVisualSettings {
    pub test_drive_sprite: usize,
    pub show_hitbox: bool,
    pub show_sensors: bool,
}

impl Default for CarVisualSettings {
    fn default() -> Self {
        Self {
            test_drive_sprite: 9,
            show_hitbox: false,
            show_sensors: false,
        }
    }
}

#[derive(Component)]
struct CarVisual;

#[derive(Component)]
struct CarVisualized;

#[derive(Component)]
struct ManualCarVisual;

fn load_car_sprite_assets(mut commands: Commands, asset_server: Res<AssetServer>) {
    let all = std::array::from_fn(|index| asset_server.load(CAR_SPRITE_PATHS[index]));
    commands.insert_resource(CarSpriteAssets { all });
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
    last_track_id: Local<'s, Option<String>>,
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
        mut last_track_id,
    } = data;
    let current_environment = (*mode, test_drive.environment);
    if *last_environment == Some(current_environment)
        && last_track_id.as_deref() == Some(track.definition.id.as_str())
    {
        return;
    }
    *last_environment = Some(current_environment);
    *last_track_id = Some(track.definition.id.clone());
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

    commands.spawn((
        TrackVisual,
        Mesh2d(meshes.add(edge_reference_mesh(&track))),
        MeshMaterial2d(materials.add(Color::srgb(0.68, 0.70, 0.71))),
        Transform::from_xyz(0.0, 0.0, 0.5),
    ));

    for curb in striped_closed_segments(&track.left_border, CURB_TARGET_STRIPE_LENGTH)
        .into_iter()
        .chain(striped_closed_segments(
            &track.right_border,
            CURB_TARGET_STRIPE_LENGTH,
        ))
    {
        let color = if curb.stripe_index % 2 == 0 {
            Color::srgb(0.82, 0.08, 0.06)
        } else {
            Color::srgb(0.96, 0.96, 0.94)
        };
        spawn_segment(
            &mut commands,
            curb.start,
            curb.end,
            WALL_THICKNESS,
            color,
            1.0,
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

fn edge_reference_mesh(track: &Track) -> Mesh {
    let inset_fraction = (EDGE_MARKER_INSET / track.width).clamp(0.0, 0.45);
    let left = track
        .left_border
        .iter()
        .zip(&track.right_border)
        .map(|(left, right)| left.lerp(*right, inset_fraction))
        .collect::<Vec<_>>();
    let right = track
        .right_border
        .iter()
        .zip(&track.left_border)
        .map(|(right, left)| right.lerp(*left, inset_fraction))
        .collect::<Vec<_>>();
    let markers = repeated_closed_segments(&left, EDGE_MARKER_SPACING, EDGE_MARKER_LENGTH)
        .into_iter()
        .chain(repeated_closed_segments(
            &right,
            EDGE_MARKER_SPACING,
            EDGE_MARKER_LENGTH,
        ));
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    let mut indices = Vec::new();

    for (start, end) in markers {
        let delta = end - start;
        let Some(direction) = delta.try_normalize() else {
            continue;
        };
        let half_width = direction.perp() * (EDGE_MARKER_WIDTH * 0.5);
        let first = positions.len() as u32;
        positions.extend([
            (start - half_width).extend(0.0).to_array(),
            (start + half_width).extend(0.0).to_array(),
            (end - half_width).extend(0.0).to_array(),
            (end + half_width).extend(0.0).to_array(),
        ]);
        normals.extend([[0.0, 0.0, 1.0]; 4]);
        uvs.extend([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
        indices.extend([first, first + 1, first + 2, first + 2, first + 1, first + 3]);
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
    assets: Res<CarSpriteAssets>,
    settings: Res<CarVisualSettings>,
    cars: Query<(Entity, &Car, Option<&ManualCar>), Without<CarVisualized>>,
) {
    for (entity, car, manual) in &cars {
        commands
            .entity(entity)
            .insert((Visibility::default(), CarVisualized));
        let image = if manual.is_some() {
            assets.sprite(settings.test_drive_sprite)
        } else {
            assets.training_sprite(car.id)
        };
        let sprite = Sprite {
            image,
            custom_size: Some(Vec2::new(CAR_LENGTH, CAR_WIDTH)),
            ..default()
        };
        if manual.is_some() {
            commands.entity(entity).with_child((
                CarVisual,
                ManualCarVisual,
                sprite,
                Transform::IDENTITY,
            ));
        } else {
            commands
                .entity(entity)
                .with_child((CarVisual, sprite, Transform::IDENTITY));
        }
    }
}

fn rendering_enabled(fast_forward: Res<TrainingFastForward>) -> bool {
    !fast_forward.is_active()
}

fn apply_turbo_visibility(
    fast_forward: Res<TrainingFastForward>,
    mut previous: Local<Option<bool>>,
    mut cars: Query<&mut Visibility, (With<Car>, Without<TrackVisual>)>,
    mut track_visuals: Query<&mut Visibility, (With<TrackVisual>, Without<Car>)>,
) {
    let turbo = fast_forward.is_active();
    if *previous == Some(turbo) {
        return;
    }
    *previous = Some(turbo);
    let visibility = if turbo {
        Visibility::Hidden
    } else {
        Visibility::Inherited
    };
    for mut current in &mut cars {
        *current = visibility;
    }
    for mut current in &mut track_visuals {
        *current = visibility;
    }
}

fn update_manual_car_sprite(
    settings: Res<CarVisualSettings>,
    assets: Res<CarSpriteAssets>,
    mut visuals: Query<&mut Sprite, With<ManualCarVisual>>,
) {
    if !settings.is_changed() {
        return;
    }
    let image = assets.sprite(settings.test_drive_sprite);
    for mut sprite in &mut visuals {
        sprite.image = image.clone();
    }
}

fn car_tint(finish_reason: Option<FinishReason>) -> Color {
    if finish_reason == Some(FinishReason::Collision) {
        COLLIDED_CAR_TINT
    } else {
        NORMAL_CAR_TINT
    }
}

fn update_car_collision_tint(
    cars: Query<(&EvaluationState, &Children), Changed<EvaluationState>>,
    mut visuals: Query<&mut Sprite, With<CarVisual>>,
) {
    for (evaluation, children) in &cars {
        let tint = car_tint(evaluation.finish_reason);
        for child in children.iter() {
            if let Ok(mut sprite) = visuals.get_mut(child) {
                sprite.color = tint;
            }
        }
    }
}

fn selected_car_isometry(transform: &Transform, state: &KinematicCar) -> Isometry2d {
    Isometry2d::new(
        transform.translation.truncate(),
        Rot2::radians(state.heading),
    )
}

fn draw_car_selection_marker(gizmos: &mut Gizmos, transform: &Transform, state: &KinematicCar) {
    gizmos.rect_2d(
        selected_car_isometry(transform, state),
        Vec2::new(CAR_LENGTH + 6.0, CAR_WIDTH + 6.0),
        Color::srgba(0.1, 0.9, 1.0, 0.8),
    );
    let forward = Vec2::from_angle(state.heading);
    let marker_center = transform.translation.truncate() + forward * (CAR_LENGTH * 0.65);
    gizmos.circle_2d(marker_center, 2.4, Color::srgb(0.1, 0.95, 1.0));
}

fn draw_car_hitbox(gizmos: &mut Gizmos, transform: &Transform, state: &KinematicCar) {
    gizmos.rect_2d(
        selected_car_isometry(transform, state),
        Vec2::new(CAR_LENGTH, CAR_WIDTH),
        Color::srgba(1.0, 0.9, 0.15, 0.95),
    );
}

#[cfg(test)]
fn png_dimensions_and_color_type(bytes: &[u8]) -> (u32, u32, u8) {
    assert_eq!(&bytes[0..8], b"\x89PNG\r\n\x1a\n");
    assert_eq!(&bytes[12..16], b"IHDR");
    (
        u32::from_be_bytes(bytes[16..20].try_into().unwrap()),
        u32::from_be_bytes(bytes[20..24].try_into().unwrap()),
        bytes[25],
    )
}

#[cfg(test)]
fn assert_normalized_rgba_png(bytes: &[u8]) {
    let (width, height, color_type) = png_dimensions_and_color_type(bytes);
    assert_eq!((width, height), (280, 150));
    assert_eq!(color_type, 6, "PNG color type must be RGBA");
}

fn draw_debug_visuals(
    mut gizmos: Gizmos,
    track: Res<Track>,
    training: Res<TrainingState>,
    laser: Res<LaserState>,
    debug: Res<TrackDebug>,
    car_visuals: Res<CarVisualSettings>,
    mode: Res<SimulationMode>,
    test_drive: Res<TestDriveSettings>,
    selected: Query<
        (
            &Transform,
            &KinematicCar,
            &SensorReadings,
            Option<&CarProgress>,
        ),
        With<SelectedCar>,
    >,
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

    if *mode == SimulationMode::Training {
        draw_laser(&mut gizmos, &track, &laser, &training);
    }

    for (transform, state, sensors, progress) in &selected {
        let origin = transform.translation.truncate();
        if car_visuals.show_sensors {
            for (index, endpoint) in sensors.endpoints.iter().enumerate() {
                let distance = sensors.normalized[index];
                let color = Color::srgb(1.0 - distance * 0.65, 0.3 + distance * 0.7, 0.18);
                gizmos.line_2d(origin, *endpoint, color);
                gizmos.circle_2d(*endpoint, 3.0, color);
            }
            draw_car_selection_marker(&mut gizmos, transform, state);
        }
        if car_visuals.show_hitbox {
            draw_car_hitbox(&mut gizmos, transform, state);
        }

        if debug.enabled
            && let Some(progress) = progress
        {
            gizmos.line_2d(origin, progress.projected_point, Color::srgb(1.0, 0.2, 0.9));
            gizmos.circle_2d(progress.projected_point, 5.0, Color::srgb(1.0, 0.2, 0.9));
        }
    }
}

fn draw_laser(gizmos: &mut Gizmos, track: &Track, laser: &LaserState, training: &TrainingState) {
    let (center, tangent) = laser_visual_pose(track, laser);
    let transverse = tangent.perp();
    let half_width = track.width * 0.68;
    let color = if laser.elapsed <= training.evaluation_config().laser.grace_period {
        Color::srgba(1.0, 0.15, 0.55, 0.55)
    } else {
        Color::srgb(1.0, 0.05, 0.25)
    };
    for longitudinal_offset in [-2.0, 0.0, 2.0] {
        let offset = tangent * longitudinal_offset;
        gizmos.line_2d(
            center - transverse * half_width + offset,
            center + transverse * half_width + offset,
            color,
        );
    }
    gizmos.circle_2d(center, 5.0, Color::srgb(1.0, 0.85, 0.15));
}

fn laser_visual_pose(track: &Track, laser: &LaserState) -> (Vec2, Vec2) {
    track.pose_at_distance(laser.track_progress())
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

#[derive(Clone, Copy, Debug)]
struct StripedSegment {
    start: Vec2,
    end: Vec2,
    stripe_index: usize,
}

fn striped_closed_segments(points: &[Vec2], target_stripe_length: f32) -> Vec<StripedSegment> {
    let total_length = closed_segments(points)
        .map(|(start, end)| start.distance(end))
        .sum::<f32>();
    let mut stripe_count = (total_length / target_stripe_length).round().max(2.0) as usize;
    if !stripe_count.is_multiple_of(2) {
        stripe_count += 1;
    }
    let stripe_length = total_length / stripe_count as f32;
    let epsilon = stripe_length * 1e-6;

    let mut striped = Vec::with_capacity(points.len() + stripe_count);
    let mut stripe_index = 0;
    let mut distance_in_stripe = 0.0;

    for (start, end) in closed_segments(points) {
        let delta = end - start;
        let segment_length = delta.length();
        let mut distance_in_segment = 0.0;

        while distance_in_segment < segment_length - epsilon {
            let chunk_length =
                (segment_length - distance_in_segment).min(stripe_length - distance_in_stripe);
            let chunk_start = start + delta * (distance_in_segment / segment_length);
            distance_in_segment += chunk_length;
            distance_in_stripe += chunk_length;
            let chunk_end = if segment_length - distance_in_segment <= epsilon {
                end
            } else {
                start + delta * (distance_in_segment / segment_length)
            };
            striped.push(StripedSegment {
                start: chunk_start,
                end: chunk_end,
                stripe_index,
            });

            if stripe_length - distance_in_stripe <= epsilon {
                stripe_index = (stripe_index + 1).min(stripe_count - 1);
                distance_in_stripe = 0.0;
            }
        }
    }

    striped
}

fn repeated_closed_segments(points: &[Vec2], spacing: f32, length: f32) -> Vec<(Vec2, Vec2)> {
    let cumulative = closed_cumulative_distances(points);
    let Some(&total_length) = cumulative.last() else {
        return Vec::new();
    };
    if points.len() < 2 || total_length <= f32::EPSILON || spacing <= 0.0 || length <= 0.0 {
        return Vec::new();
    }
    let marker_count = (total_length / spacing).floor() as usize;
    (0..marker_count)
        .filter_map(|index| {
            let start_distance = index as f32 * spacing;
            let end_distance = (start_distance + length.min(spacing)).min(total_length);
            Some((
                point_on_closed_path(points, &cumulative, start_distance)?,
                point_on_closed_path(points, &cumulative, end_distance)?,
            ))
        })
        .collect()
}

fn closed_cumulative_distances(points: &[Vec2]) -> Vec<f32> {
    let mut cumulative = Vec::with_capacity(points.len() + 1);
    cumulative.push(0.0);
    for (start, end) in closed_segments(points) {
        let next = cumulative.last().copied().unwrap_or_default() + start.distance(end);
        cumulative.push(next);
    }
    cumulative
}

fn point_on_closed_path(points: &[Vec2], cumulative: &[f32], distance: f32) -> Option<Vec2> {
    let total_length = *cumulative.last()?;
    if points.len() < 2 || total_length <= f32::EPSILON {
        return None;
    }
    let distance = distance.clamp(0.0, total_length);
    let segment = cumulative
        .partition_point(|value| *value <= distance)
        .saturating_sub(1)
        .min(points.len() - 1);
    let start_distance = cumulative[segment];
    let segment_length = cumulative[segment + 1] - start_distance;
    let fraction = if segment_length <= f32::EPSILON {
        0.0
    } else {
        (distance - start_distance) / segment_length
    };
    Some(points[segment].lerp(points[(segment + 1) % points.len()], fraction))
}

fn closed_segments(points: &[Vec2]) -> impl Iterator<Item = (Vec2, Vec2)> + '_ {
    points
        .iter()
        .copied()
        .zip(points.iter().copied().cycle().skip(1))
        .take(points.len())
}

#[cfg(test)]
mod car_sprite_tests {
    use super::*;

    #[test]
    fn all_generated_car_assets_are_normalized_rgba_pngs() {
        assert_normalized_rgba_png(include_bytes!("../../../../assets/cars/car_01.png"));
        assert_normalized_rgba_png(include_bytes!("../../../../assets/cars/car_02.png"));
        assert_normalized_rgba_png(include_bytes!("../../../../assets/cars/car_03.png"));
        assert_normalized_rgba_png(include_bytes!("../../../../assets/cars/car_04.png"));
        assert_normalized_rgba_png(include_bytes!("../../../../assets/cars/car_05.png"));
        assert_normalized_rgba_png(include_bytes!("../../../../assets/cars/car_07.png"));
        assert_normalized_rgba_png(include_bytes!("../../../../assets/cars/car_08.png"));
        assert_normalized_rgba_png(include_bytes!("../../../../assets/cars/car_09.png"));
        assert_normalized_rgba_png(include_bytes!("../../../../assets/cars/car_10.png"));
        assert_normalized_rgba_png(include_bytes!("../../../../assets/cars/car_11.png"));
        assert_normalized_rgba_png(include_bytes!("../../../../assets/cars/car_12.png"));
    }

    #[test]
    fn default_training_visual_set_has_ten_unique_entries() {
        assert_eq!(TRAINING_SPRITE_INDICES.len(), 10);
        let unique = TRAINING_SPRITE_INDICES
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(unique.len(), 10);
        assert_eq!(unique, (0..10).collect());
        assert!(!TRAINING_SPRITE_INDICES.contains(&10));
    }

    #[test]
    fn cosmetic_selection_does_not_change_physical_dimensions() {
        let physical_dimensions = (CAR_LENGTH, CAR_WIDTH);
        let settings = CarVisualSettings {
            test_drive_sprite: 11,
            show_hitbox: true,
            show_sensors: true,
        };
        assert_eq!(settings.test_drive_sprite, 11);
        assert!(settings.show_hitbox);
        assert_eq!(physical_dimensions, (CAR_LENGTH, CAR_WIDTH));
    }

    #[test]
    fn only_collision_finish_reason_darkens_car_sprite() {
        assert_eq!(car_tint(Some(FinishReason::Collision)), COLLIDED_CAR_TINT);
        assert_eq!(
            car_tint(Some(FinishReason::EliminatedByLaser)),
            NORMAL_CAR_TINT
        );
        assert_eq!(car_tint(Some(FinishReason::Completed)), NORMAL_CAR_TINT);
        assert_eq!(car_tint(Some(FinishReason::Timeout)), NORMAL_CAR_TINT);
        assert_eq!(car_tint(None), NORMAL_CAR_TINT);
    }

    #[test]
    fn laser_visual_pose_adds_episode_origin_and_relative_progress() {
        let library = crate::simulation::TrackLibrary::load_default().unwrap();
        let track = Track::from_definition(library.definition("interlagos").unwrap()).unwrap();
        let laser = LaserState {
            origin_progress: 16.0,
            progress: 40.0,
            ..default()
        };

        let actual = laser_visual_pose(&track, &laser);
        let expected = track.pose_at_distance(56.0);
        assert!(actual.0.distance(expected.0) < 1.0e-5);
        assert!(actual.1.distance(expected.1) < 1.0e-5);
    }

    #[test]
    fn striped_border_covers_a_closed_loop_and_alternates_at_the_seam() {
        let points = [
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(100.0, 100.0),
            Vec2::new(0.0, 100.0),
        ];
        let striped = striped_closed_segments(&points, 60.0);
        let covered_length = striped
            .iter()
            .map(|segment| segment.start.distance(segment.end))
            .sum::<f32>();

        assert!((covered_length - 400.0).abs() < 1e-3);
        assert_eq!(striped.first().unwrap().start, points[0]);
        assert_eq!(striped.last().unwrap().end, points[0]);
        assert_eq!(striped.first().unwrap().stripe_index, 0);
        assert_eq!(striped.last().unwrap().stripe_index % 2, 1);
        assert!(striped.iter().all(|segment| segment.start != segment.end));
        assert!(striped.windows(2).all(|segments| {
            segments[0].end.distance(segments[1].start) < 1e-4
                && segments[1].stripe_index - segments[0].stripe_index <= 1
        }));
    }

    #[test]
    fn every_bundled_track_gets_complete_stripes_on_both_borders() {
        let library = crate::simulation::TrackLibrary::load_default().unwrap();

        for definition in library.all_tracks() {
            let track = Track::from_definition(definition).unwrap();
            for border in [&track.left_border, &track.right_border] {
                let expected_length = closed_segments(border)
                    .map(|(start, end)| start.distance(end))
                    .sum::<f32>();
                let striped = striped_closed_segments(border, CURB_TARGET_STRIPE_LENGTH);
                let covered_length = striped
                    .iter()
                    .map(|segment| segment.start.distance(segment.end))
                    .sum::<f32>();

                assert!(
                    (covered_length - expected_length).abs() < expected_length * 1.0e-5,
                    "{} curb coverage differs by {}",
                    definition.id,
                    covered_length - expected_length
                );
                assert!(
                    striped
                        .windows(2)
                        .all(|segments| segments[0].end.distance(segments[1].start) < 0.002)
                );
                assert_eq!(striped.first().unwrap().start, border[0]);
                assert_eq!(striped.last().unwrap().end, border[0]);
                assert_eq!(striped.last().unwrap().stripe_index % 2, 1);
            }
        }
    }

    #[test]
    fn edge_references_are_short_and_regular_without_entities_per_marker() {
        let points = [
            Vec2::new(0.0, 0.0),
            Vec2::new(100.0, 0.0),
            Vec2::new(100.0, 100.0),
            Vec2::new(0.0, 100.0),
        ];
        let markers = repeated_closed_segments(&points, 40.0, 8.0);

        assert_eq!(markers.len(), 10);
        assert!(
            markers
                .iter()
                .all(|(start, end)| start.distance(*end) <= 8.0 + 1.0e-4)
        );
        assert_eq!(markers[0], (Vec2::ZERO, Vec2::new(8.0, 0.0)));
        assert_eq!(
            markers[5],
            (Vec2::new(100.0, 100.0), Vec2::new(92.0, 100.0))
        );
    }
}
