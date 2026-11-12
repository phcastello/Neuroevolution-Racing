//! ACO previews use the same world camera and road meshes as policy playback.
//! Diagnostic paths are never installed as Track resources or physics scenes.
use super::*;
use crate::{
    display::{CameraBehavior, CameraViewState, DisplaySet},
    simulation::TrackBounds,
    ui::search_dashboard::{
        SearchDashboard,
        viewport::{AcoScene, StrokeKind},
    },
    visual::{Mode, Shell},
};

#[derive(Component)]
pub(crate) struct AcoPreviewVisual;
#[derive(Resource, Default)]
pub(crate) struct AcoPreviewBounds(pub Option<TrackBounds>);
#[derive(Resource, Default)]
struct PreviewFrame {
    scene: AcoScene,
    signature: Option<u64>,
    saved_behavior: Option<CameraBehavior>,
}

pub(super) fn install(app: &mut App) {
    app.init_resource::<AcoPreviewBounds>()
        .init_resource::<PreviewFrame>()
        .add_systems(
            Update,
            sync_preview
                .after(crate::visual::apply_navigation)
                .before(DisplaySet::Camera),
        )
        .add_systems(Update, draw_preview.after(sync_preview));
}
#[derive(SystemParam)]
struct PreviewData<'w, 's> {
    commands: Commands<'w, 's>,
    shell: Res<'w, Shell>,
    dashboard: Res<'w, SearchDashboard>,
    frame: ResMut<'w, PreviewFrame>,
    bounds: ResMut<'w, AcoPreviewBounds>,
    camera: ResMut<'w, CameraViewState>,
    meshes: ResMut<'w, Assets<Mesh>>,
    materials: ResMut<'w, Assets<ColorMaterial>>,
    entities: Query<
        'w,
        's,
        (
            Entity,
            Option<&'static Mesh2d>,
            Option<&'static MeshMaterial2d<ColorMaterial>>,
        ),
        With<AcoPreviewVisual>,
    >,
}
fn sync_preview(data: PreviewData) {
    let PreviewData {
        mut commands,
        shell,
        dashboard,
        mut frame,
        mut bounds,
        mut camera,
        mut meshes,
        mut materials,
        entities,
    } = data;
    let active = shell.mode == Mode::Aco;
    let scene = if active {
        dashboard.viewport_scene()
    } else {
        AcoScene::default()
    };
    let signature = active.then(|| scene.signature());
    if signature == frame.signature {
        return;
    }
    for (e, mesh, material) in &entities {
        if let Some(mesh) = mesh {
            meshes.remove(mesh.0.id());
        }
        if let Some(m) = material {
            materials.remove(m.0.id());
        }
        commands.entity(e).despawn();
    }
    if active && frame.saved_behavior.is_none() {
        frame.saved_behavior = Some(camera.behavior);
        camera.behavior = CameraBehavior::Free;
    }
    if !active {
        if let Some(saved) = frame.saved_behavior.take() {
            camera.behavior = saved;
        }
    }
    let mut points: Vec<Vec2> = scene
        .strokes
        .iter()
        .flat_map(|l| &l.points)
        .copied()
        .map(Vec2::from_array)
        .collect();
    points.extend(
        scene
            .issues
            .iter()
            .flat_map(|i| &i.points)
            .copied()
            .map(Vec2::from_array),
    );
    if let Some(d) = &scene.road {
        if let Ok(track) = Track::from_definition(d) {
            points.extend(track.left_border.iter().copied());
            points.extend(track.right_border.iter().copied());
            commands.spawn((
                AcoPreviewVisual,
                Mesh2d(meshes.add(road_ribbon_mesh(&track))),
                MeshMaterial2d(materials.add(Color::srgb(0.12, 0.14, 0.16))),
                Transform::from_xyz(0., 0., -2.),
            ));
            commands.spawn((
                AcoPreviewVisual,
                Mesh2d(meshes.add(edge_reference_mesh(&track))),
                MeshMaterial2d(materials.add(Color::srgb(0.68, 0.70, 0.71))),
                Transform::from_xyz(0., 0., 0.5),
            ));
            // Build visible wall rectangles with exactly the production thickness.
            for curb in striped_closed_segments(&track.left_border, CURB_TARGET_STRIPE_LENGTH)
                .into_iter()
                .chain(striped_closed_segments(
                    &track.right_border,
                    CURB_TARGET_STRIPE_LENGTH,
                ))
            {
                let delta = curb.end - curb.start;
                let color = if curb.stripe_index % 2 == 0 {
                    Color::srgb(0.82, 0.08, 0.06)
                } else {
                    Color::srgb(0.96, 0.96, 0.94)
                };
                commands.spawn((
                    AcoPreviewVisual,
                    Sprite::from_color(color, Vec2::new(delta.length(), WALL_THICKNESS)),
                    Transform::from_translation(((curb.start + curb.end) * 0.5).extend(1.))
                        .with_rotation(Quat::from_rotation_z(delta.y.atan2(delta.x))),
                ));
            }
            let a = track.left_border[0];
            let b = track.right_border[0];
            let delta = b - a;
            commands.spawn((
                AcoPreviewVisual,
                Sprite::from_color(Color::WHITE, Vec2::new(delta.length(), 6.)),
                Transform::from_translation(((a + b) * 0.5).extend(2.))
                    .with_rotation(Quat::from_rotation_z(delta.y.atan2(delta.x))),
            ));
        }
    }
    bounds.0 = if active {
        let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
        for p in points {
            if p.is_finite() {
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        if !lo.is_finite() {
            lo = Vec2::splat(-500.);
            hi = Vec2::splat(500.);
        }
        let center = (lo + hi) * 0.5;
        let half = ((hi - lo) * 0.5).max(Vec2::splat(150.));
        Some(TrackBounds {
            min: center - half,
            max: center + half,
        })
    } else {
        None
    };
    frame.scene = scene;
    frame.signature = signature;
    camera.request_fit();
}
fn draw_preview(shell: Res<Shell>, frame: Res<PreviewFrame>, mut gizmos: Gizmos) {
    if shell.mode != Mode::Aco {
        return;
    }
    for stroke in &frame.scene.strokes {
        let color = match stroke.kind {
            StrokeKind::Path => Color::srgb(0.4, 0.8, 1.),
            StrokeKind::Selected => Color::srgb(1., 0.83, 0.18),
            StrokeKind::Completion => Color::srgb(0.3, 1., 0.5),
            StrokeKind::Rejected => Color::srgba(1., 0.2, 0.17, 0.5),
            StrokeKind::Alternative => Color::srgba(0.65, 0.7, 0.72, 0.45),
            StrokeKind::Population => Color::srgba(0.4, 0.65, 0.6, 0.42),
            StrokeKind::Comparison => Color::srgb(0.3, 0.5, 1.),
        };
        let p: Vec<_> = stroke
            .points
            .iter()
            .copied()
            .map(Vec2::from_array)
            .collect();
        for w in p.windows(2) {
            gizmos.line_2d(w[0], w[1], color);
        }
        if stroke.closed && p.len() > 2 {
            gizmos.line_2d(*p.last().unwrap(), p[0], color);
        }
        if stroke.kind == StrokeKind::Selected {
            if let Some(p) = p.last() {
                gizmos.circle_2d(*p, 8., Color::WHITE);
            }
        }
    }
    for issue in &frame.scene.issues {
        for p in &issue.points {
            gizmos.circle_2d(Vec2::from_array(*p), 18., Color::srgb(1., 0., 0.));
        }
        for pair in issue.points.chunks(2) {
            if pair.len() == 2 {
                gizmos.line_2d(
                    Vec2::from_array(pair[0]),
                    Vec2::from_array(pair[1]),
                    Color::srgb(1., 0., 0.),
                );
            }
        }
    }
}
