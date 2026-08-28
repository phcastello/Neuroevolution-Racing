use bevy::{
    camera::{Projection, ScalingMode},
    ecs::system::SystemParam,
    input::mouse::{AccumulatedMouseScroll, MouseScrollUnit},
    prelude::*,
    window::{
        Monitor, MonitorSelection, OnMonitor, PrimaryMonitor, PrimaryWindow, VideoMode,
        VideoModeSelection, WindowMode,
    },
};
use bevy_egui::input::EguiWantsInput;

use crate::simulation::{
    CarObservation, SelectedCar, SimulationMode, TestDriveEnvironment, TestDriveSettings, Track,
    TrackBounds,
};

pub const TOP_BAR_HEIGHT: f32 = 38.0;
pub const DASHBOARD_WIDTH: f32 = 355.0;
const TRACK_MARGIN: f32 = 42.0;
const MIN_VIEWPORT_DIMENSION: u32 = 2;
const MIN_ZOOM: f32 = 0.25;
const MAX_ZOOM: f32 = 24.0;
const ZOOM_SENSITIVITY: f32 = 0.12;
const ROTATION_SPEED: f32 = 1.35;
const SIGNIFICANT_RESIZE_PIXELS: u32 = 4;
const MAX_PAN_DELTA_PER_FRAME: f32 = 120.0;
const FOLLOW_DEFAULT_PROJECTION_SCALE: f32 = 0.55;
const FOLLOW_MAX_SPEED_ZOOM_OUT: f32 = 0.07;
const FOLLOW_SPEED_ZOOM_RESPONSE: f32 = 4.0;
const FOLLOW_LOOK_AHEAD: f32 = 52.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DisplayMode {
    Windowed,
    Fullscreen1080p,
    Fullscreen1440p,
    FullscreenNative,
}

impl DisplayMode {
    fn requested_size(self) -> Option<UVec2> {
        match self {
            Self::Fullscreen1080p => Some(UVec2::new(1920, 1080)),
            Self::Fullscreen1440p => Some(UVec2::new(2560, 1440)),
            Self::Windowed | Self::FullscreenNative => None,
        }
    }
}

#[derive(Resource, Debug)]
pub struct DisplaySettings {
    pub current: DisplayMode,
    pub status: String,
    pending: Option<DisplayMode>,
    windowed_physical_size: UVec2,
}

impl Default for DisplaySettings {
    fn default() -> Self {
        Self {
            current: DisplayMode::Windowed,
            status: "Windowed • F11 toggles native fullscreen".into(),
            pending: None,
            windowed_physical_size: UVec2::new(1400, 850),
        }
    }
}

impl DisplaySettings {
    pub fn request(&mut self, mode: DisplayMode) {
        self.pending = Some(mode);
    }
}

#[derive(Component)]
struct TrackCamera;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CameraBehavior {
    #[default]
    Free,
    FollowCar,
}

#[derive(Resource, Debug)]
pub struct CameraViewState {
    pub center: Vec2,
    pub zoom: f32,
    pub rotation: f32,
    pub behavior: CameraBehavior,
    base_projection_scale: f32,
    free_zoom: f32,
    follow_zoom: f32,
    speed_zoom_multiplier: f32,
    initialized: bool,
    fit_requested: bool,
    reset_requested: bool,
    last_window_size: UVec2,
    last_scale_factor: f32,
    last_scene: Option<(SimulationMode, TestDriveEnvironment)>,
    pan_cursor: Option<Vec2>,
    last_behavior: CameraBehavior,
}

impl Default for CameraViewState {
    fn default() -> Self {
        Self {
            center: Vec2::ZERO,
            zoom: 1.0,
            rotation: 0.0,
            behavior: CameraBehavior::Free,
            base_projection_scale: 1.0,
            free_zoom: 1.0,
            follow_zoom: 1.0,
            speed_zoom_multiplier: 1.0,
            initialized: false,
            fit_requested: true,
            reset_requested: false,
            last_window_size: UVec2::ZERO,
            last_scale_factor: 1.0,
            last_scene: None,
            pan_cursor: None,
            last_behavior: CameraBehavior::Free,
        }
    }
}

impl CameraViewState {
    pub fn request_fit(&mut self) {
        self.fit_requested = true;
    }

    pub fn request_reset(&mut self) {
        self.reset_requested = true;
    }

    pub fn projection_scale(&self) -> f32 {
        self.base_projection_scale / self.zoom
    }

    fn visual_projection_scale(&self) -> f32 {
        self.projection_scale() * self.speed_zoom_multiplier
    }
}

#[derive(SystemParam)]
struct CameraInput<'w, 's> {
    time: Res<'w, Time<Real>>,
    keyboard: Res<'w, ButtonInput<KeyCode>>,
    mouse_buttons: Res<'w, ButtonInput<MouseButton>>,
    mouse_scroll: Res<'w, AccumulatedMouseScroll>,
    egui_input: Res<'w, EguiWantsInput>,
    windows: Query<'w, 's, &'static Window, With<PrimaryWindow>>,
    cameras: Query<'w, 's, (&'static Camera, &'static GlobalTransform), With<TrackCamera>>,
    state: ResMut<'w, CameraViewState>,
}

pub struct DisplayPlugin;

impl Plugin for DisplayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DisplaySettings>()
            .init_resource::<CameraViewState>()
            .add_systems(Startup, setup_camera)
            .add_systems(
                Update,
                (
                    apply_display_requests,
                    handle_camera_input,
                    update_camera_view,
                    apply_camera_view,
                )
                    .chain(),
            );
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn((Camera2d, TrackCamera, Transform::from_xyz(0.0, 0.0, 1000.0)));
}

fn apply_display_requests(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut settings: ResMut<DisplaySettings>,
    mut windows: Query<(&mut Window, Option<&OnMonitor>), With<PrimaryWindow>>,
    monitors: Query<(Entity, &Monitor)>,
    primary_monitors: Query<Entity, With<PrimaryMonitor>>,
) {
    let Ok((mut window, on_monitor)) = windows.single_mut() else {
        return;
    };

    let request = if keyboard.just_pressed(KeyCode::F11) {
        Some(if window.mode == WindowMode::Windowed {
            DisplayMode::FullscreenNative
        } else {
            DisplayMode::Windowed
        })
    } else {
        settings.pending.take()
    };
    let Some(request) = request else {
        return;
    };

    if window.mode == WindowMode::Windowed
        && request != DisplayMode::Windowed
        && window.resolution.physical_width() >= MIN_VIEWPORT_DIMENSION
        && window.resolution.physical_height() >= MIN_VIEWPORT_DIMENSION
    {
        settings.windowed_physical_size = window.resolution.physical_size();
    }

    if request == DisplayMode::Windowed {
        window.mode = WindowMode::Windowed;
        window.resolution.set_physical_resolution(
            settings.windowed_physical_size.x,
            settings.windowed_physical_size.y,
        );
        settings.current = DisplayMode::Windowed;
        settings.status = format!(
            "Windowed {}×{} • F11 toggles fullscreen",
            settings.windowed_physical_size.x, settings.windowed_physical_size.y
        );
        return;
    }

    let monitor_entity = on_monitor
        .map(|monitor| monitor.0)
        .filter(|entity| monitors.get(*entity).is_ok())
        .or_else(|| {
            primary_monitors
                .iter()
                .find(|entity| monitors.get(*entity).is_ok())
        })
        .or_else(|| monitors.iter().next().map(|(entity, _)| entity));

    if request == DisplayMode::FullscreenNative {
        // Ask winit for the monitor at the moment it applies the mode change.
        // The ECS OnMonitor relationship can lag a move by one frame and was
        // selecting the previous monitor under WSLg/XWayland.
        window.mode = WindowMode::BorderlessFullscreen(MonitorSelection::Current);
        settings.current = DisplayMode::FullscreenNative;
        settings.status = "Native borderless fullscreen • F11 restores windowed mode".into();
        return;
    }

    let Some(target) = request.requested_size() else {
        return;
    };
    let Some(monitor_entity) = monitor_entity else {
        window.mode = WindowMode::BorderlessFullscreen(MonitorSelection::Current);
        settings.current = DisplayMode::FullscreenNative;
        settings.status = format!(
            "No monitor modes reported for {}×{}; using native fullscreen",
            target.x, target.y
        );
        return;
    };
    let Ok((_, monitor)) = monitors.get(monitor_entity) else {
        return;
    };
    let Some(video_mode) = select_video_mode(&monitor.video_modes, target) else {
        window.mode = WindowMode::BorderlessFullscreen(MonitorSelection::Current);
        settings.current = DisplayMode::FullscreenNative;
        settings.status = format!(
            "No exclusive modes reported for {}×{}; using native fullscreen",
            target.x, target.y
        );
        return;
    };

    window.mode = WindowMode::Fullscreen(
        MonitorSelection::Current,
        VideoModeSelection::Specific(video_mode),
    );
    settings.current = request;
    if video_mode.physical_size == target {
        settings.status = format!(
            "Fullscreen {}×{} at {:.0} Hz",
            target.x,
            target.y,
            video_mode.refresh_rate_millihertz as f32 / 1000.0
        );
    } else {
        settings.status = format!(
            "{}×{} unavailable; using closest mode {}×{}",
            target.x, target.y, video_mode.physical_size.x, video_mode.physical_size.y
        );
    }
}

fn select_video_mode(modes: &[VideoMode], target: UVec2) -> Option<VideoMode> {
    modes.iter().copied().min_by_key(|mode| {
        let width_difference = mode.physical_size.x.abs_diff(target.x) as u64;
        let height_difference = mode.physical_size.y.abs_diff(target.y) as u64;
        let resolution_distance =
            width_difference * width_difference + height_difference * height_difference;
        (
            resolution_distance,
            u32::MAX - mode.refresh_rate_millihertz,
            u16::MAX - mode.bit_depth,
        )
    })
}

fn handle_camera_input(input: CameraInput) {
    let CameraInput {
        time,
        keyboard,
        mouse_buttons,
        mouse_scroll,
        egui_input,
        windows,
        cameras,
        mut state,
    } = input;
    if !egui_input.wants_any_keyboard_input() {
        let rotation_direction =
            keyboard.pressed(KeyCode::KeyQ) as i8 - keyboard.pressed(KeyCode::KeyE) as i8;
        state.rotation = (state.rotation
            + rotation_direction as f32 * ROTATION_SPEED * time.delta_secs()
            + std::f32::consts::PI)
            .rem_euclid(std::f32::consts::TAU)
            - std::f32::consts::PI;
    }

    let window_and_cursor = windows
        .single()
        .ok()
        .and_then(|window| window.cursor_position().map(|cursor| (window, cursor)));
    let pointer_over_simulation =
        window_and_cursor.is_some_and(|(window, cursor)| cursor_is_over_simulation(window, cursor));
    if egui_input.wants_any_pointer_input() || !pointer_over_simulation {
        state.pan_cursor = None;
        return;
    }

    let pan_pressed =
        mouse_buttons.pressed(MouseButton::Middle) || mouse_buttons.pressed(MouseButton::Right);
    if pan_pressed {
        let cursor = window_and_cursor
            .expect("pointer is inside the simulation view")
            .1;
        if let Some(previous_cursor) = state.pan_cursor {
            let cursor_delta = (cursor - previous_cursor).clamp_length_max(MAX_PAN_DELTA_PER_FRAME);
            let local_drag = Vec2::new(-cursor_delta.x, cursor_delta.y) * state.projection_scale();
            let rotation = state.rotation;
            state.center += Rot2::radians(rotation) * local_drag;
        }
        state.pan_cursor = Some(cursor);
    } else {
        state.pan_cursor = None;
    }

    let scroll_lines = match mouse_scroll.unit {
        MouseScrollUnit::Line => mouse_scroll.delta.y,
        MouseScrollUnit::Pixel => mouse_scroll.delta.y / 50.0,
    };
    if scroll_lines.abs() <= f32::EPSILON {
        return;
    }
    let old_zoom = state.zoom;
    let new_zoom = (old_zoom * (scroll_lines * ZOOM_SENSITIVITY).exp()).clamp(MIN_ZOOM, MAX_ZOOM);
    if (new_zoom - old_zoom).abs() <= f32::EPSILON {
        return;
    }

    let cursor_world = window_and_cursor.and_then(|(_, cursor)| {
        cameras
            .single()
            .ok()
            .and_then(|(camera, transform)| camera.viewport_to_world_2d(transform, cursor).ok())
    });
    if let Some(cursor_world) = cursor_world {
        let scale_ratio = old_zoom / new_zoom;
        state.center = cursor_world - (cursor_world - state.center) * scale_ratio;
    }
    state.zoom = new_zoom;
    match state.behavior {
        CameraBehavior::Free => state.free_zoom = new_zoom,
        CameraBehavior::FollowCar => state.follow_zoom = new_zoom,
    }
}

fn cursor_is_over_simulation(window: &Window, cursor: Vec2) -> bool {
    let simulation_width = (window.width() - DASHBOARD_WIDTH).max(0.0);
    cursor.x >= 0.0
        && cursor.x < simulation_width
        && cursor.y >= TOP_BAR_HEIGHT
        && cursor.y < window.height()
}

fn update_camera_view(
    time: Res<Time<Real>>,
    window: Query<&Window, With<PrimaryWindow>>,
    track: Res<Track>,
    mode: Res<SimulationMode>,
    test_drive: Res<TestDriveSettings>,
    followed_car: Query<(&Transform, &CarObservation), With<SelectedCar>>,
    mut state: ResMut<CameraViewState>,
) {
    let Ok(window) = window.single() else {
        return;
    };
    let window_size = window.resolution.physical_size();
    let scale_factor = window.resolution.scale_factor();
    let scene = (*mode, test_drive.environment);
    let resized = state.last_window_size.x.abs_diff(window_size.x) >= SIGNIFICANT_RESIZE_PIXELS
        || state.last_window_size.y.abs_diff(window_size.y) >= SIGNIFICANT_RESIZE_PIXELS
        || (state.last_scale_factor - scale_factor).abs() > 0.001;
    let scene_changed = state.last_scene != Some(scene);
    state.last_window_size = window_size;
    state.last_scale_factor = scale_factor;
    state.last_scene = Some(scene);

    let behavior_changed = state.behavior != state.last_behavior;
    let entered_follow = behavior_changed && state.behavior == CameraBehavior::FollowCar;
    let exited_follow = behavior_changed && state.behavior == CameraBehavior::Free;
    let reset_requested = state.reset_requested;
    if reset_requested {
        state.rotation = 0.0;
        state.reset_requested = false;
        state.fit_requested = true;
    }

    if !state.initialized || resized || scene_changed || track.is_changed() || state.fit_requested {
        let bounds = if *mode == SimulationMode::TestDrive
            && test_drive.environment == TestDriveEnvironment::OpenField
        {
            TrackBounds {
                min: Vec2::splat(-600.0),
                max: Vec2::splat(600.0),
            }
        } else {
            track.bounds()
        };
        match calculate_camera_fit(bounds, window_size, scale_factor, state.rotation) {
            Some(fit) => {
                state.center = fit.center;
                state.base_projection_scale = fit.projection_scale;
                if state.behavior == CameraBehavior::Free {
                    state.zoom = 1.0;
                    state.free_zoom = 1.0;
                }
                state.initialized = true;
                state.fit_requested = false;
            }
            None => state.initialized = false,
        }
    }

    if entered_follow {
        state.free_zoom = state.zoom;
        state.follow_zoom = follow_default_zoom(state.base_projection_scale);
        state.zoom = state.follow_zoom;
        state.speed_zoom_multiplier = 1.0;
    } else if reset_requested && state.behavior == CameraBehavior::FollowCar {
        state.follow_zoom = follow_default_zoom(state.base_projection_scale);
        state.zoom = state.follow_zoom;
        state.speed_zoom_multiplier = 1.0;
    } else if exited_follow {
        state.follow_zoom = state.zoom;
        state.zoom = state.free_zoom;
        state.speed_zoom_multiplier = 1.0;
    }
    state.last_behavior = state.behavior;

    if state.behavior == CameraBehavior::FollowCar
        && let Ok((car_transform, observation)) = followed_car.single()
    {
        let target_multiplier = speed_visual_multiplier(observation.normalized_speed);
        let blend = 1.0 - (-FOLLOW_SPEED_ZOOM_RESPONSE * time.delta_secs()).exp();
        state.speed_zoom_multiplier +=
            (target_multiplier - state.speed_zoom_multiplier) * blend.clamp(0.0, 1.0);
        let forward = Vec2::from_angle(car_transform.rotation.to_euler(EulerRot::XYZ).2);
        state.center = car_transform.translation.truncate()
            + forward * FOLLOW_LOOK_AHEAD
            + camera_panel_offset(state.visual_projection_scale(), state.rotation);
    }
}

fn apply_camera_view(
    state: Res<CameraViewState>,
    mut cameras: Query<(&mut Camera, &mut Projection, &mut Transform), With<TrackCamera>>,
) {
    for (mut camera, mut projection, mut transform) in &mut cameras {
        camera.is_active = state.initialized;
        camera.viewport = None;
        if !state.initialized {
            continue;
        }
        transform.translation.x = state.center.x;
        transform.translation.y = state.center.y;
        transform.rotation = Quat::from_rotation_z(state.rotation);
        let Projection::Orthographic(orthographic) = &mut *projection else {
            continue;
        };
        orthographic.scaling_mode = ScalingMode::WindowSize;
        orthographic.scale = state.visual_projection_scale();
    }
}

fn follow_default_zoom(base_projection_scale: f32) -> f32 {
    (base_projection_scale / FOLLOW_DEFAULT_PROJECTION_SCALE).clamp(MIN_ZOOM, MAX_ZOOM)
}

fn speed_visual_multiplier(normalized_speed: f32) -> f32 {
    1.0 + normalized_speed.clamp(0.0, 1.0) * FOLLOW_MAX_SPEED_ZOOM_OUT
}

#[derive(Clone, Copy, Debug)]
struct CameraFit {
    center: Vec2,
    projection_scale: f32,
}

fn calculate_camera_fit(
    bounds: TrackBounds,
    window_physical_size: UVec2,
    scale_factor: f32,
    rotation: f32,
) -> Option<CameraFit> {
    if !scale_factor.is_finite() || scale_factor <= f32::EPSILON {
        return None;
    }

    let top_bar_physical = (TOP_BAR_HEIGHT * scale_factor).round() as u32;
    let dashboard_physical = (DASHBOARD_WIDTH * scale_factor).round() as u32;
    let available_width = window_physical_size.x.saturating_sub(dashboard_physical);
    let available_height = window_physical_size.y.saturating_sub(top_bar_physical);
    if available_width < MIN_VIEWPORT_DIMENSION || available_height < MIN_VIEWPORT_DIMENSION {
        return None;
    }

    let simulation_logical =
        Vec2::new(available_width as f32, available_height as f32) / scale_factor;
    if simulation_logical.x <= f32::EPSILON || simulation_logical.y <= f32::EPSILON {
        return None;
    }

    let sine = rotation.sin().abs();
    let cosine = rotation.cos().abs();
    let size = bounds.size();
    let rotated_size = Vec2::new(
        size.x * cosine + size.y * sine,
        size.x * sine + size.y * cosine,
    );
    let padded_size = rotated_size + Vec2::splat(TRACK_MARGIN * 2.0);
    let width_scale = padded_size.x / simulation_logical.x;
    let height_scale = padded_size.y / simulation_logical.y;
    let projection_scale = width_scale.max(height_scale);
    if !projection_scale.is_finite() || projection_scale <= f32::EPSILON {
        return None;
    }

    let center = bounds.center() + camera_panel_offset(projection_scale, rotation);

    Some(CameraFit {
        center,
        projection_scale,
    })
}

fn camera_panel_offset(projection_scale: f32, rotation: f32) -> Vec2 {
    let dashboard_logical = DASHBOARD_WIDTH;
    let top_bar_logical = TOP_BAR_HEIGHT;
    Rot2::radians(rotation)
        * Vec2::new(
            dashboard_logical * projection_scale * 0.5,
            top_bar_logical * projection_scale * 0.5,
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::TrackLibrary;
    use bevy::window::WindowResolution;

    fn bounds() -> TrackBounds {
        TrackBounds {
            min: Vec2::new(-600.0, -400.0),
            max: Vec2::new(600.0, 400.0),
        }
    }

    #[test]
    fn camera_fit_uses_only_the_simulation_area_at_project_resolutions() {
        for resolution in [UVec2::new(1920, 1080), UVec2::new(2560, 1440)] {
            let fit = calculate_camera_fit(bounds(), resolution, 1.0, 0.0).unwrap();
            let simulation_physical_size = UVec2::new(
                resolution.x - DASHBOARD_WIDTH as u32,
                resolution.y - TOP_BAR_HEIGHT as u32,
            );
            let visible_world = simulation_physical_size.as_vec2() * fit.projection_scale;
            let required_world = bounds().size() + Vec2::splat(TRACK_MARGIN * 2.0);

            assert!(visible_world.x + 1e-3 >= required_world.x);
            assert!(visible_world.y + 1e-3 >= required_world.y);

            let window_center = resolution.as_vec2() * 0.5;
            let track_center_on_screen = window_center
                + Vec2::new(
                    (bounds().center().x - fit.center.x) / fit.projection_scale,
                    (fit.center.y - bounds().center().y) / fit.projection_scale,
                );
            let expected_simulation_center = Vec2::new(
                simulation_physical_size.x as f32 * 0.5,
                TOP_BAR_HEIGHT + simulation_physical_size.y as f32 * 0.5,
            );
            assert!(track_center_on_screen.distance(expected_simulation_center) < 1e-3);
        }
    }

    #[test]
    fn camera_fit_contains_every_track_at_supported_and_resized_windows() {
        let library = TrackLibrary::load_default().unwrap();
        for definition in library.all_tracks() {
            let track = Track::from_definition(definition).unwrap();
            for resolution in [
                UVec2::new(1920, 1080),
                UVec2::new(2560, 1440),
                UVec2::new(1180, 720),
            ] {
                let fit = calculate_camera_fit(track.bounds(), resolution, 1.0, 0.0).unwrap();
                let available = Vec2::new(
                    resolution.x as f32 - DASHBOARD_WIDTH,
                    resolution.y as f32 - TOP_BAR_HEIGHT,
                );
                let visible_world = available * fit.projection_scale;
                let required_world = track.bounds().size() + Vec2::splat(TRACK_MARGIN * 2.0);
                assert!(
                    visible_world.x + 1.0e-3 >= required_world.x
                        && visible_world.y + 1.0e-3 >= required_world.y,
                    "camera clipped {} at {resolution:?}",
                    definition.id
                );
            }
        }
    }

    #[test]
    fn rotated_camera_fit_uses_view_oriented_bounds() {
        let rotation = 35.0_f32.to_radians();
        let resolution = UVec2::new(1920, 1080);
        let fit = calculate_camera_fit(bounds(), resolution, 1.0, rotation).unwrap();
        let size = bounds().size();
        let required = Vec2::new(
            size.x * rotation.cos().abs() + size.y * rotation.sin().abs(),
            size.x * rotation.sin().abs() + size.y * rotation.cos().abs(),
        ) + Vec2::splat(TRACK_MARGIN * 2.0);
        let available = Vec2::new(
            resolution.x as f32 - DASHBOARD_WIDTH,
            resolution.y as f32 - TOP_BAR_HEIGHT,
        );
        let visible = available * fit.projection_scale;
        assert!(visible.x + 1.0e-3 >= required.x);
        assert!(visible.y + 1.0e-3 >= required.y);
    }

    #[test]
    fn camera_pointer_controls_are_limited_to_the_simulation_view() {
        let window = Window {
            resolution: WindowResolution::new(1400, 850),
            ..default()
        };
        assert!(cursor_is_over_simulation(&window, Vec2::new(200.0, 200.0)));
        assert!(!cursor_is_over_simulation(
            &window,
            Vec2::new(1390.0, 200.0)
        ));
        assert!(!cursor_is_over_simulation(&window, Vec2::new(200.0, 10.0)));
    }

    #[test]
    fn camera_fit_rejects_zero_and_panel_only_viewports() {
        assert!(calculate_camera_fit(bounds(), UVec2::ZERO, 1.0, 0.0).is_none());
        assert!(
            calculate_camera_fit(
                bounds(),
                UVec2::new(DASHBOARD_WIDTH as u32, TOP_BAR_HEIGHT as u32),
                1.0,
                0.0,
            )
            .is_none()
        );
        assert!(calculate_camera_fit(bounds(), UVec2::new(1920, 1080), 0.0, 0.0).is_none());
    }

    #[test]
    fn follow_default_uses_a_consistent_world_scale() {
        for base_scale in [1.1, 3.0, 5.5] {
            let zoom = follow_default_zoom(base_scale);
            assert!((base_scale / zoom - FOLLOW_DEFAULT_PROJECTION_SCALE).abs() < 1.0e-5);
        }
    }

    #[test]
    fn speed_zoom_out_is_bounded_to_seven_percent() {
        assert_eq!(speed_visual_multiplier(-1.0), 1.0);
        assert_eq!(speed_visual_multiplier(0.0), 1.0);
        assert!((speed_visual_multiplier(0.5) - 1.035).abs() < 1.0e-6);
        assert!((speed_visual_multiplier(1.0) - 1.07).abs() < 1.0e-6);
        assert_eq!(speed_visual_multiplier(2.0), 1.07);
    }

    #[test]
    fn video_mode_prefers_exact_resolution_and_highest_refresh_rate() {
        let modes = [
            VideoMode {
                physical_size: UVec2::new(1920, 1080),
                bit_depth: 24,
                refresh_rate_millihertz: 60_000,
            },
            VideoMode {
                physical_size: UVec2::new(1920, 1080),
                bit_depth: 24,
                refresh_rate_millihertz: 144_000,
            },
            VideoMode {
                physical_size: UVec2::new(2560, 1440),
                bit_depth: 24,
                refresh_rate_millihertz: 60_000,
            },
        ];

        let selected = select_video_mode(&modes, UVec2::new(1920, 1080)).unwrap();
        assert_eq!(selected.physical_size, UVec2::new(1920, 1080));
        assert_eq!(selected.refresh_rate_millihertz, 144_000);
    }

    #[test]
    fn video_mode_uses_the_closest_reported_resolution_when_exact_is_missing() {
        let modes = [
            VideoMode {
                physical_size: UVec2::new(1600, 900),
                bit_depth: 24,
                refresh_rate_millihertz: 60_000,
            },
            VideoMode {
                physical_size: UVec2::new(1920, 1080),
                bit_depth: 24,
                refresh_rate_millihertz: 60_000,
            },
        ];

        let selected = select_video_mode(&modes, UVec2::new(2560, 1440)).unwrap();
        assert_eq!(selected.physical_size, UVec2::new(1920, 1080));
    }
}
