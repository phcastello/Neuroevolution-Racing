use avian2d::prelude::PhysicsLayer;
use bevy::prelude::*;

#[derive(Component, Debug)]
pub struct Car {
    pub id: usize,
}

#[derive(Component, Debug, Default)]
pub struct SelectedCar;

#[derive(Component, Debug, Default)]
pub struct ManualCar;

#[derive(Component, Debug, Default)]
pub struct TemporaryControlled;

#[derive(Component, Debug)]
pub struct KinematicCar {
    pub heading: f32,
    pub speed: f32,
}

#[derive(Component, Clone, Debug)]
pub struct SensorReadings {
    /// Normalized distances ordered left, front-left, front, front-right, right.
    pub normalized: [f32; 5],
    /// World-space endpoints retained for selected-car debug rendering.
    pub endpoints: [Vec2; 5],
}

impl Default for SensorReadings {
    fn default() -> Self {
        Self {
            normalized: [1.0; 5],
            endpoints: [Vec2::ZERO; 5],
        }
    }
}

#[derive(Component, Clone, Debug)]
pub struct CarProgress {
    pub track_distance: f32,
    pub best_track_distance: f32,
    pub normalized_progress: f32,
    pub nearest_segment: usize,
    pub projected_point: Vec2,
    pub(crate) centerline_distance: f32,
}

impl CarProgress {
    pub(crate) fn new(
        track_distance: f32,
        total_track_length: f32,
        nearest_segment: usize,
        projected_point: Vec2,
    ) -> Self {
        Self {
            track_distance,
            best_track_distance: track_distance,
            normalized_progress: (track_distance / total_track_length).clamp(0.0, 1.0),
            nearest_segment,
            projected_point,
            centerline_distance: track_distance,
        }
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct ControllerTuning {
    pub steering_bias: f32,
}

#[derive(Component, Debug)]
pub struct TrackWall;

#[derive(PhysicsLayer, Default)]
pub enum SimulationLayer {
    #[default]
    Car,
    TrackWall,
}

#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct TrackDebug {
    pub enabled: bool,
}
