use bevy::prelude::*;

#[derive(Component, Debug)]
pub struct Car {
    pub id: usize,
}

#[derive(Component, Debug, Default)]
pub struct SelectedCar;

#[derive(Component, Debug)]
pub struct KinematicCar {
    pub heading: f32,
    pub speed: f32,
    pub previous_position: Vec2,
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
    pub expected_checkpoint: usize,
    pub completed_checkpoints: u64,
    pub laps: u32,
    pub toward_next: f32,
}

impl Default for CarProgress {
    fn default() -> Self {
        Self {
            expected_checkpoint: 1,
            completed_checkpoints: 0,
            laps: 0,
            toward_next: 0.0,
        }
    }
}

#[derive(Component, Clone, Copy, Debug)]
pub struct ControllerTuning {
    pub lane_offset: f32,
    pub steering_bias: f32,
}

#[derive(Component, Debug)]
pub struct TrackWall;
