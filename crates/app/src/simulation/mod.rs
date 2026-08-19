mod components;
mod controller;
mod systems;
mod track;

pub use components::{Car, CarProgress, KinematicCar, SelectedCar, SensorReadings};
pub use track::{Checkpoint, Track};

use bevy::{prelude::*, time::Fixed};
use serde::{Deserialize, Serialize};
use systems::{
    SimulationSet, drive_cars, sample_sensors, spawn_cars, spawn_track_colliders,
    toggle_pause_from_keyboard, update_checkpoint_progress,
};

pub const CAR_LENGTH: f32 = 28.0;
pub const CAR_WIDTH: f32 = 15.0;

#[derive(Resource, Clone, Debug, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub population_size: usize,
    pub sensor_max_distance: f32,
    pub acceleration_rate: f32,
    pub braking_rate: f32,
    pub max_speed: f32,
    pub turn_rate: f32,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            population_size: 10,
            sensor_max_distance: 155.0,
            acceleration_rate: 125.0,
            braking_rate: 175.0,
            max_speed: 185.0,
            turn_rate: 2.15,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SimulationMode {
    #[default]
    Training,
    Champion,
    Race,
}

#[derive(Resource, Clone, Copy, Debug)]
pub struct PlaybackState {
    pub paused: bool,
    pub speed: f32,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            paused: false,
            speed: 1.0,
        }
    }
}

pub struct SimulationPlugin;

impl Plugin for SimulationPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<SimulationConfig>()
            .init_resource::<SimulationMode>()
            .init_resource::<PlaybackState>()
            .init_resource::<Track>()
            .insert_resource(Time::<Fixed>::from_hz(60.0))
            .configure_sets(
                FixedUpdate,
                (
                    SimulationSet::Sense,
                    SimulationSet::Control,
                    SimulationSet::Progress,
                )
                    .chain(),
            )
            .add_systems(Startup, (spawn_track_colliders, spawn_cars))
            .add_systems(
                FixedUpdate,
                (
                    sample_sensors.in_set(SimulationSet::Sense),
                    drive_cars.in_set(SimulationSet::Control),
                    update_checkpoint_progress.in_set(SimulationSet::Progress),
                ),
            )
            .add_systems(Update, toggle_pause_from_keyboard);
    }
}
