mod components;
mod controller;
mod systems;
mod track;
mod training;

pub use components::{
    Car, CarProgress, KinematicCar, ManualCar, SelectedCar, SensorReadings, TrackDebug,
};
pub(crate) use controller::MlpController;
pub use controller::{CarControls, CarObservation};
pub(crate) use systems::{desired_yaw_rate, limited_yaw_rate, max_grip_yaw_rate};
pub use track::{Track, TrackBounds, TrackLibrary};

use bevy::{prelude::*, time::Fixed};
use serde::{Deserialize, Serialize};
use systems::{
    SimulationSet, apply_track_selection, apply_vehicle_physics, finish_generation_evaluation,
    handle_test_drive_input, produce_manual_controls, produce_temporary_controls,
    rebuild_simulation, reset_manual_car, sample_sensors, select_current_leader,
    toggle_pause_from_keyboard, update_track_progress,
};

pub(crate) use training::{EvaluationState, GenerationStats, TrainingPhase, TrainingState};

#[derive(Resource, Clone, Debug)]
pub struct TrackSelection {
    pub active_id: String,
    pub status: String,
    requested_id: Option<String>,
}

impl TrackSelection {
    pub fn request(&mut self, id: impl Into<String>) {
        let id = id.into();
        if id != self.active_id {
            self.requested_id = Some(id);
        }
    }
}

pub const CAR_LENGTH: f32 = 28.0;
pub const CAR_WIDTH: f32 = 15.0;

#[derive(Resource, Clone, Debug, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub population_size: usize,
    pub sensor_max_distance: f32,
    pub acceleration_rate: f32,
    pub braking_rate: f32,
    pub coasting_deceleration: f32,
    pub acceleration_falloff_speed: f32,
    pub speed_normalization_scale: f32,
    pub turn_rate: f32,
    pub max_lateral_acceleration: f32,
    pub progress_search_radius: usize,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            population_size: 500,
            sensor_max_distance: 750.0,
            acceleration_rate: 125.0,
            braking_rate: 175.0,
            coasting_deceleration: 12.0,
            acceleration_falloff_speed: 140.0,
            speed_normalization_scale: 250.0,
            turn_rate: 2.15,
            max_lateral_acceleration: 225.0,
            progress_search_radius: 24,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SimulationMode {
    #[default]
    Training,
    Champion,
    Race,
    TestDrive,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TestDriveEnvironment {
    #[default]
    Track,
    OpenField,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ManualControlMode {
    #[default]
    Keyboard,
    Sliders,
}

#[derive(Resource, Clone, Debug)]
pub struct TestDriveSettings {
    pub environment: TestDriveEnvironment,
    pub control_mode: ManualControlMode,
    pub slider_controls: CarControls,
    pub reset_requested: bool,
}

impl Default for TestDriveSettings {
    fn default() -> Self {
        Self {
            environment: TestDriveEnvironment::Track,
            control_mode: ManualControlMode::Keyboard,
            slider_controls: CarControls::NEUTRAL,
            reset_requested: false,
        }
    }
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
        let library = TrackLibrary::load_default()
            .unwrap_or_else(|error| panic!("failed to load track library: {error}"));
        let available_track_count = library.all_tracks().count();
        let simulation_config = SimulationConfig::default();
        let training_state = TrainingState::with_config(
            simulation_config.population_size,
            &library,
            training::EvaluationConfig::default(),
        )
        .expect("failed to create training state");
        let initial_track_id = training_state
            .current_track_id()
            .expect("training must start on a track");
        let definition = library
            .definition(initial_track_id)
            .unwrap_or_else(|| panic!("initial training track {initial_track_id:?} is missing"));
        let track = Track::from_definition(definition)
            .unwrap_or_else(|error| panic!("failed to build default track: {error}"));
        let selection = TrackSelection {
            active_id: initial_track_id.into(),
            status: format!(
                "Loaded {} • {available_track_count} tracks available",
                definition.name
            ),
            requested_id: None,
        };
        app.insert_resource(simulation_config)
            .insert_resource(training_state)
            .init_resource::<SimulationMode>()
            .init_resource::<PlaybackState>()
            .init_resource::<TestDriveSettings>()
            .init_resource::<TrackDebug>()
            .insert_resource(library)
            .insert_resource(track)
            .insert_resource(selection)
            .insert_resource(Time::<Fixed>::from_hz(60.0))
            .configure_sets(
                FixedUpdate,
                (
                    SimulationSet::Sense,
                    SimulationSet::ControlSource,
                    SimulationSet::Physics,
                    SimulationSet::Progress,
                    SimulationSet::Leader,
                    SimulationSet::Evaluation,
                )
                    .chain(),
            )
            .add_systems(
                FixedUpdate,
                (
                    sample_sensors.in_set(SimulationSet::Sense),
                    (produce_temporary_controls, produce_manual_controls)
                        .in_set(SimulationSet::ControlSource),
                    apply_vehicle_physics.in_set(SimulationSet::Physics),
                    update_track_progress.in_set(SimulationSet::Progress),
                    select_current_leader.in_set(SimulationSet::Leader),
                    finish_generation_evaluation.in_set(SimulationSet::Evaluation),
                ),
            )
            .add_systems(
                Update,
                (
                    apply_track_selection,
                    rebuild_simulation,
                    handle_test_drive_input,
                    reset_manual_car,
                    toggle_pause_from_keyboard,
                )
                    .chain(),
            );
    }
}
