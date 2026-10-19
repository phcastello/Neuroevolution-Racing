//! Frozen CPU playback. RL owns all physical state; ECS entities are presentation mirrors.
//! No GA simulation, controller, optimizer update or training schedule is installed here.
use std::{path::PathBuf, rc::Rc};

use bevy::{
    asset::AssetPlugin, diagnostic::FrameTimeDiagnosticsPlugin, prelude::*,
    window::WindowResolution,
};
use bevy_egui::{EguiPlugin, input::EguiWantsInput};
use racing_rl::{
    Result,
    backend::{Cpu, matrix, read},
    config::Config,
    ensure,
    network::Network,
    persistence::{self, Checkpoint},
    rollout::{EndReason, Environment},
};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use crate::{
    display::{CameraViewState, DisplayPlugin, DisplaySet},
    rendering::{CarPresentation, RenderingPlugin, RenderingSet, SensorVisualization},
    simulation::{
        Car, CarObservation, CarProgress, KinematicCar, SelectedCar, SimulationConfig, Track,
        TrackLibrary,
        rl_environment::{RacingEnvironment, Scene},
    },
    ui::{
        PpoDashboardPlugin,
        ppo_network_view::{PpoNetworkSettings, PpoNetworkTrace},
    },
};

const TERMINAL_HOLD_SECONDS: f32 = 1.0;

#[cfg(test)]
mod tests;

#[derive(Resource)]
pub(crate) struct ViewerControls {
    pub paused: bool,
    pub speed: f32,
    pub selected: usize,
    pub auto_reset: bool,
    pub restart_requested: bool,
    pub checkpoint_requested: Option<PathBuf>,
    pub track_requested: Option<String>,
    pub network: PpoNetworkSettings,
}

impl Default for ViewerControls {
    fn default() -> Self {
        Self {
            paused: false,
            speed: 1.0,
            selected: 0,
            auto_reset: true,
            restart_requested: false,
            checkpoint_requested: None,
            track_requested: None,
            network: PpoNetworkSettings::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct CarTelemetry {
    pub id: usize,
    pub episode: u64,
    pub elapsed: f32,
    pub physical_ticks: u64,
    pub decisions: u64,
    pub progress: f32,
    pub speed: f32,
    /// Executed deterministic action, in the checkpoint's [steering, acceleration] order.
    pub action: [f32; 2],
    pub reward: f64,
    pub reason: Option<EndReason>,
    pub value: f32,
    pub mean: [f32; 2],
    pub log_std: [f32; 2],
}

impl CarTelemetry {
    fn new(id: usize, environment: &RacingEnvironment) -> Self {
        Self {
            id,
            episode: environment.episode_id(),
            elapsed: 0.0,
            physical_ticks: 0,
            decisions: 0,
            progress: environment.progress_fraction(),
            speed: environment.state.speed,
            action: [0.0; 2],
            reward: 0.0,
            reason: None,
            value: 0.0,
            mean: [0.0; 2],
            log_std: [0.0; 2],
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Pose {
    position: Vec2,
    heading: f32,
}

impl Pose {
    fn of(environment: &RacingEnvironment) -> Self {
        Self {
            position: environment.state.position,
            heading: environment.state.heading,
        }
    }

    fn interpolate(self, next: Self, fraction: f32) -> Self {
        let delta = (next.heading - self.heading + std::f32::consts::PI)
            .rem_euclid(std::f32::consts::TAU)
            - std::f32::consts::PI;
        Self {
            position: self.position.lerp(next.position, fraction),
            heading: self.heading + delta * fraction,
        }
    }
}

pub(crate) struct Viewer {
    pub checkpoint_path: PathBuf,
    pub saved: Checkpoint,
    pub config: Config,
    pub track_id: String,
    pub cars: Vec<CarTelemetry>,
    pub error: Option<String>,
    network: Network<Cpu>,
    environments: Vec<RacingEnvironment>,
    observations: Vec<Vec<f32>>,
    previous_observations: Vec<Vec<f32>>,
    previous: Vec<Pose>,
    terminal_elapsed: Vec<f32>,
    // One trace per car, all captured from the forward that supplied the last executed action.
    // Switching the selected car while paused never fabricates a new decision.
    traces: Vec<PpoNetworkTrace>,
    rendered_frames: u64,
    maximum_frames: Option<u64>,
}

impl Viewer {
    fn new(
        saved: Checkpoint,
        checkpoint_path: PathBuf,
        count: usize,
        track_id: Option<&str>,
        library: &TrackLibrary,
        maximum_frames: Option<u64>,
    ) -> Result<Self> {
        ensure((1..=16).contains(&count), "viewer cars 1..16")?;
        // Use exactly the same restore/validation path as headless evaluation.
        let (network, _, physics) = super::restore::<Cpu>(&saved, &Default::default())?;
        let config = saved.config.clone();
        let track_id = match track_id {
            Some(id) => id.to_owned(),
            None => config
                .evaluation_tracks
                .first()
                .cloned()
                .or_else(|| library.validation_tracks().next().map(|d| d.id.clone()))
                .ok_or_else(|| racing_rl::Error::Invalid("no evaluation track".into()))?,
        };
        let environments = build_environments(library, &track_id, &config, &physics, count)?;
        let cars = environments
            .iter()
            .enumerate()
            .map(|(i, e)| CarTelemetry::new(i, e))
            .collect();
        let previous = environments.iter().map(Pose::of).collect();
        let mut viewer = Self {
            checkpoint_path,
            saved,
            config,
            track_id,
            cars,
            error: None,
            network,
            environments,
            observations: Vec::new(),
            previous_observations: Vec::new(),
            previous,
            terminal_elapsed: vec![0.0; count],
            traces: Vec::new(),
            rendered_frames: 0,
            maximum_frames,
        };
        viewer.observe()?;
        viewer.previous_observations = viewer.observations.clone();
        viewer.capture_inference()?;
        Ok(viewer)
    }

    fn track(&self) -> &Track {
        &self.environments[0].scenes[0].track
    }

    fn observe(&mut self) -> Result<()> {
        self.observations = self
            .environments
            .iter_mut()
            .map(Environment::observe)
            .collect::<Result<_>>()?;
        Ok(())
    }

    pub(crate) fn trace_for(&self, selected: usize) -> Option<&PpoNetworkTrace> {
        self.traces.get(selected)
    }

    /// A real batched forward supplies BOTH the executed policy and its presentation trace.
    fn inference(
        &self,
        observations: &[Vec<f32>],
    ) -> Result<(Vec<[f32; 2]>, Vec<PpoNetworkTrace>)> {
        let input = matrix(
            &observations.iter().flatten().copied().collect::<Vec<_>>(),
            self.cars.len(),
            self.config.observation.dimension(),
            &Default::default(),
        )?;
        let forward = self.network.forward(input)?;
        let actions = read(forward.mean.clone().tanh())?;
        let names: Vec<_> = self.saved.features.iter().map(|f| f.name.clone()).collect();
        let mut result = Vec::with_capacity(self.cars.len());
        let traces = PpoNetworkTrace::batch_from_forward(&forward, &names)?;
        for i in 0..self.cars.len() {
            let action = [actions[i * 2], actions[i * 2 + 1]];
            result.push(action);
        }
        Ok((result, traces))
    }

    fn capture_inference(&mut self) -> Result<Vec<[f32; 2]>> {
        let (actions, traces) = self.inference(&self.observations)?;
        for (i, trace) in traces.into_iter().enumerate() {
            let car = &mut self.cars[i];
            // Preserve final-decision telemetry while displaying the terminal frame.
            if car.reason.is_none() {
                car.mean = trace.mean;
                car.value = trace.value;
                car.log_std = trace.log_std;
                if self.traces.len() <= i {
                    self.traces.push(trace);
                } else {
                    self.traces[i] = trace;
                }
            }
        }
        Ok(actions)
    }

    fn step(&mut self, auto_reset: bool) -> Result<()> {
        let dt = self.config.physical_dt * self.config.action_repeat as f32;
        for i in 0..self.cars.len() {
            if self.cars[i].reason.is_some() {
                self.terminal_elapsed[i] += dt;
                if auto_reset && self.terminal_elapsed[i] >= TERMINAL_HOLD_SECONDS {
                    self.environments[i].reset()?;
                    self.cars[i] = CarTelemetry::new(i, &self.environments[i]);
                    self.previous[i] = Pose::of(&self.environments[i]);
                    self.terminal_elapsed[i] = 0.0;
                }
            }
        }
        self.observe()?;
        self.previous_observations.clone_from(&self.observations);
        let actions = self.capture_inference()?;
        for (i, action) in actions.into_iter().enumerate() {
            if self.cars[i].reason.is_some() {
                continue;
            }
            self.previous[i] = Pose::of(&self.environments[i]);
            let step = self.environments[i].step(action)?;
            let car = &mut self.cars[i];
            car.action = action;
            car.physical_ticks += step.physical_ticks;
            car.decisions += 1;
            car.elapsed = car.physical_ticks as f32 * self.config.physical_dt;
            car.progress = step.progress;
            car.speed = self.environments[i].state.speed;
            car.reward += step.reward.total() as f64;
            car.reason = step.reason;
            self.observations[i] = step.observation;
        }
        Ok(())
    }

    /// Build the replacement completely before exposing it to ECS or egui.
    fn replace_checkpoint(&mut self, path: PathBuf, library: &TrackLibrary) -> Result<()> {
        let saved = persistence::load(&path)?;
        let replacement = Self::new(
            saved,
            path,
            self.cars.len(),
            Some(&self.track_id),
            library,
            self.maximum_frames,
        )?;
        self.commit_replacement(replacement);
        Ok(())
    }

    fn replace_track(&mut self, track_id: &str, library: &TrackLibrary) -> Result<()> {
        let mut environments = build_environments(
            library,
            track_id,
            &self.config,
            &self.environments[0].physics,
            self.cars.len(),
        )?;
        let observations = environments
            .iter_mut()
            .map(Environment::observe)
            .collect::<Result<Vec<_>>>()?;
        let (_, traces) = self.inference(&observations)?;
        // All fallible scene construction, observation and forward finishes before commit.
        // Track selection leaves the loaded checkpoint, parameters and network untouched.
        self.cars = environments
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let mut car = CarTelemetry::new(i, e);
                car.mean = traces[i].mean;
                car.value = traces[i].value;
                car.log_std = traces[i].log_std;
                car
            })
            .collect();
        self.previous = environments.iter().map(Pose::of).collect();
        self.terminal_elapsed.fill(0.0);
        self.environments = environments;
        self.observations = observations;
        self.previous_observations.clone_from(&self.observations);
        self.traces = traces;
        self.track_id = track_id.into();
        self.error = None;
        Ok(())
    }

    fn commit_replacement(&mut self, mut replacement: Self) {
        replacement.rendered_frames = self.rendered_frames;
        *self = replacement;
    }
}

fn build_environments(
    library: &TrackLibrary,
    track_id: &str,
    config: &Config,
    physics: &SimulationConfig,
    count: usize,
) -> Result<Vec<RacingEnvironment>> {
    let definition = library
        .definition(track_id)
        .ok_or_else(|| racing_rl::Error::Invalid(format!("unknown track {track_id}")))?;
    let track =
        Track::from_definition(definition).map_err(|e| racing_rl::Error::Invalid(e.to_string()))?;
    // Interactive inspection may choose training OR held-out tracks. This does not change
    // the evaluation suite or bypass env::scenes' train/evaluation separation in headless jobs.
    let scene = Rc::new(Scene::new(track));
    (0..count)
        .map(|i| {
            RacingEnvironment::new(
                i,
                vec![scene.clone()],
                config,
                physics,
                ChaCha12Rng::seed_from_u64(config.evaluation_seed.wrapping_add(i as u64)),
            )
        })
        .collect()
}

pub(crate) fn view(
    saved: Checkpoint,
    checkpoint_path: PathBuf,
    count: usize,
    track_id: Option<String>,
    maximum_frames: Option<u64>,
) -> Result<()> {
    let library = super::library()?;
    let viewer = Viewer::new(
        saved,
        checkpoint_path,
        count,
        track_id.as_deref(),
        &library,
        maximum_frames,
    )?;
    let track = viewer.track().clone();
    let dt = viewer.config.physical_dt as f64 * viewer.config.action_repeat as f64;
    println!(
        "PPO playback CPU f32 · {} · rollout {} · {} parameters · deterministic tanh(mean)",
        viewer.checkpoint_path.display(),
        viewer.saved.completed_rollouts,
        viewer.saved.config.parameter_count(),
    );
    let exit = App::new()
        .insert_resource(ClearColor(Color::srgb(0.035, 0.075, 0.055)))
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: crate::asset_directory(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Neuroevolution Racing · PPO".into(),
                        resolution: WindowResolution::new(1400, 850),
                        resizable: true,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins((FrameTimeDiagnosticsPlugin::default(), EguiPlugin::default()))
        .insert_non_send(viewer)
        .insert_resource(library)
        .insert_resource(track)
        .insert_resource(Time::<Fixed>::from_seconds(dt))
        .init_resource::<ViewerControls>()
        .add_plugins((DisplayPlugin, RenderingPlugin, PpoDashboardPlugin))
        .add_systems(Startup, spawn_cars)
        // Apply UI commands before any fixed decisions, including when paused.
        .add_systems(PreUpdate, apply_requests.after(bevy::input::InputSystems))
        .add_systems(FixedUpdate, step_playback)
        .add_systems(
            Update,
            mirror_cars
                .before(DisplaySet::Camera)
                .before(RenderingSet::Visuals),
        )
        .add_systems(Last, count_frames)
        .run();
    ensure(
        exit.is_success(),
        format!("PPO viewer exited with {exit:?}"),
    )
}

#[derive(Component)]
struct PlaybackCar;

fn spawn_cars(mut commands: Commands, viewer: NonSend<Viewer>) {
    for car in &viewer.cars {
        let pose = Pose::of(&viewer.environments[car.id]);
        let entity = commands
            .spawn((
                PlaybackCar,
                Car { id: car.id },
                KinematicCar {
                    heading: pose.heading,
                    speed: car.speed,
                },
                CarObservation::INITIAL,
                CarPresentation::Running,
                SensorVisualization::default(),
                Transform::from_translation(pose.position.extend(3.0))
                    .with_rotation(Quat::from_rotation_z(pose.heading)),
            ))
            .id();
        if car.id == 0 {
            commands.entity(entity).insert(SelectedCar);
        }
    }
}

fn apply_requests(
    mut viewer: NonSendMut<Viewer>,
    mut controls: ResMut<ViewerControls>,
    library: Res<TrackLibrary>,
    mut track: ResMut<Track>,
    mut camera: ResMut<CameraViewState>,
    mut fixed: ResMut<Time<Fixed>>,
    mut virtual_time: ResMut<Time<Virtual>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    egui: Res<EguiWantsInput>,
) {
    if !egui.wants_any_keyboard_input() {
        if keyboard.just_pressed(KeyCode::Space) {
            controls.paused = !controls.paused;
        }
        if keyboard.just_pressed(KeyCode::KeyR) {
            controls.restart_requested = true;
        }
    }
    let mut replaced = false;
    if let Some(path) = controls.checkpoint_requested.take() {
        match viewer.replace_checkpoint(path, &library) {
            Ok(()) => replaced = true,
            Err(error) => viewer.error = Some(format!("Checkpoint: {error}")),
        }
    }
    if let Some(id) = controls.track_requested.take() {
        match viewer.replace_track(&id, &library) {
            Ok(()) => replaced = true,
            Err(error) => viewer.error = Some(format!("Track: {error}")),
        }
    }
    if std::mem::take(&mut controls.restart_requested) {
        let id = viewer.track_id.clone();
        match viewer.replace_track(&id, &library) {
            Ok(()) => replaced = true,
            Err(error) => viewer.error = Some(format!("Reset: {error}")),
        }
    }
    if replaced {
        if track.definition.id != viewer.track_id {
            *track = viewer.track().clone();
            camera.request_fit();
        }
        // A different checkpoint can change action_repeat. Discard old overstep as well.
        *fixed = Time::<Fixed>::from_seconds(
            viewer.config.physical_dt as f64 * viewer.config.action_repeat as f64,
        );
    }
    controls.selected = controls.selected.min(viewer.cars.len() - 1);
    controls.speed = controls.speed.clamp(0.1, 8.0);
    virtual_time.set_relative_speed(controls.speed);
    if controls.paused {
        virtual_time.pause();
    } else {
        virtual_time.unpause();
    }
}

fn step_playback(mut viewer: NonSendMut<Viewer>, mut controls: ResMut<ViewerControls>) {
    // FixedUpdate may already be scheduled when a pause request reaches this frame.
    if controls.paused {
        return;
    }
    if let Err(error) = viewer.step(controls.auto_reset) {
        viewer.error = Some(format!("Playback: {error}"));
        controls.paused = true;
    }
}

fn mirror_cars(
    mut commands: Commands,
    viewer: NonSend<Viewer>,
    controls: Res<ViewerControls>,
    fixed: Res<Time<Fixed>>,
    mut cars: Query<
        (
            Entity,
            &Car,
            &mut Transform,
            &mut KinematicCar,
            &mut CarObservation,
            &mut CarPresentation,
            &mut SensorVisualization,
            Option<&SelectedCar>,
        ),
        With<PlaybackCar>,
    >,
) {
    let angles = viewer.config.observation.angles();
    for (
        entity,
        car,
        mut transform,
        mut kinematic,
        mut observation,
        mut presentation,
        mut sensors,
        selected,
    ) in &mut cars
    {
        let telemetry = &viewer.cars[car.id];
        let environment = &viewer.environments[car.id];
        let current = Pose::of(environment);
        let fraction = if controls.paused || telemetry.reason.is_some() || telemetry.decisions == 0
        {
            1.0
        } else {
            fixed.overstep_fraction()
        };
        let previous = viewer.previous[car.id];
        let pose = previous.interpolate(current, fraction);
        transform.translation = pose.position.extend(3.0 + car.id as f32 * 0.01);
        transform.rotation = Quat::from_rotation_z(pose.heading);
        kinematic.heading = pose.heading;
        kinematic.speed = telemetry.speed;
        // This generic camera component is never sent to any controller. The real PPO
        // observation remains the environment's complete ordered vector, unmodified.
        observation.normalized_speed =
            (telemetry.speed.abs() / viewer.config.observation.speed_scale).clamp(0.0, 1.0);
        *presentation = match telemetry.reason {
            Some(EndReason::Collision) => CarPresentation::Collided,
            Some(EndReason::Completed) => CarPresentation::Completed,
            _ => CarPresentation::Running,
        };
        let distances = &viewer.observations[car.id][..angles.len()];
        let previous_distances = &viewer.previous_observations[car.id][..angles.len()];
        sensors.normalized.clear();
        sensors.endpoints.clear();
        for ((angle, &distance), &previous_distance) in
            angles.iter().zip(distances).zip(previous_distances)
        {
            let range = viewer.config.observation.sensor_range;
            let before = previous.position
                + Vec2::from_angle(previous.heading + angle.to_radians())
                    * previous_distance
                    * range;
            let after = current.position
                + Vec2::from_angle(current.heading + angle.to_radians()) * distance * range;
            sensors.endpoints.push(before.lerp(after, fraction));
            sensors
                .normalized
                .push(previous_distance + (distance - previous_distance) * fraction);
        }
        if (car.id == controls.selected) != selected.is_some() {
            if car.id == controls.selected {
                commands.entity(entity).insert(SelectedCar);
            } else {
                commands.entity(entity).remove::<SelectedCar>();
            }
        }
        // Only a projection for the shared optional debug overlay; never a policy feature.
        if car.id == controls.selected {
            let projection = viewer.track().project(environment.state.position);
            commands.entity(entity).insert(CarProgress {
                track_distance: projection.track_distance,
                best_track_distance: telemetry.progress * viewer.track().total_length,
                normalized_progress: telemetry.progress,
                nearest_segment: projection.segment_index,
                projected_point: projection.point,
                centerline_distance: projection.track_distance,
            });
        }
    }
}

fn count_frames(mut viewer: NonSendMut<Viewer>, mut exit: MessageWriter<AppExit>) {
    viewer.rendered_frames += 1;
    if viewer
        .maximum_frames
        .is_some_and(|max| viewer.rendered_frames >= max)
    {
        exit.write(AppExit::Success);
    }
}
