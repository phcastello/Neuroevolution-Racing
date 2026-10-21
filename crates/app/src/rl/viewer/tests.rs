use super::*;
use crate::rl::{SaveContext, checkpoint};
use bevy::time::TimeUpdateStrategy;
use racing_rl::{optimizer::Optimizer, persistence::Rngs};
use std::{
    fs,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "racing-ppo-viewer-tests-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn checkpoint(&self, update: u64, config: &Config) -> PathBuf {
        let path = self.0.join(format!("checkpoint-{update:06}.ron"));
        persistence::save(&path, &saved_policy(config, update)).unwrap();
        path
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        // This directory was created exclusively by this test, under the OS temp directory.
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn small_config() -> Config {
    Config {
        hidden: vec![8, 6],
        num_envs: 2,
        rollout_length: 4,
        minibatch_size: 4,
        action_repeat: 2,
        timeout_ticks: 24,
        ..default()
    }
}

/// Use the application's real persistence builder, with a small deterministic policy.
fn saved_policy(config: &Config, update: u64) -> Checkpoint {
    let mut network = Network::<Cpu>::new(config, &Default::default()).unwrap();
    let mut parameters = network.snapshot().unwrap();
    let actor_bias = config.hidden.len() * 2 + 1;
    parameters[actor_bias] = vec![0.12 + update as f32 * 0.001, 0.4];
    network
        .restore(&parameters, update, config.log_std_bounds)
        .unwrap();
    let mut optimizer = Optimizer::new(config.optimizer.clone(), &network.parameters());
    optimizer.step = update;
    checkpoint(
        &network,
        &optimizer,
        SaveContext {
            config,
            physics: &SimulationConfig::default(),
            rngs: &Rngs::new(config),
            backend: "test-cpu-f32",
            completed_rollouts: update,
            decisions: update * config.rollout_size() as u64,
            physical_ticks: update * config.rollout_size() as u64 * config.action_repeat as u64,
        },
    )
    .unwrap()
}

fn load_viewer(path: &Path, cars: usize, library: &TrackLibrary) -> Viewer {
    Viewer::new(
        persistence::load(path).unwrap(),
        path.to_owned(),
        cars,
        None,
        library,
        None,
    )
    .unwrap()
}

fn assert_close(actual: f32, expected: f32) {
    assert!(
        (actual - expected).abs() <= 2.0e-6,
        "{actual} differs from {expected}",
    );
}

#[derive(Debug, PartialEq)]
struct IntegritySnapshot {
    path: PathBuf,
    config: Config,
    track: String,
    parameters: Vec<Vec<f32>>,
    saved_parameters: Vec<Vec<f32>>,
    observations: Vec<Vec<f32>>,
    previous_observations: Vec<Vec<f32>>,
    previous: Vec<(Vec2, f32)>,
    current: Vec<(Vec2, f32, f32)>,
    terminal_elapsed: Vec<f32>,
    traces: Vec<PpoNetworkTrace>,
    cars: Vec<(u64, u64, u64, f64, [f32; 2], Option<EndReason>)>,
    rendered_frames: u64,
}

fn integrity(viewer: &Viewer) -> IntegritySnapshot {
    IntegritySnapshot {
        path: viewer.checkpoint_path.clone(),
        config: viewer.config.clone(),
        track: viewer.track_id.clone(),
        parameters: viewer.network.snapshot().unwrap(),
        saved_parameters: viewer.saved.parameters.clone(),
        observations: viewer.observations.clone(),
        previous_observations: viewer.previous_observations.clone(),
        previous: viewer
            .previous
            .iter()
            .map(|p| (p.position, p.heading))
            .collect(),
        current: viewer
            .environments
            .iter()
            .map(|e| (e.state.position, e.state.heading, e.state.speed))
            .collect(),
        terminal_elapsed: viewer.terminal_elapsed.clone(),
        traces: viewer.traces.clone(),
        cars: viewer
            .cars
            .iter()
            .map(|c| {
                (
                    c.episode,
                    c.decisions,
                    c.physical_ticks,
                    c.reward,
                    c.action,
                    c.reason,
                )
            })
            .collect(),
        rendered_frames: viewer.rendered_frames,
    }
}

#[test]
fn checkpoint_loading_preserves_architecture_parameters_physics_and_real_trace() {
    let directory = TestDirectory::new();
    let config = small_config();
    let path = directory.checkpoint(25, &config);
    let library = TrackLibrary::load_default().unwrap();
    let viewer = load_viewer(&path, 2, &library);
    assert_eq!(viewer.config, config);
    assert_eq!(viewer.saved.completed_rollouts, 25);
    assert_eq!(viewer.network.version(), 25);
    assert_eq!(viewer.network.snapshot().unwrap(), viewer.saved.parameters);
    assert_eq!(
        viewer
            .network
            .parameters()
            .iter()
            .map(|p| p.dims())
            .collect::<Vec<_>>(),
        viewer.saved.parameter_shapes,
    );
    assert_eq!(
        viewer.saved.parameters.iter().map(Vec::len).sum::<usize>(),
        config.parameter_count()
    );
    assert_eq!(viewer.environments[0].physics, SimulationConfig::default());
    let trace = viewer.trace_for(1).unwrap();
    assert_eq!(trace.input, viewer.observations[1]);
    assert_eq!(trace.input.len(), 25);
    assert_eq!(
        trace.input_names,
        config
            .observation
            .features()
            .iter()
            .map(|f| f.name.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        trace.trunk.iter().map(Vec::len).collect::<Vec<_>>(),
        config.hidden
    );
    assert_eq!(trace.log_std, config.initial_log_std);
    assert_eq!(trace.version, 25);
    assert!(viewer.trace_for(2).is_none());
}

#[test]
fn checkpoint_switch_changes_policy_and_rejects_invalid_replacements_atomically() {
    let directory = TestDirectory::new();
    let config = small_config();
    let first = directory.checkpoint(1, &config);
    let mut next_config = config.clone();
    next_config.hidden = vec![5, 4, 3];
    next_config.action_repeat = 3;
    next_config.initial_log_std = [-0.7, -0.4];
    let second = directory.checkpoint(50, &next_config);
    let library = TrackLibrary::load_default().unwrap();
    let mut viewer = load_viewer(&first, 2, &library);
    viewer.step(false).unwrap();
    let old_action = viewer.cars[0].action;
    viewer.rendered_frames = 42;
    let old_track = viewer.track_id.clone();
    viewer.replace_checkpoint(second.clone(), &library).unwrap();
    assert_eq!(viewer.checkpoint_path, second);
    assert_eq!(viewer.config, next_config);
    assert_eq!(viewer.track_id, old_track);
    assert_eq!(viewer.rendered_frames, 42);
    assert_eq!(viewer.cars.len(), 2);
    assert!(viewer.cars.iter().all(|c| c.decisions == 0
        && c.physical_ticks == 0
        && c.reward == 0.0
        && c.reason.is_none()));
    assert_eq!(viewer.network.snapshot().unwrap(), viewer.saved.parameters);
    assert_eq!(
        viewer
            .trace_for(0)
            .unwrap()
            .trunk
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>(),
        next_config.hidden
    );
    viewer.step(false).unwrap();
    assert_ne!(viewer.cars[0].action, old_action);

    let malformed = directory.0.join("malformed.ron");
    fs::write(&malformed, "not a PPO checkpoint").unwrap();
    let shape_mismatch = directory.0.join("wrong-architecture.ron");
    let mut saved = saved_policy(&config, 100);
    // This remains a valid serialized tensor size, but contradicts W[out,input].
    saved.parameter_shapes[0].swap(0, 1);
    saved.validate().unwrap();
    persistence::save(&shape_mismatch, &saved).unwrap();
    for path in [directory.0.join("missing.ron"), malformed, shape_mismatch] {
        let before = integrity(&viewer);
        assert!(viewer.replace_checkpoint(path, &library).is_err());
        assert_eq!(integrity(&viewer), before);
    }
}

#[test]
fn selecting_training_and_held_out_tracks_and_restart_preserves_policy_and_continuity() {
    let library = TrackLibrary::load_default().unwrap();
    let training = library.training_tracks().next().unwrap().id.clone();
    let held_out = library.validation_tracks().next().unwrap().id.clone();
    let mut viewer = Viewer::new(
        saved_policy(&small_config(), 1),
        "policy.ron".into(),
        1,
        None,
        &library,
        None,
    )
    .unwrap();
    assert_eq!(viewer.track_id, held_out);
    let parameters = viewer.network.snapshot().unwrap();
    viewer.step(false).unwrap();
    viewer.replace_track(&training, &library).unwrap();
    assert_eq!(viewer.track_id, training);
    assert_eq!(viewer.track().definition.id, training);
    assert_eq!(viewer.network.snapshot().unwrap(), parameters);
    assert_eq!(viewer.cars[0].physical_ticks, 0);
    assert_eq!(viewer.observations, viewer.previous_observations);
    assert_eq!(viewer.trace_for(0).unwrap().input, viewer.observations[0]);
    viewer.step(false).unwrap();
    let first_action = viewer.cars[0].action;
    let first_pose = Pose::of(&viewer.environments[0]);
    let first_reward = viewer.cars[0].reward;
    viewer.step(false).unwrap();
    viewer.replace_track(&training, &library).unwrap();
    assert_eq!(viewer.cars[0].decisions, 0);
    assert_eq!(viewer.cars[0].reward, 0.0);
    assert!(viewer.cars[0].reason.is_none());
    viewer.step(false).unwrap();
    assert_eq!(viewer.cars[0].action, first_action);
    assert_eq!(viewer.environments[0].state.position, first_pose.position);
    assert_eq!(viewer.environments[0].state.heading, first_pose.heading);
    assert_eq!(viewer.cars[0].reward, first_reward);
    viewer.replace_track(&held_out, &library).unwrap();
    assert_eq!(viewer.track_id, held_out);
    assert_eq!(viewer.network.snapshot().unwrap(), parameters);
    let before = integrity(&viewer);
    assert!(viewer.replace_track("not-a-real-track", &library).is_err());
    assert_eq!(integrity(&viewer), before);
}

#[test]
fn playback_matches_independent_headless_inference_and_environment_until_terminal() {
    let config = small_config();
    let library = TrackLibrary::load_default().unwrap();
    let mut viewer = Viewer::new(
        saved_policy(&config, 25),
        "policy.ron".into(),
        1,
        None,
        &library,
        None,
    )
    .unwrap();
    let (network, _, physics) =
        crate::rl::restore::<Cpu>(&viewer.saved, &Default::default()).unwrap();
    let mut environment = RacingEnvironment::new(
        0,
        vec![Rc::new(Scene::new(viewer.track().clone()))],
        &config,
        &physics,
        ChaCha12Rng::seed_from_u64(config.evaluation_seed),
    )
    .unwrap();
    let frozen_parameters = viewer.network.snapshot().unwrap();
    let mut reward = 0.0;
    let mut ticks = 0;
    let mut ended = false;
    for decision in 1..=config.timeout_ticks {
        let input = environment.observe().unwrap();
        assert_eq!(input.len(), 25);
        let output = network
            .infer(matrix(&input, 1, input.len(), &Default::default()).unwrap())
            .unwrap();
        let mean = read(output.mean.clone()).unwrap();
        let value = read(output.value).unwrap()[0];
        let action_values = read(output.mean.tanh()).unwrap();
        let action = [action_values[0], action_values[1]];
        let result = environment.step(action).unwrap();
        viewer.step(false).unwrap();
        reward += result.reward.total() as f64;
        ticks += result.physical_ticks;
        let car = &viewer.cars[0];
        let trace = viewer.trace_for(0).unwrap();
        assert_eq!(trace.input, input);
        assert_eq!(viewer.previous_observations[0], input);
        assert_eq!(viewer.observations[0], result.observation);
        assert_close(trace.mean[0], mean[0]);
        assert_close(trace.mean[1], mean[1]);
        assert_close(trace.value, value);
        assert_close(car.action[0], action[0]);
        assert_close(car.action[1], action[1]);
        assert_close(car.value, value);
        assert_close(car.progress, result.progress);
        assert_eq!(car.reward, reward);
        assert_eq!(car.decisions, decision);
        assert_eq!(car.physical_ticks, ticks);
        assert_eq!(car.reason, result.reason);
        assert_eq!(
            viewer.environments[0].state.position,
            environment.state.position
        );
        assert_eq!(
            viewer.environments[0].state.heading,
            environment.state.heading
        );
        if result.terminated || result.truncated {
            ended = true;
            break;
        }
    }
    assert!(
        ended,
        "the bounded episode must reach a real RL terminal condition"
    );
    assert_eq!(viewer.network.snapshot().unwrap(), frozen_parameters);
    assert_eq!(viewer.saved.parameters, frozen_parameters);
    assert_eq!(viewer.saved.policy_version, 25);
    assert_eq!(viewer.saved.optimizer.step, 25);
}

#[test]
fn batched_playback_trace_and_actions_match_each_cars_actual_distinct_observation() {
    let config = small_config();
    let library = TrackLibrary::load_default().unwrap();
    let mut viewer = Viewer::new(
        saved_policy(&config, 1),
        "policy.ron".into(),
        2,
        None,
        &library,
        None,
    )
    .unwrap();
    // Deliberately separate the physical test states to detect a swapped/duplicated trace row.
    viewer.environments[1].state.speed = 80.0;
    viewer.environments[1].state.heading += 0.02;
    let inputs = viewer
        .environments
        .iter_mut()
        .map(Environment::observe)
        .collect::<Result<Vec<_>>>()
        .unwrap();
    assert_ne!(inputs[0], inputs[1]);
    let (network, _, _) = crate::rl::restore::<Cpu>(&viewer.saved, &Default::default()).unwrap();
    let expected = inputs
        .iter()
        .map(|input| {
            let output = network
                .infer(matrix(input, 1, input.len(), &Default::default()).unwrap())
                .unwrap();
            (
                read(output.mean.tanh()).unwrap(),
                read(output.value).unwrap()[0],
            )
        })
        .collect::<Vec<_>>();
    viewer.step(false).unwrap();
    for (id, input) in inputs.iter().enumerate() {
        assert_eq!(viewer.trace_for(id).unwrap().input, *input);
        assert_close(viewer.cars[id].action[0], expected[id].0[0]);
        assert_close(viewer.cars[id].action[1], expected[id].0[1]);
        assert_close(viewer.trace_for(id).unwrap().value, expected[id].1);
    }
}

#[test]
fn terminal_hold_retains_final_pose_trace_reward_and_auto_reset_resumes_real_steps() {
    let config = Config {
        timeout_ticks: 6,
        ..small_config()
    };
    let library = TrackLibrary::load_default().unwrap();
    let mut viewer = Viewer::new(
        saved_policy(&config, 1),
        "policy.ron".into(),
        1,
        None,
        &library,
        None,
    )
    .unwrap();
    while viewer.cars[0].reason.is_none() {
        viewer.step(false).unwrap();
    }
    let final_pose = Pose::of(&viewer.environments[0]);
    let final_car = viewer.cars[0].clone();
    let final_trace = viewer.trace_for(0).unwrap().clone();
    for _ in 0..3 {
        viewer.step(false).unwrap();
        assert_eq!(viewer.cars[0].decisions, final_car.decisions);
        assert_eq!(viewer.cars[0].physical_ticks, final_car.physical_ticks);
        assert_eq!(viewer.cars[0].reward, final_car.reward);
        assert_eq!(viewer.trace_for(0).unwrap(), &final_trace);
        assert_eq!(viewer.environments[0].state.position, final_pose.position);
        assert_eq!(viewer.environments[0].state.heading, final_pose.heading);
    }
    let dt = config.physical_dt * config.action_repeat as f32;
    while viewer.terminal_elapsed[0] + dt < TERMINAL_HOLD_SECONDS {
        viewer.step(true).unwrap();
        assert_eq!(viewer.cars[0].episode, final_car.episode);
        assert_eq!(viewer.cars[0].decisions, final_car.decisions);
    }
    viewer.step(true).unwrap();
    assert_eq!(viewer.cars[0].episode, final_car.episode + 1);
    assert_eq!(viewer.cars[0].decisions, 1);
    assert_eq!(viewer.cars[0].physical_ticks, config.action_repeat as u64);
    assert!(viewer.cars[0].reason.is_none());
    assert_eq!(viewer.terminal_elapsed[0], 0.0);
    assert_eq!(
        viewer.trace_for(0).unwrap().input,
        viewer.previous_observations[0]
    );
}

fn playback_app(viewer: Viewer, library: TrackLibrary) -> App {
    let timestep = viewer.config.physical_dt as f64 * viewer.config.action_repeat as f64;
    let track = viewer.track().clone();
    let mut app = App::new();
    app.add_plugins(MinimalPlugins)
        .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
            timestep,
        )))
        .insert_resource(Time::<Fixed>::from_seconds(timestep))
        .insert_resource(track)
        .insert_resource(library)
        .insert_non_send(viewer)
        .insert_resource(ViewerControls {
            paused: true,
            ..default()
        })
        .init_resource::<CameraViewState>()
        .init_resource::<ButtonInput<KeyCode>>()
        .init_resource::<EguiWantsInput>()
        .add_systems(Startup, spawn_cars)
        .add_systems(PreUpdate, apply_requests)
        .add_systems(FixedUpdate, step_playback)
        .add_systems(Update, mirror_cars);
    app
}

#[test]
fn paused_ui_requests_switch_reset_and_select_without_ga_resources_or_stale_state() {
    let directory = TestDirectory::new();
    let config = small_config();
    let first = directory.checkpoint(1, &config);
    let next_config = Config {
        hidden: vec![7, 3],
        action_repeat: 3,
        ..config.clone()
    };
    let second = directory.checkpoint(100, &next_config);
    let library = TrackLibrary::load_default().unwrap();
    let training = library.training_tracks().next().unwrap().id.clone();
    let viewer = load_viewer(&first, 2, &library);
    let mut app = playback_app(viewer, library);
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(
        app.world().non_send_resource::<Viewer>().cars[0].decisions,
        0
    );
    assert!(app.world().resource::<Time<Virtual>>().is_paused());
    assert!(
        !app.world()
            .contains_resource::<crate::simulation::TrainingState>()
    );
    assert!(
        !app.world()
            .contains_resource::<crate::simulation::SimulationMode>()
    );
    app.world_mut().resource_mut::<ViewerControls>().paused = false;
    for _ in 0..3 {
        app.update();
    }
    assert!(app.world().non_send_resource::<Viewer>().cars[0].decisions > 0);
    app.world_mut().resource_mut::<ViewerControls>().paused = true;
    let before_pause = integrity(app.world().non_send_resource::<Viewer>());
    for _ in 0..3 {
        app.update();
    }
    assert_eq!(
        integrity(app.world().non_send_resource::<Viewer>()),
        before_pause
    );
    {
        let mut controls = app.world_mut().resource_mut::<ViewerControls>();
        controls.checkpoint_requested = Some(second.clone());
        controls.track_requested = Some(training.clone());
        controls.selected = usize::MAX;
        controls.speed = 20.0;
    }
    app.update();
    let viewer = app.world().non_send_resource::<Viewer>();
    assert_eq!(viewer.checkpoint_path, second);
    assert_eq!(viewer.track_id, training);
    assert_eq!(viewer.config, next_config);
    assert!(
        viewer
            .cars
            .iter()
            .all(|c| c.decisions == 0 && c.reason.is_none())
    );
    assert_eq!(app.world().resource::<Track>().definition.id, training);
    assert_eq!(app.world().resource::<ViewerControls>().selected, 1);
    assert_eq!(app.world().resource::<ViewerControls>().speed, 8.0);
    assert_eq!(
        app.world().resource::<Time<Fixed>>().timestep(),
        Duration::from_secs_f64(next_config.physical_dt as f64 * next_config.action_repeat as f64)
    );
    let selected = app
        .world_mut()
        .query_filtered::<&Car, With<SelectedCar>>()
        .iter(app.world())
        .map(|car| car.id)
        .collect::<Vec<_>>();
    assert_eq!(selected, vec![1]);
    app.world_mut()
        .resource_mut::<Time<Fixed>>()
        .accumulate_overstep(Duration::from_millis(10));
    app.world_mut()
        .resource_mut::<ViewerControls>()
        .restart_requested = true;
    app.update();
    assert!(!app.world().resource::<ViewerControls>().restart_requested);
    assert_eq!(
        app.world().resource::<Time<Fixed>>().overstep(),
        Duration::ZERO
    );
    assert_eq!(
        app.world().non_send_resource::<Viewer>().cars[1].decisions,
        0
    );
    let before_invalid = integrity(app.world().non_send_resource::<Viewer>());
    app.world_mut()
        .resource_mut::<ViewerControls>()
        .checkpoint_requested = Some(directory.0.join("missing.ron"));
    app.update();
    assert_eq!(
        integrity(app.world().non_send_resource::<Viewer>()),
        before_invalid
    );
    assert!(
        app.world()
            .non_send_resource::<Viewer>()
            .error
            .as_deref()
            .is_some_and(|e| e.contains("Checkpoint"))
    );
    assert!(app.world().resource::<ViewerControls>().paused);
}

#[test]
fn viewer_restores_saved_speed_semantics_and_metadata_without_reinterpreting_weights() {
    use racing_rl::config::SpeedNormalization;
    let library = TrackLibrary::load_default().unwrap();
    for (mode, expected) in [
        (SpeedNormalization::LinearClamp, 1.),
        (SpeedNormalization::SignedAsymptotic, 2. / 3.),
    ] {
        let mut c = small_config();
        c.observation.speed_normalization = mode;
        let saved = saved_policy(&c, 0);
        let text = ron::to_string(&saved).unwrap();
        let text = if mode == SpeedNormalization::LinearClamp {
            text.replace("speed_normalization:LinearClamp,", "")
        } else {
            assert!(text.contains("speed_normalization:SignedAsymptotic"));
            text
        };
        let restored: Checkpoint = ron::from_str(&text).unwrap();
        restored.validate().unwrap();
        let mut incompatible: Checkpoint = ron::from_str(&text).unwrap();
        incompatible.config.observation.speed_normalization =
            if mode == SpeedNormalization::LinearClamp {
                SpeedNormalization::SignedAsymptotic
            } else {
                SpeedNormalization::LinearClamp
            };
        assert!(
            incompatible.validate().is_err(),
            "same dimensions cannot hide changed metadata"
        );
        let mut viewer = Viewer::new(
            restored,
            PathBuf::from("synthetic.ron"),
            1,
            Some("monza"),
            &library,
            None,
        )
        .unwrap();
        viewer.environments[0].state.speed = 500.;
        viewer.observe().unwrap();
        viewer.capture_inference().unwrap();
        assert_close(viewer.observations[0][21], expected);
        assert_eq!(viewer.trace_for(0).unwrap().input, viewer.observations[0]);
        assert_eq!(viewer.saved.features, c.observation.features());
        assert_eq!(viewer.config.observation.speed_normalization, mode);
    }
}
