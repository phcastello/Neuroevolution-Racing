use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    ecs::system::SystemParam,
    prelude::*,
};
use bevy_egui::{EguiContexts, egui};

use crate::display::{
    CameraBehavior, CameraViewState, DASHBOARD_WIDTH, DisplayMode, DisplaySettings, TOP_BAR_HEIGHT,
};
use crate::rendering::{CAR_SPRITE_LABELS, CarVisualSettings};
use crate::simulation::{
    Car, CarControls, CarObservation, CarProgress, CheckpointStore, EvaluationState, KinematicCar,
    LaserState, LoadedNetwork, ManualControlMode, MlpController, PlaybackState, SelectedCar,
    SimulationConfig, SimulationMode, TestDriveEnvironment, TestDriveSettings, Track, TrackDebug,
    TrackLibrary, TrackSelection, TrainingFastForward, TrainingPhase, TrainingState,
    desired_yaw_rate, limited_yaw_rate, max_grip_yaw_rate,
};

use super::{
    checkpoint_browser::draw_checkpoint_browser, fitness_plot::draw_fitness_plot,
    network_view::draw_network,
};

const KMH_PER_UNIT_PER_SECOND: f32 = 0.578_52;

type SelectedCarTelemetry<'w, 's> = Query<
    'w,
    's,
    (
        &'static KinematicCar,
        &'static CarControls,
        &'static CarObservation,
        &'static Transform,
        Option<&'static CarProgress>,
        Option<&'static MlpController>,
        &'static Car,
        Option<&'static EvaluationState>,
    ),
    With<SelectedCar>,
>;

#[derive(SystemParam)]
pub struct DashboardData<'w, 's> {
    playback: ResMut<'w, PlaybackState>,
    fast_forward: ResMut<'w, TrainingFastForward>,
    mode: ResMut<'w, SimulationMode>,
    config: Res<'w, SimulationConfig>,
    track: Res<'w, Track>,
    track_library: Res<'w, TrackLibrary>,
    track_selection: ResMut<'w, TrackSelection>,
    track_debug: ResMut<'w, TrackDebug>,
    car_visuals: ResMut<'w, CarVisualSettings>,
    training: Res<'w, TrainingState>,
    laser: Res<'w, LaserState>,
    checkpoints: ResMut<'w, CheckpointStore>,
    loaded_network: ResMut<'w, LoadedNetwork>,
    display: ResMut<'w, DisplaySettings>,
    camera: ResMut<'w, CameraViewState>,
    test_drive: ResMut<'w, TestDriveSettings>,
    diagnostics: Res<'w, DiagnosticsStore>,
    evaluations: Query<'w, 's, &'static EvaluationState>,
    selected: SelectedCarTelemetry<'w, 's>,
    virtual_time: ResMut<'w, Time<Virtual>>,
}

pub fn dashboard_system(mut contexts: EguiContexts, mut data: DashboardData) -> Result {
    let ctx = contexts.ctx_mut()?;
    let viewport_rect = ctx.viewport_rect();
    if !viewport_rect.is_finite() || viewport_rect.width() < 1.0 || viewport_rect.height() < 1.0 {
        return Ok(());
    }
    let mut viewport_ui = egui::Ui::new(
        ctx.clone(),
        "dashboard_viewport".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(viewport_rect),
    );
    egui::Panel::top("title_bar")
        .exact_size(TOP_BAR_HEIGHT.min(viewport_rect.height()))
        .show(&mut viewport_ui, |ui| {
            ui.horizontal_centered(|ui| {
                ui.heading("Neuroevolution Racing");
                if ui.available_width() > 520.0 {
                    ui.separator();
                    ui.label("live neuroevolution • MLP + genetic algorithm");
                }
            });
        });

    egui::Panel::right("dashboard")
        .exact_size(DASHBOARD_WIDTH.min(viewport_rect.width()))
        .resizable(false)
        .show(&mut viewport_ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    if data.fast_forward.is_active() {
                        let target = data.fast_forward.target_generation.unwrap();
                        let current = data.training.generation();
                        ui.heading("FAST-FORWARD");
                        ui.strong("Turbo de treinamento / rendering mínimo");
                        ui.separator();
                        egui::Grid::new("fast_forward_status")
                            .num_columns(2)
                            .show(ui, |ui| {
                                ui.label("Current generation");
                                ui.monospace(current.to_string());
                                ui.end_row();
                                ui.label("Target generation");
                                ui.monospace(target.to_string());
                                ui.end_row();
                                ui.label("Elapsed wall time");
                                ui.monospace(format!(
                                    "{:.1}s",
                                    data.fast_forward.elapsed().as_secs_f64()
                                ));
                                ui.end_row();
                                ui.label("Throughput");
                                ui.monospace(format!(
                                    "{:.2} generations/s",
                                    data.fast_forward.generations_per_second(current)
                                ));
                                ui.end_row();
                                ui.label("Fixed tick rate");
                                ui.monospace(format!(
                                    "{:.0} ticks/s",
                                    data.fast_forward.fixed_ticks_per_second()
                                ));
                                ui.end_row();
                            });
                        ui.small("Every logical tick remains exactly 1/60 s; no generation, training track, validation, or autosave is skipped.");
                        ui.add_space(8.0);
                        if ui.button("Cancel fast-forward").clicked()
                            && let Some(restore) = data.fast_forward.cancel()
                        {
                            data.playback.speed = restore.speed;
                            data.playback.paused = false;
                            data.virtual_time.set_relative_speed(restore.speed);
                            data.virtual_time.set_max_delta(restore.max_delta);
                            data.virtual_time.unpause();
                        }
                        return;
                    }

                    ui.heading("Simulation");
                    let mut selected_mode = *data.mode;
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut selected_mode,
                            SimulationMode::Training,
                            "Training",
                        );
                        ui.selectable_value(
                            &mut selected_mode,
                            SimulationMode::Champion,
                            "Champion",
                        );
                        ui.add_enabled_ui(false, |ui| {
                            ui.selectable_value(
                                &mut selected_mode,
                                SimulationMode::Race,
                                "Race (later)",
                            );
                        });
                    });
                    ui.selectable_value(
                        &mut selected_mode,
                        SimulationMode::TestDrive,
                        "Test Drive",
                    );
                    if selected_mode != *data.mode {
                        *data.mode = selected_mode;
                    }
                    if *data.mode == SimulationMode::Champion {
                        if let Some(loaded) = &data.loaded_network.checkpoint {
                            ui.small(format!(
                                "Loaded checkpoint: {}",
                                loaded.source_filename
                            ));
                        } else {
                            ui.small("Champion source: current training session");
                        }
                    }
                    ui.add_space(4.0);
                    egui::Grid::new("summary_grid")
                        .num_columns(2)
                        .show(ui, |ui| {
                            ui.label("Generation");
                            ui.monospace(data.training.generation().to_string());
                            ui.end_row();
                            ui.label("Population");
                            ui.monospace(data.training.population().len().to_string());
                            ui.end_row();
                            let active = data
                                .evaluations
                                .iter()
                                .filter(|evaluation| !evaluation.is_finished())
                                .count();
                            let finished = data.evaluations.iter().len().saturating_sub(active);
                            ui.label("Active / finished");
                            ui.monospace(format!("{active} / {finished}"));
                            ui.end_row();
                            ui.label("Phase");
                            ui.monospace(match data.training.phase() {
                                TrainingPhase::TrainingTrack { index, total, .. } => {
                                    format!("training {}/{}", index + 1, total)
                                }
                                TrainingPhase::Validation { .. } => "validation".into(),
                                TrainingPhase::Evolving => "evolving".into(),
                            });
                            ui.end_row();
                            ui.label("Episode");
                            let elapsed = data
                                .evaluations
                                .iter()
                                .map(|evaluation| evaluation.elapsed)
                                .fold(0.0_f32, f32::max);
                            ui.monospace(format!(
                                "{elapsed:.1}s / {:.1}s safety cap",
                                data.training
                                    .evaluation_config()
                                    .maximum_episode_duration
                            ));
                            ui.end_row();
                        });

                    ui.separator();
                    ui.heading("Track");
                    let mut selected_id = data.track_selection.active_id.clone();
                    egui::ComboBox::from_id_salt("track_selector")
                        .selected_text(&data.track.definition.name)
                        .width(ui.available_width())
                        .show_ui(ui, |ui| {
                            ui.label(egui::RichText::new("TRAINING").strong());
                            for definition in data.track_library.training_tracks() {
                                ui.selectable_value(
                                    &mut selected_id,
                                    definition.id.clone(),
                                    &definition.name,
                                );
                            }
                            ui.separator();
                            ui.label(egui::RichText::new("VALIDATION — HELD OUT").strong());
                            for definition in data.track_library.validation_tracks() {
                                ui.selectable_value(
                                    &mut selected_id,
                                    definition.id.clone(),
                                    format!("{}  [validation]", definition.name),
                                );
                            }
                        });
                    if selected_id != data.track_selection.active_id {
                        data.track_selection.request(selected_id);
                    }
                    egui::Grid::new("track_metadata")
                        .num_columns(2)
                        .show(ui, |ui| {
                            ui.label("Role");
                            ui.monospace(data.track.definition.role.label());
                            ui.end_row();
                            ui.label("Category");
                            ui.monospace(data.track.definition.category.label());
                            ui.end_row();
                            ui.label("Difficulty");
                            ui.monospace(data.track.definition.difficulty.label());
                            ui.end_row();
                            ui.label("Length");
                            ui.monospace(format!(
                                "{:.0} simulation units",
                                data.track.total_length
                            ));
                            ui.end_row();
                            ui.label("Width");
                            ui.monospace(format!("{:.0} units", data.track.width));
                            ui.end_row();
                        });
                    if let Some(country) = &data.track.definition.country {
                        ui.small(country);
                    }
                    ui.small(&data.track_selection.status);

                    if *data.mode == SimulationMode::TestDrive {
                        ui.separator();
                        ui.heading("Test Drive");
                        ui.horizontal(|ui| {
                            ui.label("Environment:");
                            let mut environment = data.test_drive.environment;
                            ui.radio_value(&mut environment, TestDriveEnvironment::Track, "Track");
                            ui.radio_value(
                                &mut environment,
                                TestDriveEnvironment::OpenField,
                                "Open Field",
                            );
                            if environment != data.test_drive.environment {
                                data.test_drive.environment = environment;
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Input:");
                            let mut control_mode = data.test_drive.control_mode;
                            ui.radio_value(
                                &mut control_mode,
                                ManualControlMode::Keyboard,
                                "Keyboard",
                            );
                            ui.radio_value(
                                &mut control_mode,
                                ManualControlMode::Sliders,
                                "Sliders",
                            );
                            if control_mode != data.test_drive.control_mode {
                                data.test_drive.control_mode = control_mode;
                            }
                        });
                        if data.test_drive.control_mode == ManualControlMode::Sliders {
                            let mut steering = data.test_drive.slider_controls.steering;
                            let mut acceleration = data.test_drive.slider_controls.acceleration;
                            ui.add(
                                egui::Slider::new(&mut steering, -1.0..=1.0)
                                    .text("Steering (+ left)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut acceleration, -1.0..=1.0)
                                    .text("Acceleration"),
                            );
                            data.test_drive.slider_controls =
                                CarControls::new(acceleration, steering);
                        }
                        ui.horizontal(|ui| {
                            ui.label("Vehicle sprite:");
                            egui::ComboBox::from_id_salt("vehicle_sprite_selector")
                                .selected_text(
                                    CAR_SPRITE_LABELS[data
                                        .car_visuals
                                        .test_drive_sprite
                                        .min(CAR_SPRITE_LABELS.len() - 1)],
                                )
                                .show_ui(ui, |ui| {
                                    for (index, label) in CAR_SPRITE_LABELS.iter().enumerate() {
                                        ui.selectable_value(
                                            &mut data.car_visuals.test_drive_sprite,
                                            index,
                                            *label,
                                        );
                                    }
                                });
                        });
                        ui.small("Cosmetic only; vehicle physics and controls stay identical.");
                        if ui.button("Reset Car  [R]").clicked() {
                            data.test_drive.reset_requested = true;
                        }
                        ui.small("W/S = acceleration • A/D = left/right steering");
                    }

                    ui.add_space(8.0);
                    if ui
                        .add_enabled(
                            !data.fast_forward.is_active(),
                            egui::Button::new(if data.playback.paused {
                                "▶ Resume"
                            } else {
                                "⏸ Pause"
                            }),
                        )
                        .clicked()
                    {
                        data.playback.paused = !data.playback.paused;
                        if data.playback.paused {
                            data.virtual_time.pause();
                        } else {
                            data.virtual_time.unpause();
                        }
                    }
                    ui.horizontal(|ui| {
                        ui.label("Speed:");
                        for speed in [1.0, 2.0, 10.0, 25.0] {
                            if ui
                                .add_enabled(
                                    !data.fast_forward.is_active(),
                                    egui::Button::selectable(
                                        data.playback.speed == speed,
                                        format!("{speed:.0}×"),
                                    ),
                                )
                                .clicked()
                            {
                                data.playback.speed = speed;
                                data.virtual_time.set_relative_speed(speed);
                            }
                        }
                    });
                    if *data.mode == SimulationMode::Training {
                        ui.add_space(4.0);
                        ui.horizontal(|ui| {
                            ui.label("Target generation:");
                            ui.add_enabled(
                                !data.fast_forward.is_active(),
                                egui::TextEdit::singleline(&mut data.fast_forward.target_input)
                                    .desired_width(82.0)
                                    .hint_text("200"),
                            );
                        });
                        if data.fast_forward.is_active() {
                            let target = data.fast_forward.target_generation.unwrap();
                            ui.strong(format!("FAST-FORWARD → Gen {target}"));
                            ui.monospace(format!("Current Gen: {}", data.training.generation()));
                            if ui.button("Cancel fast-forward").clicked()
                                && let Some(restore) = data.fast_forward.cancel()
                            {
                                data.playback.speed = restore.speed;
                                data.playback.paused = false;
                                data.virtual_time.set_relative_speed(restore.speed);
                                data.virtual_time.set_max_delta(restore.max_delta);
                                data.virtual_time.unpause();
                            }
                        } else if ui.button("Fast-forward").clicked() {
                            match data.fast_forward.target_input.trim().parse::<usize>() {
                                Ok(target)
                                    if data.fast_forward.start(
                                        target,
                                        data.training.generation(),
                                        data.playback.speed,
                                        data.virtual_time.max_delta(),
                                    ) =>
                                {
                                    data.playback.paused = false;
                                    // The turbo driver owns exact FixedMain ticks. Pausing
                                    // virtual-time prevents Bevy's regular frame-driven loop
                                    // from adding a second, FPS-coupled backlog.
                                    data.virtual_time.pause();
                                }
                                Ok(_) => {}
                                Err(_) => {
                                    data.fast_forward.status =
                                        "Enter a non-negative generation number".into();
                                }
                            }
                        }
                        if !data.fast_forward.status.is_empty() {
                            ui.small(&data.fast_forward.status);
                        }
                    }
                    ui.small("Space toggles pause. Simulation logic runs at a fixed 60 Hz.");
                    ui.checkbox(&mut data.track_debug.enabled, "Track debug");
                    ui.checkbox(&mut data.car_visuals.show_hitbox, "Show car hitbox");
                    ui.checkbox(&mut data.car_visuals.show_sensors, "Show sensors");

                    ui.separator();
                    ui.heading("Display");
                    for (mode, label) in [
                        (DisplayMode::Windowed, "Windowed"),
                        (DisplayMode::Fullscreen1080p, "Fullscreen 1080p"),
                        (DisplayMode::Fullscreen1440p, "Fullscreen 1440p"),
                        (DisplayMode::FullscreenNative, "Fullscreen Native"),
                    ] {
                        if ui
                            .selectable_label(data.display.current == mode, label)
                            .clicked()
                        {
                            data.display.request(mode);
                        }
                    }
                    ui.small(&data.display.status);
                    let fps_diagnostic = data.diagnostics.get(&FrameTimeDiagnosticsPlugin::FPS);
                    ui.monospace(
                        match fps_diagnostic.and_then(|diagnostic| diagnostic.value()) {
                            Some(fps) => format!("FPS atual: {fps:.0}"),
                            None => "FPS atual: coletando...".into(),
                        },
                    );
                    ui.monospace(
                        match fps_diagnostic.and_then(|diagnostic| diagnostic.smoothed()) {
                            Some(fps) => format!("FPS médio: {fps:.0}"),
                            None => "FPS médio: coletando...".into(),
                        },
                    );

                    ui.separator();
                    ui.heading("Camera");
                    egui::Grid::new("camera_status")
                        .num_columns(2)
                        .show(ui, |ui| {
                            ui.label("Zoom");
                            ui.monospace(format!("{:.2}×", data.camera.zoom));
                            ui.end_row();
                            ui.label("Rotation");
                            ui.monospace(format!("{:+.0}°", data.camera.rotation.to_degrees()));
                            ui.end_row();
                        });
                    ui.horizontal(|ui| {
                        if ui.button("Fit Track").clicked() {
                            data.camera.request_fit();
                        }
                        if ui.button("Reset View").clicked() {
                            data.camera.request_reset();
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Behavior:");
                        ui.radio_value(&mut data.camera.behavior, CameraBehavior::Free, "Free");
                        ui.radio_value(
                            &mut data.camera.behavior,
                            CameraBehavior::FollowCar,
                            if *data.mode == SimulationMode::TestDrive {
                                "Follow Car"
                            } else {
                                "Follow Leader"
                            },
                        );
                    });
                    ui.small("Wheel = zoom • Q/E = rotate • Middle/right drag = pan");

                    ui.separator();
                    ui.heading(if *data.mode == SimulationMode::TestDrive {
                        "Physics telemetry"
                    } else {
                        "Selected car"
                    });
                    if let Some((
                        car,
                        controls,
                        observation,
                        transform,
                        progress,
                        _,
                        identity,
                        evaluation,
                    )) = data.selected.iter().next()
                    {
                        if *data.mode != SimulationMode::TestDrive {
                            ui.label(format!("Current leader: car #{}", identity.id));
                        }
                        ui.label(format!(
                            "Speed: {:+.1} u/s ({:+.1} km/h)",
                            car.speed,
                            car.speed * KMH_PER_UNIT_PER_SECOND
                        ));
                        if *data.mode == SimulationMode::TestDrive {
                            ui.label(format!("Heading: {:+.1}°", car.heading.to_degrees()));
                            ui.label(format!(
                                "Position: ({:+.1}, {:+.1})",
                                transform.translation.x, transform.translation.y
                            ));
                            ui.monospace(format!(
                                "MLP-equivalent output: [steering: {:+.2}, acceleration: {:+.2}]",
                                controls.steering, controls.acceleration
                            ));
                            let requested_yaw_rate =
                                desired_yaw_rate(car.speed, controls.steering, &data.config);
                            let grip_yaw_limit = max_grip_yaw_rate(car.speed, &data.config);
                            let actual_yaw_rate =
                                limited_yaw_rate(car.speed, controls.steering, &data.config);
                            ui.label("Live cornering limits");
                            for (label, value, unit) in [
                                ("Requested yaw rate", requested_yaw_rate, "rad/s"),
                                ("Grip-limited max", grip_yaw_limit, "rad/s"),
                                ("Actual yaw rate", actual_yaw_rate, "rad/s"),
                                (
                                    "Estimated lateral accel",
                                    (car.speed * actual_yaw_rate).abs(),
                                    "u/sÂ²",
                                ),
                            ] {
                                ui.monospace(format!("{label}: {value:.2} {unit}"));
                            }
                        }
                        if let Some(progress) = progress {
                            ui.label(format!(
                                "Track distance: {:.1} / {:.1}",
                                progress.track_distance, data.track.total_length
                            ));
                            ui.label(format!(
                                "Progress: {:.1}%  •  Best: {:.1}",
                                progress.normalized_progress * 100.0,
                                progress.best_track_distance
                            ));
                            if data.track_debug.enabled {
                                ui.small(format!(
                                    "Projected segment: {}  •  Track width: {:.0}",
                                    progress.nearest_segment, data.track.width
                                ));
                            }
                        }
                        if let Some(evaluation) = evaluation
                            && let Some(reason) = evaluation.finish_reason
                        {
                            ui.label(format!("Episode finished: {}", reason.label()));
                        }

                        ui.add_space(4.0);
                        ui.label("MLP observation (live CarObservation)");
                        for (label, value) in [
                            "Left 60°",
                            "Left 30°",
                            "Front",
                            "Right 30°",
                            "Right 60°",
                            "Speed",
                        ]
                        .into_iter()
                        .zip(observation.as_inputs())
                        {
                            ui.horizontal(|ui| {
                                ui.label(format!("{label:>11}:"));
                                ui.monospace(format!("{value:.2}"));
                            });
                        }
                    }

                    if *data.mode == SimulationMode::TestDrive {
                        ui.separator();
                        ui.heading("Shared vehicle physics");
                        egui::Grid::new("physics_config")
                            .num_columns(2)
                            .show(ui, |ui| {
                                for (label, value) in [
                                    ("Acceleration", data.config.acceleration_rate),
                                    ("Braking", data.config.braking_rate),
                                    ("Coasting loss", data.config.coasting_deceleration),
                                    ("Accel falloff", data.config.acceleration_falloff_speed),
                                    ("Speed normalization", data.config.speed_normalization_scale),
                                    ("Turn rate", data.config.turn_rate),
                                    (
                                        "Max lateral acceleration",
                                        data.config.max_lateral_acceleration,
                                    ),
                                ] {
                                    ui.label(label);
                                    ui.monospace(format!("{value:.2}"));
                                    ui.end_row();
                                }
                            });
                        ui.small("These are the same SimulationConfig values used by AI cars.");
                    } else {
                        ui.separator();
                        ui.heading("Fitness");
                        if data.training.history().is_empty() {
                            ui.small("The first fitness sample appears when generation 0 finishes.");
                        }
                        draw_fitness_plot(ui, data.training.history());

                        let laser_config = data.training.evaluation_config().laser;
                        ui.small(format!(
                            "laser: grace {:.1}s • speed {:.1}/{:.1} u/s • progress {:.1}u",
                            laser_config.grace_period,
                            data.laser.speed,
                            laser_config.maximum_speed,
                            data.laser.progress,
                        ));
                        ui.small(format!(
                            "fitness: progress={:.2}, useful speed={:.2}, completion bonus={:.2}, collision penalty={:.2}",
                            data.training.evaluation_config().progress_weight,
                            data.training.evaluation_config().speed_weight,
                            data.training.evaluation_config().completion_bonus,
                            data.training.evaluation_config().collision_penalty,
                        ));

                        ui.separator();
                        ui.heading("Current Leader Neural Network");
                        if let Some((
                            _,
                            _,
                            _,
                            _,
                            Some(progress),
                            Some(controller),
                            identity,
                            _,
                        )) = data.selected.iter().next()
                        {
                            let telemetry = controller.telemetry();
                            ui.label(format!(
                                "Generation {} • car #{} • live progress {:.1}%",
                                data.training.generation(),
                                identity.id,
                                progress.normalized_progress * 100.0
                            ));
                            draw_network(
                                ui,
                                telemetry.layer_sizes,
                                telemetry.parameters,
                                telemetry.activations,
                            );
                            ui.small(
                                "Activations update after every controller inference; the view follows the live first-place car.",
                            );
                        } else {
                            ui.small("Waiting for the current generation leader telemetry.");
                        }
                        if let Some(stats) = data.training.current_training_fitness() {
                            ui.label(format!(
                                "Training fitness: best {:.3} / average {:.3}",
                                stats.best_fitness, stats.average_fitness
                            ));
                        }
                        if let Some(validation) = data.training.latest_validation() {
                            ui.label(format!(
                                "Validation: {} / {:.3} / {}",
                                validation.track_id,
                                validation.score,
                                validation.finish_reason.label()
                            ));
                        }

                        ui.separator();
                        draw_checkpoint_browser(
                            ui,
                            &data.training,
                            &data.config,
                            &mut data.checkpoints,
                            &mut data.loaded_network,
                            &mut data.mode,
                        );
                    }
                });
        });
    Ok(())
}
