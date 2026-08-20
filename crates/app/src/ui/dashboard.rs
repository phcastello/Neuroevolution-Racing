use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    ecs::system::SystemParam,
    prelude::*,
};
use bevy_egui::{EguiContexts, egui};

use crate::display::{
    CameraBehavior, CameraViewState, DASHBOARD_WIDTH, DisplayMode, DisplaySettings, TOP_BAR_HEIGHT,
};
use crate::simulation::{
    Car, CarControls, CarObservation, CarProgress, KinematicCar, ManualControlMode, PlaybackState,
    SelectedCar, SimulationConfig, SimulationMode, TestDriveEnvironment, TestDriveSettings, Track,
    TrackDebug, TrackLibrary, TrackSelection,
};

use super::{
    FitnessHistory, fitness_plot::draw_fitness_plot, network_view::draw_network_placeholder,
};

type SelectedCarTelemetry<'w, 's> = Query<
    'w,
    's,
    (
        &'static KinematicCar,
        &'static CarControls,
        &'static CarObservation,
        &'static Transform,
        Option<&'static CarProgress>,
    ),
    With<SelectedCar>,
>;

#[derive(SystemParam)]
pub struct DashboardData<'w, 's> {
    playback: ResMut<'w, PlaybackState>,
    mode: ResMut<'w, SimulationMode>,
    config: Res<'w, SimulationConfig>,
    track: Res<'w, Track>,
    track_library: Res<'w, TrackLibrary>,
    track_selection: ResMut<'w, TrackSelection>,
    track_debug: ResMut<'w, TrackDebug>,
    history: Res<'w, FitnessHistory>,
    display: ResMut<'w, DisplaySettings>,
    camera: ResMut<'w, CameraViewState>,
    test_drive: ResMut<'w, TestDriveSettings>,
    diagnostics: Res<'w, DiagnosticsStore>,
    cars: Query<'w, 's, (), With<Car>>,
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
                    ui.label("simulation infrastructure • AI not implemented");
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
                    ui.add_space(4.0);
                    egui::Grid::new("summary_grid")
                        .num_columns(2)
                        .show(ui, |ui| {
                            ui.label("Generation");
                            ui.monospace("--");
                            ui.end_row();
                            ui.label("Population");
                            ui.monospace(data.cars.iter().count().to_string());
                            ui.end_row();
                            ui.label("Alive");
                            ui.monospace(format!("{} (temporary)", data.cars.iter().count()));
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
                        if ui.button("Reset Car  [R]").clicked() {
                            data.test_drive.reset_requested = true;
                        }
                        ui.small("W/S = acceleration • A/D = left/right steering");
                    }

                    ui.add_space(8.0);
                    if ui
                        .button(if data.playback.paused {
                            "▶ Resume"
                        } else {
                            "⏸ Pause"
                        })
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
                                .selectable_label(
                                    data.playback.speed == speed,
                                    format!("{speed:.0}×"),
                                )
                                .clicked()
                            {
                                data.playback.speed = speed;
                                data.virtual_time.set_relative_speed(speed);
                            }
                        }
                    });
                    ui.small("Space toggles pause. Simulation logic runs at a fixed 60 Hz.");
                    ui.checkbox(&mut data.track_debug.enabled, "Track debug");

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
                    if *data.mode == SimulationMode::TestDrive {
                        ui.horizontal(|ui| {
                            ui.label("Behavior:");
                            ui.radio_value(&mut data.camera.behavior, CameraBehavior::Free, "Free");
                            ui.radio_value(
                                &mut data.camera.behavior,
                                CameraBehavior::FollowCar,
                                "Follow Car",
                            );
                        });
                    }
                    ui.small("Wheel = zoom • Q/E = rotate • Middle/right drag = pan");

                    ui.separator();
                    ui.heading(if *data.mode == SimulationMode::TestDrive {
                        "Physics telemetry"
                    } else {
                        "Selected car"
                    });
                    if let Some((car, controls, observation, transform, progress)) =
                        data.selected.iter().next()
                    {
                        ui.label(format!("Speed: {:+.1}", car.speed));
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

                        ui.add_space(4.0);
                        ui.label("MLP observation (live CarObservation)");
                        for (label, value) in [
                            ("Left", observation.sensors[0]),
                            ("Front-left", observation.sensors[1]),
                            ("Front", observation.sensors[2]),
                            ("Front-right", observation.sensors[3]),
                            ("Right", observation.sensors[4]),
                            ("Speed", observation.normalized_speed),
                        ] {
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
                        if data.history.is_mock {
                            ui.colored_label(
                                egui::Color32::YELLOW,
                                "Preview data — no GA is running",
                            );
                        }
                        draw_fitness_plot(ui, &data.history);

                        ui.separator();
                        ui.heading("Neural Network");
                        draw_network_placeholder(ui);
                        ui.small("Static layout preview only — no neurons or weights exist yet.");
                    }
                });
        });
    Ok(())
}
