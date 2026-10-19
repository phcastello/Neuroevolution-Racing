//! PPO presentation depends on playback telemetry, never GA training resources.
use bevy::{diagnostic::DiagnosticsStore, prelude::*};
use bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use racing_rl::rollout::EndReason;

use crate::{
    display::{CameraViewState, DisplaySettings},
    rendering::CarVisualSettings,
    rl::viewer::{Viewer, ViewerControls},
    simulation::{Track, TrackDebug, TrackLibrary},
};

use super::{
    dashboard::{draw_camera_controls, draw_dashboard, draw_display_controls},
    ppo_checkpoint_browser::{PpoCheckpointBrowser, draw_policy_history},
    ppo_network_view::draw_ppo_network,
};

pub(crate) struct PpoDashboardPlugin;

impl Plugin for PpoDashboardPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PpoCheckpointBrowser>()
            .init_resource::<PpoDashboardState>()
            .add_systems(EguiPrimaryContextPass, ppo_dashboard_system);
    }
}

#[derive(Resource)]
struct PpoDashboardState {
    network_open: bool,
}

impl Default for PpoDashboardState {
    fn default() -> Self {
        Self {
            network_open: false,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn ppo_dashboard_system(
    mut contexts: EguiContexts,
    viewer: NonSend<Viewer>,
    mut controls: ResMut<ViewerControls>,
    mut browser: ResMut<PpoCheckpointBrowser>,
    mut state: ResMut<PpoDashboardState>,
    track: Res<Track>,
    library: Res<TrackLibrary>,
    mut camera: ResMut<CameraViewState>,
    mut display: ResMut<DisplaySettings>,
    mut visuals: ResMut<CarVisualSettings>,
    mut debug: ResMut<TrackDebug>,
    diagnostics: Res<DiagnosticsStore>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    browser.sync_source(&viewer.checkpoint_path, &viewer.saved.config.results_root);
    draw_dashboard(
        ctx,
        "PPO · frozen Actor-Critic policy · CPU playback",
        |ui| {
            ui.heading("PPO playback");
            ui.label(
                egui::RichText::new(
                    viewer
                        .checkpoint_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy(),
                )
                .strong(),
            )
            .on_hover_text(viewer.checkpoint_path.display().to_string());
            let trunk = viewer
                .saved
                .config
                .hidden
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(" → ");
            summary_grid(ui, "ppo_policy_summary", |ui| {
                row(ui, "Completed rollouts", viewer.saved.completed_rollouts);
                row(ui, "Policy version", viewer.saved.policy_version);
                row(
                    ui,
                    "Parameters",
                    viewer.saved.parameters.iter().map(Vec::len).sum::<usize>(),
                );
                row(
                    ui,
                    "Shared trunk",
                    format!("{} → {trunk}", viewer.saved.config.observation.dimension()),
                );
                row(ui, "Heads", "Actor 2 · Critic 1");
                row(
                    ui,
                    "Policy",
                    if viewer.saved.trained {
                        "Trained PPO"
                    } else {
                        "Untrained parameters"
                    },
                );
            });
            if let Some(error) = &viewer.error {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            }

            ui.separator();
            ui.heading("Playback");
            ui.horizontal(|ui| {
                let label = if controls.paused { "Resume" } else { "Pause" };
                if ui.button(label).clicked() {
                    controls.paused = !controls.paused;
                }
                if ui.button("Restart episodes").clicked() {
                    controls.restart_requested = true;
                }
            });
            ui.add(
                egui::Slider::new(&mut controls.speed, 0.1..=8.0)
                    .logarithmic(true)
                    .text("Playback speed"),
            );
            ui.checkbox(
                &mut controls.auto_reset,
                "Automatically restart finished episodes",
            );
            ui.small("Space = pause/resume. Episodes retain their terminal pose before automatically restarting.");

            ui.separator();
            ui.heading("Track");
            let mut track_id = viewer.track_id.clone();
            egui::ComboBox::from_id_salt("ppo_track_selector")
                .selected_text(&track.definition.name)
                .width(ui.available_width())
                .show_ui(ui, |ui| {
                    for definition in library.all_tracks() {
                        ui.selectable_value(
                            &mut track_id,
                            definition.id.clone(),
                            format!("{} [{}]", definition.name, definition.role.label()),
                        );
                    }
                });
            if track_id != viewer.track_id {
                controls.track_requested = Some(track_id);
            }
            ui.small(format!(
                "{} · {} · {:.0} units",
                track.definition.category.label(),
                track.definition.difficulty.label(),
                track.total_length
            ));

            ui.separator();
            ui.heading("Inspected car");
            egui::ComboBox::from_id_salt("ppo_car_selector")
                .selected_text(format!("Car #{}", controls.selected))
                .width(ui.available_width())
                .show_ui(ui, |ui| {
                    for (index, car) in viewer.cars.iter().enumerate() {
                        ui.selectable_value(
                            &mut controls.selected,
                            index,
                            format!("Car #{}", car.id),
                        );
                    }
                });
            if let Some(car) = viewer.cars.get(controls.selected) {
                ui.add(
                    egui::ProgressBar::new(car.progress.clamp(0.0, 1.0))
                        .text(format!("Lap progress {:.1}%", car.progress * 100.0)),
                );
                summary_grid(ui, "ppo_car_telemetry", |ui| {
                    row(ui, "Episode", car.episode);
                    row(ui, "Elapsed", format!("{:.2} s", car.elapsed));
                    row(
                        ui,
                        "State",
                        match car.reason {
                            None => "Running",
                            Some(EndReason::Collision) => "Collision",
                            Some(EndReason::Completed) => "Lap completed",
                            Some(EndReason::Timeout) => "Timeout",
                        },
                    );
                    row(ui, "Speed", format!("{:.1} units/s", car.speed));
                    row(ui, "Steering", format!("{:+.4}", car.action[0]));
                    row(ui, "Acceleration", format!("{:+.4}", car.action[1]));
                    row(ui, "Episode reward", format!("{:+.5}", car.reward));
                    row(ui, "Critic V(s)", format!("{:+.5}", car.value));
                    row(
                        ui,
                        "Actor μ",
                        format!("[{:+.4}, {:+.4}]", car.mean[0], car.mean[1]),
                    );
                    row(
                        ui,
                        "Global log σ",
                        format!("[{:+.3}, {:+.3}]", car.log_std[0], car.log_std[1]),
                    );
                    row(
                        ui,
                        "Decisions / ticks",
                        format!("{} / {}", car.decisions, car.physical_ticks),
                    );
                });
                ui.small("Actor μ is the Gaussian latent mean; executed actions are tanh(μ), exactly as in deterministic PPO evaluation.");
                ui.small("Network activations and Critic reflect the last real policy decision, before its physical step.");
            }
            ui.checkbox(&mut state.network_open, "Inspect live Actor-Critic network");

            ui.separator();
            egui::CollapsingHeader::new("Visual overlays")
                .default_open(true)
                .show(ui, |ui| {
                    ui.checkbox(&mut visuals.show_hitbox, "Show car hitbox");
                    ui.checkbox(&mut visuals.show_sensors, "Show sensors");
                    ui.checkbox(&mut debug.enabled, "Track debug");
                });
            ui.separator();
            draw_camera_controls(ui, &mut camera, "Follow Selected");
            ui.separator();
            draw_policy_history(
                ui,
                &mut browser,
                &viewer.checkpoint_path,
                &mut controls.checkpoint_requested,
            );
            ui.separator();
            egui::CollapsingHeader::new("Display").show(ui, |ui| {
                draw_display_controls(ui, &mut display, &diagnostics);
            });
        },
    );

    egui::Window::new("PPO Actor-Critic · live activations")
        .id(egui::Id::new("ppo_network_window"))
        .open(&mut state.network_open)
        .default_pos(egui::pos2(18.0, 54.0))
        .default_size(egui::vec2(980.0, 760.0))
        .resizable(true)
        .vscroll(true)
        .show(ctx, |ui| {
            ui.label(format!(
                "Rollout {} · car #{} · last policy decision",
                viewer.saved.completed_rollouts, controls.selected
            ));
            if let Some(trace) = viewer.trace_for(controls.selected) {
                draw_ppo_network(
                    ui,
                    trace,
                    &viewer.saved.config.hidden,
                    &viewer.saved.parameters,
                    &mut controls.network,
                );
            } else {
                ui.small("Waiting for the selected car's first policy decision.");
            }
        });
    Ok(())
}

fn summary_grid(ui: &mut egui::Ui, id: &str, contents: impl FnOnce(&mut egui::Ui)) {
    let width = ((ui.available_width() - 12.0) * 0.5).max(60.0);
    egui::Grid::new(id)
        .num_columns(2)
        .min_col_width(0.0)
        .max_col_width(width)
        .show(ui, contents);
}

fn row(ui: &mut egui::Ui, label: &str, value: impl std::fmt::Display) {
    ui.label(label);
    ui.monospace(value.to_string());
    ui.end_row();
}
