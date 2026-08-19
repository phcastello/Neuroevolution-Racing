use bevy::{ecs::system::SystemParam, prelude::*};
use bevy_egui::{EguiContexts, egui};

use crate::simulation::{
    Car, CarProgress, KinematicCar, PlaybackState, SelectedCar, SensorReadings, SimulationConfig,
    SimulationMode,
};

use super::{
    FitnessHistory, fitness_plot::draw_fitness_plot, network_view::draw_network_placeholder,
};

#[derive(SystemParam)]
pub struct DashboardData<'w, 's> {
    playback: ResMut<'w, PlaybackState>,
    mode: ResMut<'w, SimulationMode>,
    config: Res<'w, SimulationConfig>,
    history: Res<'w, FitnessHistory>,
    cars: Query<'w, 's, (), With<Car>>,
    selected: Query<
        'w,
        's,
        (
            &'static KinematicCar,
            &'static SensorReadings,
            &'static CarProgress,
        ),
        With<SelectedCar>,
    >,
    virtual_time: ResMut<'w, Time<Virtual>>,
}

pub fn dashboard_system(mut contexts: EguiContexts, mut data: DashboardData) -> Result {
    let ctx = contexts.ctx_mut()?;
    let mut viewport_ui = egui::Ui::new(
        ctx.clone(),
        "dashboard_viewport".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );
    egui::Panel::top("title_bar")
        .exact_size(38.0)
        .show(&mut viewport_ui, |ui| {
            ui.horizontal_centered(|ui| {
                ui.heading("Neuroevolution Racing");
                ui.separator();
                ui.label("simulation infrastructure • AI not implemented");
            });
        });

    egui::Panel::right("dashboard")
        .exact_size(355.0)
        .resizable(false)
        .show(&mut viewport_ui, |ui| {
            ui.heading("Simulation");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut *data.mode, SimulationMode::Training, "Training");
                ui.selectable_value(&mut *data.mode, SimulationMode::Champion, "Champion");
                ui.add_enabled_ui(false, |ui| {
                    ui.selectable_value(&mut *data.mode, SimulationMode::Race, "Race (later)");
                });
            });
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
                        .selectable_label(data.playback.speed == speed, format!("{speed:.0}×"))
                        .clicked()
                    {
                        data.playback.speed = speed;
                        data.virtual_time.set_relative_speed(speed);
                    }
                }
            });
            ui.small("Space toggles pause. Simulation logic runs at a fixed 60 Hz.");

            ui.separator();
            ui.heading("Selected car");
            if let Some((car, sensors, progress)) = data.selected.iter().next() {
                ui.label(format!(
                    "Speed: {:>5.1} / {:.0}",
                    car.speed, data.config.max_speed
                ));
                ui.label(format!(
                    "Checkpoint: {}  •  Laps: {}",
                    progress.expected_checkpoint, progress.laps
                ));
                ui.label(format!(
                    "Completed: {}  •  Next-leg: {:.0}%",
                    progress.completed_checkpoints,
                    progress.toward_next * 100.0
                ));
                ui.horizontal(|ui| {
                    for value in sensors.normalized {
                        ui.monospace(format!("{value:.2}"));
                    }
                });
            }

            ui.separator();
            ui.heading("Fitness");
            if data.history.is_mock {
                ui.colored_label(egui::Color32::YELLOW, "Preview data — no GA is running");
            }
            draw_fitness_plot(ui, &data.history);

            ui.separator();
            ui.heading("Neural Network");
            draw_network_placeholder(ui);
            ui.small("Static layout preview only — no neurons or weights exist yet.");
        });
    Ok(())
}
