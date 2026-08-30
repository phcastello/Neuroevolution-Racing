use bevy_egui::egui;

use crate::simulation::{
    CheckpointStore, LoadedNetwork, SimulationConfig, SimulationMode, TrainingState,
};

pub fn draw_checkpoint_browser(
    ui: &mut egui::Ui,
    training: &TrainingState,
    simulation: &SimulationConfig,
    checkpoints: &mut CheckpointStore,
    loaded_network: &mut LoadedNetwork,
    mode: &mut SimulationMode,
) {
    ui.heading("Network checkpoints");
    ui.horizontal(|ui| {
        if ui.button("Refresh").clicked() {
            checkpoints.refresh();
        }
        let can_save = training.completed_champion().is_some();
        if ui
            .add_enabled(can_save, egui::Button::new("Save current champion"))
            .clicked()
        {
            if let Err(error) = checkpoints.save_current_champion(training, simulation) {
                checkpoints.status = format!("Save failed: {error}");
            }
        }
    });
    if training.completed_champion().is_none() {
        ui.small("Manual save becomes available after a champion finishes held-out validation.");
    }

    ui.horizontal(|ui| {
        ui.checkbox(&mut checkpoints.settings.enabled, "Auto-save enabled");
        ui.label("every");
        ui.add(
            egui::DragValue::new(&mut checkpoints.settings.interval_generations).range(1..=100_000),
        );
        ui.label("generations");
    });
    ui.small("Auto-save runs only after held-out validation has completed.");
    ui.small(format!("Directory: {}", checkpoints.directory().display()));
    ui.small(&checkpoints.status);

    if let Some(loaded) = &loaded_network.checkpoint {
        ui.label(format!(
            "Champion source: {} (generation {})",
            loaded.source_filename, loaded.saved.generation
        ));
    }

    let mut load_index = None;
    for (index, entry) in checkpoints.entries.iter().enumerate() {
        let title = if entry.summary.is_some() {
            entry.filename.clone()
        } else {
            format!("{} [invalid]", entry.filename)
        };
        egui::CollapsingHeader::new(title)
            .id_salt(("checkpoint", &entry.path))
            .show(ui, |ui| {
                if let Some(summary) = &entry.summary {
                    egui::Grid::new(("checkpoint_metadata", &entry.path))
                        .num_columns(2)
                        .show(ui, |ui| {
                            row(ui, "Format", format!("V{}", summary.format_version));
                            row(ui, "Generation", summary.generation.to_string());
                            row(ui, "Architecture", summary.architecture.clone());
                            row(
                                ui,
                                "Champion fitness",
                                format!("{:.5}", summary.training.champion_training_fitness),
                            );
                            row(
                                ui,
                                "Population average",
                                format!("{:.5}", summary.training.population_average_fitness),
                            );
                            row(
                                ui,
                                "Avg useful speed",
                                format!(
                                    "{:.2} u/s",
                                    summary.training.average_useful_progress_speed
                                ),
                            );
                            row(
                                ui,
                                "Completion",
                                format!("{:.1}%", summary.training.completion_rate * 100.0),
                            );
                            row(
                                ui,
                                "Training tracks",
                                summary.training.training_tracks.join(", "),
                            );
                            row(ui, "Validation track", summary.validation.track_id.clone());
                            row(
                                ui,
                                "Validation score",
                                format!("{:.5}", summary.validation.score),
                            );
                            row(
                                ui,
                                "Validation reason",
                                summary.validation.finish_reason.label().into(),
                            );
                        });
                    let counts = summary.training.finish_counts;
                    if summary.format_version == 1 {
                        ui.small(format!(
                            "Champion episodes (legacy): completed={} collision={} stalled={} timeout={}",
                            counts.completed, counts.collision, counts.stalled, counts.timeout
                        ));
                        ui.small(format!(
                            "Legacy evaluation: stall {:.0}u / {:.1}s • timeout {:.1}s",
                            summary.evaluation.significant_progress_epsilon,
                            summary.evaluation.stall_timeout,
                            summary.evaluation.maximum_episode_duration,
                        ));
                    } else {
                        ui.small(format!(
                            "Champion episodes: completed={} collision={} laser={} timeout={}",
                            counts.completed,
                            counts.collision,
                            counts.laser_eliminated,
                            counts.timeout
                        ));
                        if let Some(laser) = summary.evaluation.laser {
                            ui.small(format!(
                                "Laser: grace {:.1}s • accel {:.1}u/s² • max {:.1}u/s • sensor {:.0}u • timeout {:.1}s",
                                laser.grace_period,
                                laser.acceleration,
                                laser.maximum_speed,
                                summary.evaluation.sensor_max_distance.unwrap_or_default(),
                                summary.evaluation.maximum_episode_duration,
                            ));
                        }
                    }
                    if summary.format_version >= 3 {
                        ui.small(format!(
                            "Useful speed: asymptotic v/(v+k), k={:.0}u/s (half-saturation)",
                            summary
                                .evaluation
                                .progress_speed_half_saturation
                        ));
                    } else {
                        ui.small(format!(
                            "Useful speed (legacy): hard clamp at {:.0}u/s",
                            summary
                                .evaluation
                                .progress_speed_half_saturation
                        ));
                    }
                    if ui.button("Load / Test").clicked() {
                        load_index = Some(index);
                    }
                } else if let Some(error) = &entry.error {
                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                }
            });
    }

    if let Some(index) = load_index {
        match checkpoints.load(index) {
            Ok(loaded) => {
                loaded_network.checkpoint = Some(loaded);
                *mode = SimulationMode::Champion;
            }
            Err(error) => checkpoints.status = format!("Load failed: {error}"),
        }
    }
}

fn row(ui: &mut egui::Ui, label: &str, value: String) {
    ui.label(label);
    ui.monospace(value);
    ui.end_row();
}
