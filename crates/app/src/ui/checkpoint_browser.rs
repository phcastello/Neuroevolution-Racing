use bevy_egui::egui;

use crate::simulation::{
    CheckpointEntry, CheckpointStore, LoadedNetwork, RaceRoster, RaceRosterAdd, RaceSession,
    SimulationConfig, SimulationMode, TrainingState, latest_checkpoint_indices,
};

pub fn draw_checkpoint_browser(
    ui: &mut egui::Ui,
    training: &TrainingState,
    simulation: &SimulationConfig,
    checkpoints: &mut CheckpointStore,
    loaded_network: &mut LoadedNetwork,
    mode: &mut SimulationMode,
    roster: &mut RaceRoster,
    race: &mut RaceSession,
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
    let add_latest = ui.button("Add latest from each architecture").clicked();
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
    let mut race_index = None;
    let mut group_start = 0;
    while group_start < checkpoints.entries.len() {
        let folder = checkpoints.entries[group_start].folder.clone();
        let group_end = checkpoints.entries[group_start..]
            .iter()
            .position(|entry| entry.folder != folder)
            .map_or(checkpoints.entries.len(), |offset| group_start + offset);
        let entry_count = group_end - group_start;
        let folder_title = format!("📁 {} ({entry_count})", folder.display());

        egui::CollapsingHeader::new(folder_title)
            .id_salt(("checkpoint_folder", &folder))
            .default_open(folder == checkpoints.directory())
            .show(ui, |ui| {
                for (index, entry) in checkpoints.entries[group_start..group_end]
                    .iter()
                    .enumerate()
                    .map(|(offset, entry)| (group_start + offset, entry))
                {
                    let (load, add_to_race) = draw_checkpoint_entry(ui, entry);
                    if load {
                        load_index = Some(index);
                    }
                    if add_to_race {
                        race_index = Some(index);
                    }
                }
            });
        group_start = group_end;
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
    if let Some(index) = race_index {
        add_checkpoint_to_race(checkpoints, roster, race, index);
    }
    if add_latest {
        let indices = latest_checkpoint_indices(&checkpoints.entries);
        let mut added = 0usize;
        let mut duplicates = 0usize;
        let mut failures = Vec::new();
        for index in indices {
            match checkpoints
                .load_for_race(index)
                .map_err(|error| error.to_string())
                .and_then(|(path, saved)| roster.add(path, saved))
            {
                Ok(RaceRosterAdd::Added) => added += 1,
                Ok(RaceRosterAdd::Duplicate) => duplicates += 1,
                Err(error) => failures.push(error),
            }
        }
        race.prepare(roster.len());
        checkpoints.status = if failures.is_empty() {
            format!("Added {added} latest architecture checkpoint(s); {duplicates} already present")
        } else {
            format!(
                "Added {added}; {duplicates} duplicate(s); {} failed: {}",
                failures.len(),
                failures.join(" | ")
            )
        };
    }
}

fn add_checkpoint_to_race(
    checkpoints: &mut CheckpointStore,
    roster: &mut RaceRoster,
    race: &mut RaceSession,
    index: usize,
) {
    match checkpoints
        .load_for_race(index)
        .map_err(|error| error.to_string())
        .and_then(|(path, saved)| roster.add(path, saved))
    {
        Ok(RaceRosterAdd::Added) => {
            race.prepare(roster.len());
            checkpoints.status = "Checkpoint added to Race roster".into();
        }
        Ok(RaceRosterAdd::Duplicate) => {
            checkpoints.status = "Checkpoint is already in the Race roster".into();
        }
        Err(error) => checkpoints.status = format!("Add to Race failed: {error}"),
    }
}

fn draw_checkpoint_entry(ui: &mut egui::Ui, entry: &CheckpointEntry) -> (bool, bool) {
    let mut load_requested = false;
    let mut race_requested = false;
    let title = if entry.summary.is_some() {
        entry.filename.clone()
    } else {
        format!("{} [invalid]", entry.filename)
    };
    egui::CollapsingHeader::new(title)
        .id_salt(("checkpoint", &entry.path))
        .show(ui, |ui| {
            // Keep expanded metadata inside the fixed-width dashboard. A Grid
            // without a finite maximum column width asks egui to grow to fit
            // long values, which visually pushes content across the panel edge.
            ui.set_max_width(ui.available_width());
            if let Some(summary) = &entry.summary {
                let column_width = ((ui.available_width() - 12.0) * 0.5).max(80.0);
                egui::Grid::new(("checkpoint_metadata", &entry.path))
                    .num_columns(2)
                    .min_col_width(0.0)
                    .max_col_width(column_width)
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
                            format!("{:.2} u/s", summary.training.average_useful_progress_speed),
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
                        summary.evaluation.progress_speed_half_saturation
                    ));
                } else {
                    ui.small(format!(
                        "Useful speed (legacy): hard clamp at {:.0}u/s",
                        summary.evaluation.progress_speed_half_saturation
                    ));
                }
                ui.horizontal(|ui| {
                    load_requested = ui.button("Load / Test").clicked();
                    race_requested = ui.button("Add to Race").clicked();
                });
            } else if let Some(error) = &entry.error {
                ui.colored_label(egui::Color32::LIGHT_RED, error);
            }
        });
    (load_requested, race_requested)
}

fn row(ui: &mut egui::Ui, label: &str, value: String) {
    ui.label(label);
    ui.monospace(value);
    ui.end_row();
}
