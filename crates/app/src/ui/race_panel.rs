use std::path::Path;

use bevy_egui::egui;

use crate::simulation::{
    RacePhase, RaceRoster, RaceSession, RaceStanding, RaceStatus, Track, export_race_results,
    sort_race_standings,
};

pub fn draw_race_panel(
    ui: &mut egui::Ui,
    roster: &mut RaceRoster,
    session: &mut RaceSession,
    track: &Track,
    standings: &mut Vec<RaceStanding>,
) {
    ui.separator();
    ui.heading("Race roster");
    if roster.is_empty() {
        ui.colored_label(
            egui::Color32::LIGHT_YELLOW,
            "Add checkpoints to the race roster",
        );
    }

    let mut remove = None;
    for (index, entry) in roster.entries().iter().enumerate() {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(
                egui::Color32::from_rgb(entry.color.0[0], entry.color.0[1], entry.color.0[2]),
                "●",
            );
            let response = ui.label(format!(
                "{}  gen {}  ·  {}",
                entry.architecture, entry.generation, entry.display_name
            ));
            response.on_hover_ui(|ui| {
                ui.strong("CHECKPOINT METADATA");
                ui.monospace(&entry.source_display);
                ui.label(format!(
                    "training fitness: {:.5}",
                    entry.saved.training_metadata.champion_training_fitness
                ));
                ui.label(format!(
                    "population average: {:.5}",
                    entry.saved.training_metadata.population_average_fitness
                ));
                ui.label(format!(
                    "training completion: {:.1}%",
                    entry.saved.training_metadata.completion_rate * 100.0
                ));
                ui.label(format!(
                    "validation: {} · {:.5} · {}",
                    entry.saved.validation_metadata.track_id,
                    entry.saved.validation_metadata.score,
                    entry.saved.validation_metadata.finish_reason.label()
                ));
            });
            if ui.small_button("Remove").clicked() {
                remove = Some(index);
            }
        });
    }
    if let Some(index) = remove
        && roster.remove(index)
    {
        session.prepare(roster.len());
    }
    ui.horizontal(|ui| {
        if ui
            .add_enabled(!roster.is_empty(), egui::Button::new("Clear roster"))
            .clicked()
        {
            roster.clear();
            session.prepare(0);
        }
        let label = if session.phase == RacePhase::Ready {
            "Start Race"
        } else {
            "Restart Race"
        };
        if ui
            .add_enabled(!roster.is_empty(), egui::Button::new(label))
            .clicked()
        {
            session.start_or_restart(roster.len());
        }
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut session.show_trails, "Show race trails");
        ui.checkbox(&mut session.auto_select_leader, "Auto-select leader");
    });
    ui.horizontal(|ui| {
        ui.label("Safety timeout:");
        ui.add(
            egui::DragValue::new(&mut session.safety_timeout)
                .range(30.0..=3600.0)
                .suffix(" s"),
        );
    });
    ui.small(&session.status);

    sort_race_standings(standings);
    if !standings.is_empty() {
        ui.separator();
        ui.heading(match session.phase {
            RacePhase::Ready => "Ready lineup",
            RacePhase::Running => "Live leaderboard",
            RacePhase::Finished => "Final leaderboard",
        });
        for (rank, standing) in standings.iter().enumerate() {
            let Some(entry) = roster.entries().get(standing.roster_index) else {
                continue;
            };
            ui.horizontal_wrapped(|ui| {
                ui.monospace(format!("{:>2}.", rank + 1));
                ui.colored_label(
                    egui::Color32::from_rgb(entry.color.0[0], entry.color.0[1], entry.color.0[2]),
                    "●",
                );
                ui.label(format!("{} · gen {}", entry.architecture, entry.generation));
                match standing.metrics.status {
                    RaceStatus::Completed => {
                        ui.strong(format!(
                            "COMPLETED {:.2}s",
                            standing.metrics.finish_time.unwrap_or_default()
                        ));
                    }
                    RaceStatus::Running => {
                        ui.monospace(format!(
                            "{:.1}% · {:.1}u/s",
                            standing.metrics.normalized_progress * 100.0,
                            standing.metrics.useful_progress_speed
                        ));
                    }
                    RaceStatus::Collision | RaceStatus::Timeout => {
                        ui.strong(format!(
                            "{} {:.1}%",
                            standing.metrics.status.label(),
                            standing.metrics.normalized_progress * 100.0
                        ));
                    }
                }
                if ui.small_button("Inspect").clicked() {
                    session.auto_select_leader = false;
                    session.inspected_roster_index = Some(standing.roster_index);
                }
            });
        }
    }

    if ui
        .add_enabled(
            session.phase == RacePhase::Finished && !standings.is_empty(),
            egui::Button::new("Export results"),
        )
        .clicked()
    {
        session.export_status = match export_race_results(
            Path::new("results/races"),
            &track.definition.id,
            roster,
            standings,
        ) {
            Ok(path) => format!("Exported {}", path.display()),
            Err(error) => format!("Export failed: {error}"),
        };
    }
    if !session.export_status.is_empty() {
        ui.small(&session.export_status);
    }
}
