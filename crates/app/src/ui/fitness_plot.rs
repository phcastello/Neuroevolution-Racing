use bevy_egui::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};

use crate::simulation::GenerationStats;

pub fn draw_fitness_plot(ui: &mut egui::Ui, history: &[GenerationStats]) {
    let best = PlotPoints::from_iter(
        history
            .iter()
            .map(|s| [s.generation as f64, s.best_fitness as f64]),
    );
    let average = PlotPoints::from_iter(
        history
            .iter()
            .map(|s| [s.generation as f64, s.average_fitness as f64]),
    );

    Plot::new("fitness_history")
        .height(170.0)
        .legend(Legend::default())
        .allow_drag(false)
        .allow_scroll(false)
        .show(ui, |plot_ui| {
            plot_ui.line(Line::new("Best", best).color(egui::Color32::LIGHT_GREEN));
            plot_ui.line(Line::new("Average", average).color(egui::Color32::LIGHT_BLUE));
        });
}
