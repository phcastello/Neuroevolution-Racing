use bevy::prelude::*;
use bevy_egui::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};

#[derive(Clone, Copy, Debug)]
pub struct FitnessSample {
    pub generation: u32,
    pub best_fitness: f64,
    pub average_fitness: f64,
}

#[derive(Resource, Debug)]
pub struct FitnessHistory {
    pub samples: Vec<FitnessSample>,
    pub is_mock: bool,
}

impl Default for FitnessHistory {
    fn default() -> Self {
        // Clearly labeled preview data exercises the plot before a GA exists.
        Self {
            samples: vec![
                FitnessSample {
                    generation: 0,
                    best_fitness: 1.0,
                    average_fitness: 0.6,
                },
                FitnessSample {
                    generation: 1,
                    best_fitness: 2.2,
                    average_fitness: 1.2,
                },
                FitnessSample {
                    generation: 2,
                    best_fitness: 2.8,
                    average_fitness: 1.9,
                },
                FitnessSample {
                    generation: 3,
                    best_fitness: 4.1,
                    average_fitness: 2.5,
                },
            ],
            is_mock: true,
        }
    }
}

pub fn draw_fitness_plot(ui: &mut egui::Ui, history: &FitnessHistory) {
    let best = PlotPoints::from_iter(
        history
            .samples
            .iter()
            .map(|s| [s.generation as f64, s.best_fitness]),
    );
    let average = PlotPoints::from_iter(
        history
            .samples
            .iter()
            .map(|s| [s.generation as f64, s.average_fitness]),
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
