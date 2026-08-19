mod dashboard;
mod fitness_plot;
mod network_view;

use bevy::prelude::*;
use bevy_egui::EguiPrimaryContextPass;
use dashboard::dashboard_system;

pub use fitness_plot::FitnessHistory;

pub struct DashboardPlugin;

impl Plugin for DashboardPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FitnessHistory>()
            .add_systems(EguiPrimaryContextPass, dashboard_system);
    }
}
