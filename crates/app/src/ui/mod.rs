mod checkpoint_browser;
mod dashboard;
mod fitness_plot;
mod network_view;
mod ppo_checkpoint_browser;
mod ppo_dashboard;
pub(crate) mod ppo_network_view;
mod race_panel;

use bevy::prelude::*;
use bevy_egui::EguiPrimaryContextPass;
use dashboard::dashboard_system;
pub(crate) use ppo_dashboard::PpoDashboardPlugin;

pub struct DashboardPlugin;

impl Plugin for DashboardPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(EguiPrimaryContextPass, dashboard_system);
    }
}
