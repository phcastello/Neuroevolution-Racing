pub(crate) mod experiment_dashboard;
pub(crate) mod search_dashboard;
pub(crate) use dashboard::draw_dashboard_with_toolbar;
pub(crate) use dashboard::{draw_camera_controls, draw_dashboard, draw_display_controls};
mod checkpoint_browser;
mod dashboard;
mod fitness_plot;
mod network_view;
pub(crate) mod ppo_checkpoint_browser;
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
        app.add_systems(
            EguiPrimaryContextPass,
            dashboard_system.run_if(crate::visual::ga_active),
        );
    }
}
