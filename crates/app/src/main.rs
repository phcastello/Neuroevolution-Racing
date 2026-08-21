mod display;
mod rendering;
mod simulation;
mod ui;

use avian2d::prelude::*;
use bevy::{
    asset::AssetPlugin, diagnostic::FrameTimeDiagnosticsPlugin, prelude::*,
    window::WindowResolution,
};
use bevy_egui::EguiPlugin;
use display::DisplayPlugin;
use rendering::RenderingPlugin;
use simulation::SimulationPlugin;
use ui::DashboardPlugin;

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.035, 0.075, 0.055)))
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: "../../assets".into(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Neuroevolution Racing".into(),
                        resolution: WindowResolution::new(1400, 850),
                        resizable: true,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(PhysicsPlugins::default().with_length_unit(20.0))
        .insert_resource(Gravity::ZERO)
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(EguiPlugin::default())
        .add_plugins((
            SimulationPlugin,
            DisplayPlugin,
            RenderingPlugin,
            DashboardPlugin,
        ))
        .run();
}
