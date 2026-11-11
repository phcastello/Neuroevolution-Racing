//! One Bevy application, one renderer and independent controller adapters.
use std::path::PathBuf;

use avian2d::prelude::*;
use bevy::{
    asset::AssetPlugin, diagnostic::FrameTimeDiagnosticsPlugin, prelude::*,
    window::WindowResolution,
};
use bevy_egui::{EguiContexts, EguiPlugin, EguiPrimaryContextPass, egui};
use racing_rl::persistence::{self, Checkpoint};

use crate::{
    display::DisplayPlugin,
    rendering::RenderingPlugin,
    rl::viewer::{self, Viewer},
    simulation::{self, SimulationPlugin},
    ui::{
        DashboardPlugin, experiment_dashboard::ExperimentDashboard,
        search_dashboard::SearchDashboard,
    },
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Mode {
    #[default]
    Menu,
    Ga,
    Ppo,
    Aco,
    Experiments,
}

#[derive(Resource, Default)]
pub(crate) struct Shell {
    pub mode: Mode,
    pending: Option<Mode>,
    pub policy_path: String,
    pub track: String,
    pub cars: usize,
    pub load_policy: bool,
    pub status: String,
}

pub(crate) fn ga_active(world: &World) -> bool {
    world
        .get_resource::<Shell>()
        .is_none_or(|shell| shell.mode == Mode::Ga)
}
pub(crate) fn playback_active(world: &World) -> bool {
    world.contains_non_send::<Viewer>()
        && world
            .get_resource::<Shell>()
            .is_none_or(|shell| shell.mode == Mode::Ppo)
}
pub(crate) fn request_menu(ctx: &egui::Context) {
    ctx.data_mut(|data| data.insert_temp(egui::Id::new("visual_menu_requested"), true));
}

fn base_app() -> App {
    let mut app = App::new();
    app.insert_resource(ClearColor(Color::srgb(0.035, 0.075, 0.055)))
        .add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    file_path: crate::asset_directory(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Neuroevolution Racing · GA · PPO · ACO · Experimentos".into(),
                        resolution: WindowResolution::new(1400, 850),
                        resizable: true,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins(PhysicsPlugins::default().with_length_unit(20.0))
        .insert_resource(Gravity::ZERO)
        .add_plugins((FrameTimeDiagnosticsPlugin::default(), EguiPlugin::default()))
        .insert_resource(Shell {
            cars: 4,
            ..default()
        })
        .init_resource::<SearchDashboard>()
        .init_resource::<ExperimentDashboard>()
        .add_plugins((
            SimulationPlugin::interactive(),
            DisplayPlugin,
            RenderingPlugin,
            DashboardPlugin,
            viewer::PlaybackPlugin,
        ))
        .add_systems(Update, apply_navigation)
        .add_systems(EguiPrimaryContextPass, shell_ui);
    app
}

pub(crate) fn run() -> Result<(), String> {
    let exit = base_app().run();
    if exit.is_success() {
        Ok(())
    } else {
        Err(format!("visual application exited: {exit:?}"))
    }
}

pub(crate) fn run_with_policy(
    saved: Checkpoint,
    path: PathBuf,
    cars: usize,
    track: Option<String>,
    frames: Option<u64>,
) -> Result<(), String> {
    let mut app = base_app();
    viewer::attach(
        app.world_mut(),
        saved,
        path.clone(),
        cars,
        track.as_deref(),
        frames,
    )
    .map_err(|e| e.to_string())?;
    {
        let mut shell = app.world_mut().resource_mut::<Shell>();
        shell.mode = Mode::Ppo;
        shell.policy_path = path.display().to_string();
        shell.cars = cars;
    }
    let exit = app.run();
    if exit.is_success() {
        Ok(())
    } else {
        Err(format!("visual application exited: {exit:?}"))
    }
}

fn apply_navigation(world: &mut World) {
    let (pending, load, path, cars, mut track) = {
        let mut shell = world.resource_mut::<Shell>();
        (
            shell.pending.take(),
            std::mem::take(&mut shell.load_policy),
            shell.policy_path.clone(),
            shell.cars,
            shell.track.clone(),
        )
    };
    // Register generated definitions only in the training partition for frozen playback.
    let selected = world
        .resource_mut::<SearchDashboard>()
        .pending_track
        .take()
        .or_else(|| {
            world
                .resource_mut::<ExperimentDashboard>()
                .pending_track
                .take()
        });
    if let Some(definition) = selected {
        let result = if definition.id.starts_with("generated_") {
            world
                .resource_mut::<simulation::TrackLibrary>()
                .insert_generated(definition.clone())
        } else {
            Ok(())
        };
        match result.and_then(|()| simulation::Track::from_definition(&definition)) {
            Ok(selected_track) => {
                track = definition.id.clone();
                world.resource_mut::<Shell>().track = track.clone();
                world.insert_resource(selected_track);
                world
                    .resource_mut::<crate::display::CameraViewState>()
                    .request_fit();
            }
            Err(error) => world.resource_mut::<Shell>().status = error.to_string(),
        }
    }
    if load {
        let result = persistence::load::<Checkpoint>(&PathBuf::from(&path)).and_then(|saved| {
            viewer::attach(
                world,
                saved,
                PathBuf::from(&path),
                cars,
                (!track.is_empty()).then_some(track.as_str()),
                None,
            )
        });
        match result {
            Ok(()) => {
                let mut shell = world.resource_mut::<Shell>();
                shell.mode = Mode::Ppo;
                shell.status.clear();
            }
            Err(error) => world.resource_mut::<Shell>().status = error.to_string(),
        }
    }
    if pending.is_some() || load {
        if let Some(restore) = world
            .resource_mut::<simulation::TrainingFastForward>()
            .cancel()
        {
            let mut time = world.resource_mut::<Time<Virtual>>();
            time.set_relative_speed(restore.speed);
            time.set_max_delta(restore.max_delta);
        }
    }
    if let Some(mode) = pending {
        let result = match mode {
            Mode::Ga => {
                let playback = *world.resource::<simulation::PlaybackState>();
                {
                    let mut time = world.resource_mut::<Time<Virtual>>();
                    time.set_relative_speed(playback.speed);
                    if playback.paused {
                        time.pause();
                    } else {
                        time.unpause();
                    }
                }
                simulation::restore_visual_ga(world)
            }
            Mode::Ppo if world.contains_non_send::<Viewer>() => {
                viewer::resume(world).map_err(|e| e.to_string())
            }
            _ => {
                simulation::clear_visual_cars(world);
                Ok(())
            }
        };
        let mut shell = world.resource_mut::<Shell>();
        if let Err(error) = result {
            shell.status = error;
        } else {
            shell.mode = mode;
        }
    }
    let mode = world.resource::<Shell>().mode;
    if !matches!(mode, Mode::Ga | Mode::Ppo) {
        world.resource_mut::<Time<Virtual>>().unpause();
    }
}

fn shell_ui(
    mut contexts: EguiContexts,
    mut shell: ResMut<Shell>,
    mut search: ResMut<SearchDashboard>,
    mut experiments: ResMut<ExperimentDashboard>,
    has_policy: Option<NonSend<Viewer>>,
    mut camera: ResMut<crate::display::CameraViewState>,
    mut display: ResMut<crate::display::DisplaySettings>,
    diagnostics: Res<bevy::diagnostic::DiagnosticsStore>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    if ctx
        .data_mut(|data| data.remove_temp::<bool>(egui::Id::new("visual_menu_requested")))
        .unwrap_or(false)
    {
        shell.pending = Some(Mode::Menu);
    }
    match shell.mode {
        Mode::Ga => {}
        Mode::Ppo if has_policy.is_some() => {}
        Mode::Menu => {
            egui::Window::new("Neuroevolution Racing")
                .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO).collapsible(false).resizable(false)
                .show(ctx, |ui| {
                    ui.heading("Escolha o modo");
                    ui.label("GA treina visualmente. PPO observa políticas congeladas. ACO reproduz a busca de pistas.");
                    ui.add_space(12.0);
                    for (mode, label) in [(Mode::Ga,"GA · evolução e corridas"),(Mode::Ppo,"PPO · avaliar checkpoint"),(Mode::Aco,"ACO · decisões e pistas"),(Mode::Experiments,"Experimentos · resultados e configurações")] {
                        if ui.add_sized([420.0, 38.0], egui::Button::new(label)).clicked() { shell.pending = Some(mode); }
                    }
                    if !shell.status.is_empty() { ui.colored_label(egui::Color32::LIGHT_RED, &shell.status); }
                });
        }
        Mode::Ppo => {
            crate::ui::draw_dashboard(ctx, "PPO · política congelada · inferência CPU", |ui| {
                ui.heading("Abrir política PPO");
                policy_input(ui, &mut shell);
                ui.small("O treinamento permanece headless. O checkpoint determina observações, arquitetura e física.");
            });
        }
        Mode::Aco => search.show(ctx, &mut shell),
        Mode::Experiments => experiments.show(ctx, &mut shell),
    }
    if matches!(shell.mode, Mode::Aco | Mode::Experiments) {
        egui::Window::new("Câmera e exibição")
            .default_pos([750.0, 600.0])
            .default_open(false)
            .show(ctx, |ui| {
                crate::ui::draw_camera_controls(ui, &mut camera, "Follow Selected");
                crate::ui::draw_display_controls(ui, &mut display, &diagnostics);
            });
    }
    Ok(())
}

pub(crate) fn policy_input(ui: &mut egui::Ui, shell: &mut Shell) {
    ui.label("Checkpoint (.ron)");
    ui.text_edit_singleline(&mut shell.policy_path);
    ui.add(egui::Slider::new(&mut shell.cars, 1..=16).text("Veículos"));
    ui.label("ID da pista (vazio: checkpoint)");
    ui.text_edit_singleline(&mut shell.track);
    if ui.button("Carregar política").clicked() {
        shell.load_policy = true;
    }
    if !shell.status.is_empty() {
        ui.colored_label(egui::Color32::LIGHT_RED, &shell.status);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mode_roundtrips_clear_old_cars_and_disable_training_without_display() {
        let mut world = World::new();
        world.insert_resource(Shell::default());
        world.init_resource::<SearchDashboard>();
        world.init_resource::<ExperimentDashboard>();
        world.init_resource::<simulation::TrainingFastForward>();
        world.init_resource::<Time<Virtual>>();
        for mode in [
            Mode::Aco,
            Mode::Experiments,
            Mode::Menu,
            Mode::Aco,
            Mode::Menu,
        ] {
            world.spawn(simulation::Car { id: 0 });
            world.resource_mut::<Shell>().pending = Some(mode);
            apply_navigation(&mut world);
            assert_eq!(world.resource::<Shell>().mode, mode);
            assert_eq!(world.query::<&simulation::Car>().iter(&world).count(), 0);
            assert!(!ga_active(&world));
            assert!(!playback_active(&world));
            assert!(
                world
                    .resource::<simulation::TrainingFastForward>()
                    .target_generation
                    .is_none()
            );
        }
    }
    #[test]
    fn menu_does_not_run_ga_or_ppo_and_workers_keep_original_behavior() {
        let mut world = World::new();
        assert!(ga_active(&world));
        assert!(!playback_active(&world));
        world.insert_resource(Shell::default());
        assert!(!ga_active(&world));
        assert!(!playback_active(&world));
        world.resource_mut::<Shell>().mode = Mode::Ga;
        assert!(ga_active(&world));
        world.resource_mut::<Shell>().mode = Mode::Aco;
        assert!(!ga_active(&world));
    }
}
