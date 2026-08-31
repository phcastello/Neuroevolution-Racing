mod display;
mod experiment;
mod rendering;
mod simulation;
mod ui;

use std::{path::PathBuf, process::ExitCode, time::Duration};

use avian2d::prelude::*;
use bevy::{
    app::{ScheduleRunnerPlugin, TaskPoolOptions, TaskPoolPlugin},
    asset::AssetPlugin,
    diagnostic::FrameTimeDiagnosticsPlugin,
    prelude::*,
    window::WindowResolution,
};
use bevy_egui::EguiPlugin;
use clap::{Parser, Subcommand};
use display::DisplayPlugin;
use experiment::{WorkerOptions, WorkerRuntime, mark_run_failed, prepare_worker, run_batch};
use rendering::RenderingPlugin;
use simulation::{SimulationPlugin, TrackLibrary, TrainingFastForward, TrainingSetup};
use ui::DashboardPlugin;

#[derive(Parser)]
#[command(name = "neuroevolution-racing", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<CommandLine>,
}

#[derive(Subcommand)]
enum CommandLine {
    /// Launches one isolated worker process per architecture.
    Batch { config: PathBuf },
    /// Runs one architecture with no window, UI, camera, or rendering plugins.
    Worker {
        #[arg(long)]
        architecture: String,
        #[arg(long)]
        target_generation: usize,
        #[arg(long)]
        seed: u64,
        #[arg(long, default_value = "results")]
        results_root: PathBuf,
        #[arg(long, default_value_t = 500)]
        population_size: usize,
        #[arg(long, default_value_t = 1)]
        worker_threads: usize,
        #[arg(long)]
        resume: bool,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        None => {
            run_interactive();
            Ok(())
        }
        Some(CommandLine::Batch { config }) => run_batch(&config),
        Some(CommandLine::Worker {
            architecture,
            target_generation,
            seed,
            results_root,
            population_size,
            worker_threads,
            resume,
        }) => experiment::parse_architecture(&architecture).and_then(|architecture| {
            run_worker(WorkerOptions {
                architecture,
                target_generation,
                seed,
                results_root,
                population_size,
                worker_threads,
                resume,
            })
        }),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run_interactive() {
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
            SimulationPlugin::interactive(),
            DisplayPlugin,
            RenderingPlugin,
            DashboardPlugin,
        ))
        .run();
}

fn run_worker(options: WorkerOptions) -> Result<(), String> {
    let library = TrackLibrary::load_default().map_err(|error| error.to_string())?;
    let prepared = prepare_worker(&options, &library)?;
    let run_directory = prepared.run_directory.clone();
    let result = (|| {
        let current_generation = prepared
            .resume_checkpoint
            .as_ref()
            .map_or(0, |checkpoint| checkpoint.generation);
        if current_generation >= options.target_generation {
            let mut manifest = prepared.manifest.clone();
            manifest.status = experiment::RunStatus::Completed;
            manifest.completed_generation = Some(current_generation);
            experiment::write_run_manifest(&prepared.run_directory.join("run.ron"), &manifest)?;
            return Ok(());
        }
        let runtime = WorkerRuntime::new(&prepared)?;
        let mut fast_forward = TrainingFastForward::default();
        if !fast_forward.start(
            options.target_generation,
            current_generation,
            1.0,
            Duration::from_millis(250),
        ) {
            return Err(fast_forward.status);
        }
        let setup = TrainingSetup {
            population_size: options.population_size,
            architecture: options.architecture.clone(),
            seed: options.seed,
            evaluation_config: prepared.manifest.evaluation_config.clone(),
            checkpoint_directory: prepared.run_directory.join("bests_by_gen"),
            resume_checkpoint: prepared.resume_checkpoint.clone(),
        };
        let exit = App::new()
            .add_plugins(
                MinimalPlugins
                    .set(TaskPoolPlugin {
                        task_pool_options: TaskPoolOptions::with_num_threads(
                            options.worker_threads.max(1),
                        ),
                    })
                    .set(ScheduleRunnerPlugin::run_loop(Duration::ZERO)),
            )
            .add_plugins(TransformPlugin)
            .add_plugins(PhysicsPlugins::default().with_length_unit(20.0))
            .insert_resource(Gravity::ZERO)
            .insert_resource(runtime)
            .insert_resource(fast_forward)
            .add_plugins(SimulationPlugin::headless(setup))
            .run();
        if exit.is_success() {
            Ok(())
        } else {
            Err(format!("headless Bevy app exited with {exit:?}"))
        }
    })();
    if result.is_err() && run_directory.join("run.ron").exists() {
        mark_run_failed(&run_directory);
    }
    result
}
