use std::{
    collections::VecDeque,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    process::{Child, Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use bevy::prelude::Resource;
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};

use crate::simulation::{
    CheckpointStore, CompletedChampion, EvaluationConfig, SavedGeneticConfig, SimulationConfig,
    TrackLibrary, TrainingCheckpoint, TrainingState, load_saved_network, racing_architecture,
};
use neuroevolution::genetic::Config as GeneticConfig;

pub const RUN_FORMAT_VERSION: u32 = 1;
const TRAINING_CHECKPOINT_KEEP: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RunStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Interrupted,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    pub format_version: u32,
    #[serde(default)]
    pub name: String,
    pub architecture: Vec<usize>,
    pub parameter_count: usize,
    pub seed: u64,
    pub population_size: usize,
    pub target_generation: usize,
    pub created_at: u64,
    pub genetic_config: SavedGeneticConfig,
    pub evaluation_config: EvaluationConfig,
    pub simulation_config: SimulationConfig,
    pub sensor_range: f32,
    pub training_tracks: Vec<String>,
    pub validation_tracks: Vec<String>,
    pub rng: String,
    pub app_version: String,
    pub checkpoint_version: u32,
    pub status: RunStatus,
    pub completed_generation: Option<usize>,
}

impl RunManifest {
    pub fn new(
        name: String,
        architecture: Vec<usize>,
        target_generation: usize,
        core_genetic_config: GeneticConfig,
        library: &TrackLibrary,
    ) -> Result<Self, String> {
        let architecture_model = racing_architecture(architecture.clone())?;
        core_genetic_config.validate().map_err(str::to_string)?;
        if core_genetic_config.genome_length != architecture_model.parameter_count() {
            return Err("genetic genome_length does not match the architecture".into());
        }
        let genetic_config = SavedGeneticConfig::from_config(&core_genetic_config);
        let simulation_config = SimulationConfig {
            population_size: core_genetic_config.population_size,
            ..SimulationConfig::default()
        };
        Ok(Self {
            format_version: RUN_FORMAT_VERSION,
            name,
            architecture,
            parameter_count: architecture_model.parameter_count(),
            seed: core_genetic_config.seed,
            population_size: core_genetic_config.population_size,
            target_generation,
            created_at: unix_timestamp()?,
            genetic_config,
            evaluation_config: EvaluationConfig::default(),
            sensor_range: simulation_config.sensor_max_distance,
            simulation_config,
            training_tracks: library
                .training_tracks()
                .map(|track| track.id.clone())
                .collect(),
            validation_tracks: library
                .validation_tracks()
                .map(|track| track.id.clone())
                .collect(),
            rng: crate::simulation::TRAINING_RNG_ID.into(),
            app_version: env!("CARGO_PKG_VERSION").into(),
            checkpoint_version: crate::simulation::TRAINING_CHECKPOINT_FORMAT_VERSION,
            status: RunStatus::Pending,
            completed_generation: None,
        })
    }

    pub fn validate_compatible(&self, requested: &Self) -> Result<(), String> {
        let mismatch = self.format_version != requested.format_version
            || self.name != requested.name
            || self.architecture != requested.architecture
            || self.parameter_count != requested.parameter_count
            || self.seed != requested.seed
            || self.population_size != requested.population_size
            || self.genetic_config != requested.genetic_config
            || self.evaluation_config != requested.evaluation_config
            || self.simulation_config != requested.simulation_config
            || self.sensor_range != requested.sensor_range
            || self.training_tracks != requested.training_tracks
            || self.validation_tracks != requested.validation_tracks
            || self.rng != requested.rng
            || self.checkpoint_version != requested.checkpoint_version;
        if mismatch {
            Err("run.ron is incompatible with the requested scientific configuration".into())
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub target_generation: usize,
    pub max_parallel_runs: usize,
    #[serde(default)]
    pub resume_existing: bool,
    #[serde(alias = "base_seed")]
    pub seed: u64,
    #[serde(default = "default_population_size")]
    pub population_size: usize,
    #[serde(default = "default_results_root")]
    pub results_root: PathBuf,
    #[serde(default)]
    pub worker_threads: Option<usize>,
    #[serde(default)]
    pub architectures: Vec<Vec<usize>>,
    #[serde(default)]
    pub runs: Vec<ExperimentRunConfig>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentRunConfig {
    pub name: String,
    pub architecture: Vec<usize>,
    #[serde(default = "default_crossover_probability")]
    pub crossover_probability: f32,
    #[serde(default = "default_mutation_probability")]
    pub mutation_probability: f32,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct ResolvedExperimentRun {
    pub name: String,
    pub architecture: Vec<usize>,
    pub seed: u64,
    pub population_size: usize,
    pub target_generation: usize,
    pub crossover_probability: f32,
    pub mutation_probability: f32,
}

impl ResolvedExperimentRun {
    pub fn directory(&self, results_root: &Path) -> PathBuf {
        results_root.join(&self.name)
    }

    pub fn genetic_config(&self) -> Result<GeneticConfig, String> {
        let architecture = racing_architecture(self.architecture.clone())?;
        let config = GeneticConfig {
            population_size: self.population_size,
            genome_length: architecture.parameter_count(),
            crossover_probability: self.crossover_probability,
            mutation_probability: self.mutation_probability,
            seed: self.seed,
            ..GeneticConfig::default()
        };
        config.validate().map_err(str::to_string)?;
        Ok(config)
    }
}

fn default_population_size() -> usize {
    500
}

fn default_results_root() -> PathBuf {
    PathBuf::from("results")
}

fn default_crossover_probability() -> f32 {
    GeneticConfig::default().crossover_probability
}

fn default_mutation_probability() -> f32 {
    GeneticConfig::default().mutation_probability
}

impl ExperimentConfig {
    pub fn load(path: &Path) -> Result<Self, String> {
        let contents = fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let config: Self = ron::from_str(&contents)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), String> {
        if self.target_generation == 0 {
            return Err("target_generation must be greater than zero".into());
        }
        if self.max_parallel_runs == 0 || self.population_size == 0 {
            return Err("max_parallel_runs and population_size must be greater than zero".into());
        }
        if self.worker_threads == Some(0) {
            return Err("worker_threads must be greater than zero".into());
        }
        if self.architectures.is_empty() && self.runs.is_empty() {
            return Err("either architectures or runs must not be empty".into());
        }
        if !self.architectures.is_empty() && !self.runs.is_empty() {
            return Err(
                "architectures and runs are alternative batch formats; use only one".into(),
            );
        }
        let mut names = std::collections::BTreeSet::new();
        for run in self.resolved_runs()? {
            validate_run_name(&run.name)?;
            run.genetic_config()?;
            if !names.insert(run.name.clone()) {
                return Err(format!("duplicate run name {}", run.name));
            }
        }
        Ok(())
    }

    pub fn resolved_runs(&self) -> Result<Vec<ResolvedExperimentRun>, String> {
        if !self.runs.is_empty() {
            return Ok(self
                .runs
                .iter()
                .map(|run| ResolvedExperimentRun {
                    name: run.name.clone(),
                    architecture: run.architecture.clone(),
                    seed: run.seed.unwrap_or(self.seed),
                    population_size: self.population_size,
                    target_generation: self.target_generation,
                    crossover_probability: run.crossover_probability,
                    mutation_probability: run.mutation_probability,
                })
                .collect());
        }
        Ok(self
            .architectures
            .iter()
            .map(|architecture| ResolvedExperimentRun {
                name: architecture_slug(architecture),
                architecture: architecture.clone(),
                seed: architecture_seed(self.seed, architecture),
                population_size: self.population_size,
                target_generation: self.target_generation,
                crossover_probability: default_crossover_probability(),
                mutation_probability: default_mutation_probability(),
            })
            .collect())
    }
}

fn validate_run_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(format!(
            "invalid run name {name:?}; use only ASCII letters, digits, '-' and '_'"
        ));
    }
    Ok(())
}

pub fn architecture_slug(architecture: &[usize]) -> String {
    architecture
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("_")
}

pub fn parse_architecture(value: &str) -> Result<Vec<usize>, String> {
    let architecture = value
        .split(',')
        .map(|part| {
            part.trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid architecture layer {part:?}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    racing_architecture(architecture.clone())?;
    Ok(architecture)
}

pub fn architecture_seed(base_seed: u64, architecture: &[usize]) -> u64 {
    // Stable FNV-1a, intentionally independent of Rust's randomized HashMap hasher.
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ base_seed;
    for byte in architecture_slug(architecture).bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

struct RunningWorker {
    name: String,
    child: Child,
}

pub fn run_batch(config_path: &Path) -> Result<(), String> {
    let config = ExperimentConfig::load(config_path)?;
    let runs = config.resolved_runs()?;
    let library = TrackLibrary::load_default().map_err(|error| error.to_string())?;
    preflight_batch_runs(&config, &runs, &library)?;
    let executable = std::env::current_exe()
        .map_err(|error| format!("failed to locate current executable: {error}"))?;
    let derived_threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .checked_div(config.max_parallel_runs)
        .unwrap_or(1)
        .max(1);
    let worker_threads = config.worker_threads.unwrap_or(derived_threads);
    let mut queued = VecDeque::new();
    for run in runs.clone() {
        println!("[QUEUED ] {}", run.name);
        queued.push_back(run);
    }
    let mut running: Vec<RunningWorker> = Vec::new();
    let mut failures = 0usize;
    while !queued.is_empty() || !running.is_empty() {
        while can_start_worker(running.len(), config.max_parallel_runs) && !queued.is_empty() {
            let run = queued.pop_front().unwrap();
            let run_directory = run.directory(&config.results_root);
            let manifest_path = run_directory.join("run.ron");
            let requested = RunManifest::new(
                run.name.clone(),
                run.architecture.clone(),
                run.target_generation,
                run.genetic_config()?,
                &library,
            )?;
            if manifest_path.exists() {
                let existing = load_run_manifest(&manifest_path)?;
                existing.validate_compatible(&requested)?;
            }
            fs::create_dir_all(&run_directory).map_err(|error| {
                format!("failed to create {}: {error}", run_directory.display())
            })?;
            let log = OpenOptions::new()
                .create(true)
                .append(true)
                .open(run_directory.join("worker.log"))
                .map_err(|error| format!("failed to open worker log for {}: {error}", run.name))?;
            let stderr = log
                .try_clone()
                .map_err(|error| format!("failed to clone worker log: {error}"))?;
            let mut command = Command::new(&executable);
            command
                .arg("worker")
                .arg("--name")
                .arg(&run.name)
                .arg("--architecture")
                .arg(
                    run.architecture
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                )
                .arg("--target-generation")
                .arg(run.target_generation.to_string())
                .arg("--seed")
                .arg(run.seed.to_string())
                .arg("--results-root")
                .arg(&config.results_root)
                .arg("--population-size")
                .arg(run.population_size.to_string())
                .arg("--crossover-probability")
                .arg(run.crossover_probability.to_string())
                .arg("--mutation-probability")
                .arg(run.mutation_probability.to_string())
                .arg("--worker-threads")
                .arg(worker_threads.to_string())
                .stdout(Stdio::from(log))
                .stderr(Stdio::from(stderr));
            if config.resume_existing && manifest_path.exists() {
                command.arg("--resume");
            }
            let child = match command.spawn() {
                Ok(child) => child,
                Err(error) => {
                    eprintln!("[FAILED ] {} spawn error={error}", run.name);
                    if manifest_path.exists() {
                        mark_run_failed(&run_directory);
                    } else {
                        let mut failed_manifest = requested;
                        failed_manifest.status = RunStatus::Failed;
                        write_run_manifest(&manifest_path, &failed_manifest)?;
                    }
                    failures += 1;
                    continue;
                }
            };
            println!("[RUNNING] {:<24} pid={}", run.name, child.id());
            running.push(RunningWorker {
                name: run.name,
                child,
            });
        }

        let mut index = 0;
        while index < running.len() {
            match running[index].child.try_wait() {
                Ok(Some(status)) => {
                    let finished = running.swap_remove(index);
                    print_worker_exit(&finished.name, status, &config.results_root);
                    if !status.success() {
                        failures += 1;
                    }
                }
                Ok(None) => index += 1,
                Err(error) => {
                    let failed = running.swap_remove(index);
                    eprintln!("[FAILED ] {} poll error={error}", failed.name);
                    mark_run_failed(&config.results_root.join(&failed.name));
                    failures += 1;
                }
            }
        }
        if !running.is_empty() {
            thread::sleep(Duration::from_millis(200));
        }
    }
    let summary_result = write_batch_summary(&config.results_root, &runs);
    match (failures, summary_result) {
        (0, Ok(())) => Ok(()),
        (0, Err(error)) => Err(error),
        (count, Ok(())) => Err(format!("{count} worker(s) failed")),
        (count, Err(error)) => Err(format!(
            "{count} worker(s) failed and summary generation failed: {error}"
        )),
    }
}

fn preflight_batch_runs(
    config: &ExperimentConfig,
    runs: &[ResolvedExperimentRun],
    library: &TrackLibrary,
) -> Result<(), String> {
    for run in runs {
        let run_directory = run.directory(&config.results_root);
        let manifest_path = run_directory.join("run.ron");
        if !manifest_path.exists() {
            continue;
        }
        if !config.resume_existing {
            return Err(format!(
                "{} already contains an experiment and resume_existing is false",
                run_directory.display()
            ));
        }
        let existing = load_run_manifest(&manifest_path)?;
        let requested = RunManifest::new(
            run.name.clone(),
            run.architecture.clone(),
            run.target_generation,
            run.genetic_config()?,
            library,
        )?;
        existing.validate_compatible(&requested)?;
    }
    Ok(())
}

fn can_start_worker(running: usize, max_parallel_runs: usize) -> bool {
    running < max_parallel_runs
}

fn print_worker_exit(name: &str, status: ExitStatus, results_root: &Path) {
    if status.success() {
        let generation = load_run_manifest(&results_root.join(name).join("run.ron"))
            .ok()
            .and_then(|manifest| manifest.completed_generation)
            .map_or_else(|| "?".into(), |value| value.to_string());
        println!("[DONE   ] {name:<24} generation={generation}");
    } else {
        let manifest_path = results_root.join(name).join("run.ron");
        if let Ok(mut manifest) = load_run_manifest(&manifest_path) {
            manifest.status = RunStatus::Failed;
            let _ = write_run_manifest(&manifest_path, &manifest);
        }
        println!("[FAILED ] {name:<24} exit={status}");
    }
}

fn write_batch_summary(results_root: &Path, runs: &[ResolvedExperimentRun]) -> Result<(), String> {
    fs::create_dir_all(results_root)
        .map_err(|error| format!("failed to create {}: {error}", results_root.display()))?;
    let summary_path = results_root.join("summary.csv");
    let mut output = String::from(
        "execution,name,seed,architecture,crossover_probability,mutation_probability,population_size,generations,best_final_fitness,status\n",
    );
    for (index, run) in runs.iter().enumerate() {
        let run_directory = run.directory(results_root);
        let manifest = load_run_manifest(&run_directory.join("run.ron"));
        let (fitness, status) = match manifest {
            Ok(manifest) if manifest.status == RunStatus::Completed => {
                let result = manifest
                    .completed_generation
                    .and_then(|generation| generation.checked_sub(1))
                    .ok_or_else(|| "completed run has no final evaluated generation".to_string())
                    .and_then(|generation| final_generation_fitness(&run_directory, generation));
                match result {
                    Ok(fitness) => (Some(fitness), "completed"),
                    Err(_) => (None, "invalid_result"),
                }
            }
            Ok(manifest) => (None, run_status_label(&manifest.status)),
            Err(_) => (None, "not_started"),
        };
        let architecture = run
            .architecture
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("-");
        let fitness = fitness.map_or_else(String::new, |value| format!("{value:.9}"));
        output.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{}\n",
            index + 1,
            csv_field(&run.name),
            run.seed,
            architecture,
            run.crossover_probability,
            run.mutation_probability,
            run.population_size,
            run.target_generation,
            fitness,
            status,
        ));
    }
    fs::write(&summary_path, output)
        .map_err(|error| format!("failed to write {}: {error}", summary_path.display()))
}

fn run_status_label(status: &RunStatus) -> &'static str {
    match status {
        RunStatus::Pending => "pending",
        RunStatus::Running => "running",
        RunStatus::Completed => "completed",
        RunStatus::Failed => "failed",
        RunStatus::Interrupted => "interrupted",
    }
}

fn final_generation_fitness(run_directory: &Path, generation: usize) -> Result<f32, String> {
    let checkpoint_directory = run_directory.join("bests_by_gen");
    let entries = fs::read_dir(&checkpoint_directory).map_err(|error| {
        format!(
            "failed to read final checkpoints in {}: {error}",
            checkpoint_directory.display()
        )
    })?;
    let mut matching = Vec::new();
    for path in entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "ron"))
    {
        if let Ok(saved) = load_saved_network(&path)
            && saved.generation == generation
        {
            matching.push((
                saved.saved_at_unix_seconds,
                saved.training_metadata.champion_training_fitness,
            ));
        }
    }
    matching
        .into_iter()
        .max_by_key(|(saved_at, _)| *saved_at)
        .map(|(_, fitness)| fitness)
        .ok_or_else(|| {
            format!("no valid champion checkpoint for final evaluated generation {generation}")
        })
}

#[derive(Clone, Debug)]
pub struct WorkerOptions {
    pub name: String,
    pub architecture: Vec<usize>,
    pub target_generation: usize,
    pub seed: u64,
    pub results_root: PathBuf,
    pub population_size: usize,
    pub crossover_probability: f32,
    pub mutation_probability: f32,
    pub worker_threads: usize,
    pub resume: bool,
}

pub struct PreparedWorker {
    pub run_directory: PathBuf,
    pub manifest: RunManifest,
    pub resume_checkpoint: Option<TrainingCheckpoint>,
    pub resume_messages: Vec<String>,
}

pub fn prepare_worker(
    options: &WorkerOptions,
    library: &TrackLibrary,
) -> Result<PreparedWorker, String> {
    if options.target_generation == 0 || options.population_size == 0 || options.worker_threads == 0
    {
        return Err(
            "target_generation, population_size and worker_threads must be greater than zero"
                .into(),
        );
    }
    let architecture = racing_architecture(options.architecture.clone())?;
    validate_run_name(&options.name)?;
    let genetic_config = GeneticConfig {
        population_size: options.population_size,
        genome_length: architecture.parameter_count(),
        crossover_probability: options.crossover_probability,
        mutation_probability: options.mutation_probability,
        seed: options.seed,
        ..GeneticConfig::default()
    };
    genetic_config.validate().map_err(str::to_string)?;
    let run_directory = options.results_root.join(&options.name);
    let manifest_path = run_directory.join("run.ron");
    let requested = RunManifest::new(
        options.name.clone(),
        options.architecture.clone(),
        options.target_generation,
        genetic_config,
        library,
    )?;
    let (mut manifest, resume_checkpoint, resume_messages) = if options.resume {
        if !manifest_path.exists() {
            return Err(format!(
                "--resume requested but {} does not exist",
                manifest_path.display()
            ));
        }
        let mut existing = load_run_manifest(&manifest_path)?;
        existing.validate_compatible(&requested)?;
        existing.target_generation = options.target_generation;
        let (checkpoint, messages) = load_latest_training_checkpoint(
            &run_directory.join("training_checkpoint"),
            &existing,
            library,
        )?;
        (existing, Some(checkpoint), messages)
    } else {
        if manifest_path.exists() {
            return Err(format!(
                "{} already contains an experiment; use --resume with a compatible run",
                run_directory.display()
            ));
        }
        if run_directory.exists() {
            let unexpected = fs::read_dir(&run_directory)
                .map_err(|error| format!("failed to inspect {}: {error}", run_directory.display()))?
                .filter_map(Result::ok)
                .map(|entry| entry.file_name())
                .any(|name| name != "worker.log");
            if unexpected {
                return Err(format!(
                    "{} is non-empty but has no run.ron; refusing to mix experiment data",
                    run_directory.display()
                ));
            }
        }
        (requested, None, Vec::new())
    };
    fs::create_dir_all(run_directory.join("bests_by_gen"))
        .map_err(|error| format!("failed to create result directories: {error}"))?;
    fs::create_dir_all(run_directory.join("training_checkpoint"))
        .map_err(|error| format!("failed to create result directories: {error}"))?;
    manifest.status = RunStatus::Running;
    write_run_manifest(&manifest_path, &manifest)?;
    Ok(PreparedWorker {
        run_directory,
        manifest,
        resume_checkpoint,
        resume_messages,
    })
}

#[derive(Resource)]
pub struct WorkerRuntime {
    run_directory: PathBuf,
    manifest: RunManifest,
    target_generation: usize,
    metrics: BufWriter<File>,
    log: BufWriter<File>,
    generation_started: Instant,
    total_fixed_ticks: u64,
    generation_start_tick: u64,
    completed: bool,
}

impl WorkerRuntime {
    pub fn new(prepared: &PreparedWorker) -> Result<Self, String> {
        let metrics_path = prepared.run_directory.join("metrics.csv");
        if let Some(checkpoint) = &prepared.resume_checkpoint {
            truncate_metrics_for_resume(&metrics_path, checkpoint.generation)?;
        }
        let write_header = !metrics_path.exists()
            || fs::metadata(&metrics_path)
                .map(|metadata| metadata.len() == 0)
                .unwrap_or(true);
        let mut metrics = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&metrics_path)
                .map_err(|error| format!("failed to open {}: {error}", metrics_path.display()))?,
        );
        if write_header {
            writeln!(metrics, "generation,wall_clock_seconds,best_training_fitness,population_average_fitness,champion_useful_speed,champion_completion_rate,validation_track,validation_score,validation_progress,validation_finish_reason,completed_count,collision_count,laser_eliminated_count,timeout_count,total_fixed_ticks,fixed_ticks_per_second,simulated_seconds")
                .map_err(|error| format!("failed to write metrics header: {error}"))?;
            metrics.flush().map_err(|error| error.to_string())?;
        }
        let log_path = prepared.run_directory.join("worker.log");
        let mut log = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&log_path)
                .map_err(|error| format!("failed to open {}: {error}", log_path.display()))?,
        );
        writeln!(
            log,
            "worker start name={} architecture={} seed={} target={} crossover={} mutation={} status={:?}",
            prepared.manifest.name,
            architecture_slug(&prepared.manifest.architecture),
            prepared.manifest.seed,
            prepared.manifest.target_generation,
            prepared
                .manifest
                .genetic_config
                .crossover_probability,
            prepared.manifest.genetic_config.mutation_probability,
            prepared.manifest.status
        )
        .map_err(|error| error.to_string())?;
        for message in &prepared.resume_messages {
            writeln!(log, "resume: {message}").map_err(|error| error.to_string())?;
        }
        log.flush().map_err(|error| error.to_string())?;
        Ok(Self {
            run_directory: prepared.run_directory.clone(),
            manifest: prepared.manifest.clone(),
            target_generation: prepared.manifest.target_generation,
            metrics,
            log,
            generation_started: Instant::now(),
            total_fixed_ticks: 0,
            generation_start_tick: 0,
            completed: false,
        })
    }

    pub fn record_fixed_tick(&mut self) {
        self.total_fixed_ticks = self.total_fixed_ticks.saturating_add(1);
    }

    pub fn is_completed(&self) -> bool {
        self.completed
    }

    pub fn on_generation_boundary(
        &mut self,
        training: &TrainingState,
        report: &CompletedChampion,
        simulation: &SimulationConfig,
        library: &TrackLibrary,
        checkpoints: &mut CheckpointStore,
        best_was_saved: bool,
    ) -> Result<bool, String> {
        let duration = self.generation_started.elapsed();
        let ticks = self
            .total_fixed_ticks
            .saturating_sub(self.generation_start_tick);
        let seconds = duration.as_secs_f64();
        let counts = report.track_stats.iter().fold(
            crate::simulation::FinishReasonCounts::default(),
            |mut total, track| {
                total.completed += track.finish_counts.completed;
                total.collision += track.finish_counts.collision;
                total.laser_eliminated += track.finish_counts.laser_eliminated;
                total.timeout += track.finish_counts.timeout;
                total
            },
        );
        writeln!(
            self.metrics,
            "{},{:.6},{:.9},{:.9},{:.6},{:.6},{},{:.9},{:.9},{},{},{},{},{},{},{:.3},{:.6}",
            report.generation,
            seconds,
            report.training.training_fitness,
            report.training.population_average_fitness,
            report.training.average_useful_progress_speed,
            report.training.completion_rate,
            csv_field(&report.validation.track_id),
            report.validation.score,
            report.validation.normalized_progress,
            report.validation.finish_reason.label(),
            counts.completed,
            counts.collision,
            counts.laser_eliminated,
            counts.timeout,
            ticks,
            if seconds > 0.0 {
                ticks as f64 / seconds
            } else {
                0.0
            },
            ticks as f64 / 60.0,
        )
        .map_err(|error| format!("failed to append metrics: {error}"))?;
        self.metrics
            .flush()
            .map_err(|error| format!("failed to flush metrics: {error}"))?;

        let training_path = save_training_checkpoint(
            &self.run_directory.join("training_checkpoint"),
            &training.training_checkpoint(),
            library,
        )?;
        let reached_target = training.generation() >= self.target_generation;
        let best_path = if reached_target && !best_was_saved {
            Some(
                checkpoints
                    .save_current_champion(training, simulation)
                    .map_err(|error| error.to_string())?,
            )
        } else {
            None
        };
        writeln!(
            self.log,
            "generation={} next_generation={} duration={:.3}s ticks={} training_checkpoint={} best={}",
            report.generation,
            training.generation(),
            seconds,
            ticks,
            training_path.display(),
            best_path
                .as_deref()
                .map_or_else(|| if best_was_saved { "autosaved".into() } else { "not-due".into() }, |path| path.display().to_string())
        )
        .map_err(|error| format!("failed to append worker log: {error}"))?;
        self.log
            .flush()
            .map_err(|error| format!("failed to flush worker log: {error}"))?;
        self.generation_started = Instant::now();
        self.generation_start_tick = self.total_fixed_ticks;
        if reached_target {
            self.manifest.status = RunStatus::Completed;
            self.manifest.completed_generation = Some(training.generation());
            write_run_manifest(&self.run_directory.join("run.ron"), &self.manifest)?;
            self.metrics.flush().map_err(|error| error.to_string())?;
            self.log.flush().map_err(|error| error.to_string())?;
            self.completed = true;
        }
        Ok(reached_target)
    }
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.into()
    }
}

fn truncate_metrics_for_resume(path: &Path, prepared_generation: usize) -> Result<(), String> {
    let contents = match fs::read_to_string(path) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(format!("failed to read {}: {error}", path.display())),
    };
    let mut lines = contents.lines();
    let Some(header) = lines.next() else {
        return Ok(());
    };
    let mut retained = vec![header.to_string()];
    for line in lines {
        let generation = line
            .split(',')
            .next()
            .and_then(|value| value.parse::<usize>().ok());
        if generation.is_some_and(|generation| generation < prepared_generation) {
            retained.push(line.to_string());
        }
    }
    let mut rewritten = retained.join("\n");
    rewritten.push('\n');
    fs::write(path, rewritten)
        .map_err(|error| format!("failed to reconcile {} for resume: {error}", path.display()))
}

pub fn save_training_checkpoint(
    directory: &Path,
    checkpoint: &TrainingCheckpoint,
    library: &TrackLibrary,
) -> Result<PathBuf, String> {
    checkpoint.validate(library)?;
    fs::create_dir_all(directory)
        .map_err(|error| format!("failed to create {}: {error}", directory.display()))?;
    let bytes = bincode::serde::encode_to_vec(checkpoint, bincode::config::standard())
        .map_err(|error| format!("failed to encode training checkpoint: {error}"))?;
    let stem = format!("generation_{:06}", checkpoint.generation);
    let temporary = directory.join(format!("{stem}.tmp"));
    let final_path = directory.join(format!("{stem}.bin"));
    if final_path.exists() {
        let existing = decode_training_checkpoint(&final_path)?;
        if existing.generation == checkpoint.generation {
            return Ok(final_path);
        }
        return Err(format!(
            "refusing to replace incompatible snapshot {}",
            final_path.display()
        ));
    }
    {
        let mut file = File::create(&temporary)
            .map_err(|error| format!("failed to create {}: {error}", temporary.display()))?;
        file.write_all(&bytes)
            .map_err(|error| format!("failed to write {}: {error}", temporary.display()))?;
        file.flush()
            .and_then(|()| file.sync_all())
            .map_err(|error| format!("failed to finalize {}: {error}", temporary.display()))?;
    }
    let decoded = decode_training_checkpoint(&temporary)?;
    if decoded.generation != checkpoint.generation {
        return Err("temporary checkpoint validation changed generation".into());
    }
    fs::rename(&temporary, &final_path).map_err(|error| {
        format!(
            "failed to atomically rename {} to {}: {error}",
            temporary.display(),
            final_path.display()
        )
    })?;
    retain_latest_checkpoints(directory, TRAINING_CHECKPOINT_KEEP)?;
    Ok(final_path)
}

pub fn decode_training_checkpoint(path: &Path) -> Result<TrainingCheckpoint, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let (checkpoint, consumed): (TrainingCheckpoint, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())
            .map_err(|error| format!("failed to decode {}: {error}", path.display()))?;
    if consumed != bytes.len() {
        return Err(format!("{} contains trailing bytes", path.display()));
    }
    Ok(checkpoint)
}

fn checkpoint_paths_newest_first(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(format!("failed to scan {}: {error}", directory.display())),
    };
    let mut paths = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "bin"))
        .collect::<Vec<_>>();
    paths.sort_by(|left, right| right.file_name().cmp(&left.file_name()));
    Ok(paths)
}

fn retain_latest_checkpoints(directory: &Path, keep: usize) -> Result<(), String> {
    for path in checkpoint_paths_newest_first(directory)?
        .into_iter()
        .skip(keep)
    {
        fs::remove_file(&path).map_err(|error| {
            format!(
                "failed to remove old checkpoint {}: {error}",
                path.display()
            )
        })?;
    }
    Ok(())
}

pub fn load_latest_training_checkpoint(
    directory: &Path,
    manifest: &RunManifest,
    library: &TrackLibrary,
) -> Result<(TrainingCheckpoint, Vec<String>), String> {
    let paths = checkpoint_paths_newest_first(directory)?;
    if paths.is_empty() {
        return Err(format!(
            "no training checkpoints found in {}",
            directory.display()
        ));
    }
    let mut errors = Vec::new();
    for path in paths {
        let loaded = decode_training_checkpoint(&path).and_then(|checkpoint| {
            checkpoint.validate(library)?;
            validate_checkpoint_manifest(&checkpoint, manifest)?;
            Ok(checkpoint)
        });
        match loaded {
            Ok(checkpoint) => {
                errors.push(format!(
                    "selected {} at generation {}",
                    path.display(),
                    checkpoint.generation
                ));
                return Ok((checkpoint, errors));
            }
            Err(error) => errors.push(format!("rejected {}: {error}", path.display())),
        }
    }
    Err(format!(
        "no valid training checkpoint could be restored:\n{}",
        errors.join("\n")
    ))
}

fn validate_checkpoint_manifest(
    checkpoint: &TrainingCheckpoint,
    manifest: &RunManifest,
) -> Result<(), String> {
    if checkpoint.architecture != manifest.architecture
        || checkpoint.individuals.len() != manifest.population_size
        || checkpoint.genetic_config != manifest.genetic_config
        || checkpoint.evaluation_config != manifest.evaluation_config
        || checkpoint.seed != manifest.seed
        || checkpoint.rng_id != manifest.rng
    {
        Err("training checkpoint is incompatible with run.ron".into())
    } else {
        Ok(())
    }
}

pub fn load_run_manifest(path: &Path) -> Result<RunManifest, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut manifest: RunManifest = ron::from_str(&contents)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
    if manifest.name.is_empty() {
        manifest.name = path
            .parent()
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
    }
    Ok(manifest)
}

pub fn write_run_manifest(path: &Path, manifest: &RunManifest) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let contents = ron::ser::to_string_pretty(manifest, PrettyConfig::default())
        .map_err(|error| format!("failed to serialize run manifest: {error}"))?;
    fs::write(path, contents)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

pub fn mark_run_failed(run_directory: &Path) {
    let path = run_directory.join("run.ron");
    if let Ok(mut manifest) = load_run_manifest(&path) {
        manifest.status = RunStatus::Failed;
        let _ = write_run_manifest(&path, &manifest);
    }
}

fn unix_timestamp() -> Result<u64, String> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .map_err(|error| format!("system clock error: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{
        EpisodeResult, FinishReason, TrackAdvance, TrainingPhase, test_saved_network,
    };
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha12Rng;

    fn temp_directory(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "neuroevolution-experiment-{label}-{}-{}",
            std::process::id(),
            unix_timestamp().unwrap()
        ))
    }

    fn test_run(name: &str, crossover: f32, mutation: f32) -> ResolvedExperimentRun {
        ResolvedExperimentRun {
            name: name.into(),
            architecture: vec![6, 8, 2],
            seed: 12345,
            population_size: 3,
            target_generation: 1,
            crossover_probability: crossover,
            mutation_probability: mutation,
        }
    }

    fn test_manifest(
        name: &str,
        seed: u64,
        population_size: usize,
        target_generation: usize,
        crossover: f32,
        mutation: f32,
        library: &TrackLibrary,
    ) -> RunManifest {
        let mut run = test_run(name, crossover, mutation);
        run.seed = seed;
        run.population_size = population_size;
        run.target_generation = target_generation;
        RunManifest::new(
            run.name.clone(),
            run.architecture.clone(),
            run.target_generation,
            run.genetic_config().unwrap(),
            library,
        )
        .unwrap()
    }

    #[test]
    fn slugs_and_directories_are_architecture_specific() {
        assert_eq!(architecture_slug(&[6, 8, 2]), "6_8_2");
        assert_eq!(architecture_slug(&[6, 8, 8, 2]), "6_8_8_2");
        assert_ne!(
            Path::new("results").join(architecture_slug(&[6, 8, 2])),
            Path::new("results").join(architecture_slug(&[6, 16, 2]))
        );
    }

    #[test]
    fn named_runs_allow_the_same_architecture_and_resolve_distinct_directories() {
        let config = ExperimentConfig {
            target_generation: 100,
            max_parallel_runs: 2,
            resume_existing: true,
            seed: 12345,
            population_size: 500,
            results_root: "results".into(),
            worker_threads: Some(1),
            architectures: Vec::new(),
            runs: vec![
                ExperimentRunConfig {
                    name: "baseline".into(),
                    architecture: vec![6, 8, 2],
                    crossover_probability: 0.8,
                    mutation_probability: 0.1,
                    seed: None,
                },
                ExperimentRunConfig {
                    name: "mutation_high".into(),
                    architecture: vec![6, 8, 2],
                    crossover_probability: 0.8,
                    mutation_probability: 0.2,
                    seed: None,
                },
            ],
        };
        config.validate().unwrap();
        let runs = config.resolved_runs().unwrap();
        assert_eq!(runs[0].seed, runs[1].seed);
        assert_eq!(runs[0].architecture, runs[1].architecture);
        assert_ne!(
            runs[0].directory(&config.results_root),
            runs[1].directory(&config.results_root)
        );
    }

    #[test]
    fn duplicate_run_names_are_rejected() {
        let mut config = ExperimentConfig {
            target_generation: 1,
            max_parallel_runs: 1,
            resume_existing: false,
            seed: 1,
            population_size: 2,
            results_root: "results".into(),
            worker_threads: None,
            architectures: Vec::new(),
            runs: vec![
                ExperimentRunConfig {
                    name: "same".into(),
                    architecture: vec![6, 3, 2],
                    crossover_probability: 0.8,
                    mutation_probability: 0.1,
                    seed: None,
                },
                ExperimentRunConfig {
                    name: "same".into(),
                    architecture: vec![6, 4, 2],
                    crossover_probability: 0.8,
                    mutation_probability: 0.1,
                    seed: None,
                },
            ],
        };
        assert!(config.validate().is_err());
        config.runs[1].name = "different".into();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn experiment_probabilities_outside_unit_interval_are_rejected_centrally() {
        for (crossover, mutation) in [(-0.1, 0.1), (1.1, 0.1), (0.8, -0.1), (0.8, 1.1)] {
            let run = test_run("invalid", crossover, mutation);
            assert!(run.genetic_config().is_err());
        }
    }

    #[test]
    fn comparison_ron_declares_five_named_runs_with_one_shared_seed() {
        let contents = include_str!("../../../experiments/ag_parameter_comparison.ron");
        let config: ExperimentConfig = ron::from_str(contents).unwrap();
        config.validate().unwrap();
        let runs = config.resolved_runs().unwrap();
        assert_eq!(runs.len(), 5);
        assert!(runs.iter().all(|run| run.seed == 12345));
        assert!(runs.iter().all(|run| run.architecture == vec![6, 8, 2]));
    }

    #[test]
    fn legacy_architecture_sweep_format_still_loads_and_derives_seeds() {
        let config: ExperimentConfig = ron::from_str(
            r#"(
                target_generation: 10,
                max_parallel_runs: 2,
                base_seed: 12345,
                population_size: 4,
                architectures: [[6, 3, 2], [6, 4, 2]],
            )"#,
        )
        .unwrap();
        config.validate().unwrap();
        let runs = config.resolved_runs().unwrap();
        assert_eq!(runs[0].name, "6_3_2");
        assert_eq!(runs[1].name, "6_4_2");
        assert_ne!(runs[0].seed, runs[1].seed);
        assert_eq!(runs[0].crossover_probability, 0.8);
        assert_eq!(runs[0].mutation_probability, 0.1);
    }

    #[test]
    fn configured_probabilities_reach_manifest_and_training_genetic_config() {
        let library = TrackLibrary::load_default().unwrap();
        let run = test_run("rates", 0.6, 0.2);
        let genetic = run.genetic_config().unwrap();
        let manifest = RunManifest::new(
            run.name.clone(),
            run.architecture.clone(),
            run.target_generation,
            genetic.clone(),
            &library,
        )
        .unwrap();
        assert_eq!(manifest.name, "rates");
        assert_eq!(manifest.genetic_config.crossover_probability, 0.6);
        assert_eq!(manifest.genetic_config.mutation_probability, 0.2);

        let state = TrainingState::with_genetic_config(
            &library,
            EvaluationConfig::default(),
            racing_architecture(run.architecture).unwrap(),
            genetic,
        )
        .unwrap();
        let saved = state.training_checkpoint().genetic_config;
        assert_eq!(saved.crossover_probability, 0.6);
        assert_eq!(saved.mutation_probability, 0.2);
    }

    #[test]
    fn batch_slot_guard_never_exceeds_configured_parallelism() {
        assert!(can_start_worker(0, 3));
        assert!(can_start_worker(2, 3));
        assert!(!can_start_worker(3, 3));
        assert!(!can_start_worker(4, 3));
    }

    #[test]
    fn racing_architecture_validates_contract_and_parameter_count() {
        assert_eq!(
            racing_architecture(vec![6, 8, 2])
                .unwrap()
                .parameter_count(),
            74
        );
        assert_eq!(
            racing_architecture(vec![6, 16, 2])
                .unwrap()
                .parameter_count(),
            146
        );
        assert!(racing_architecture(vec![5, 8, 2]).is_err());
        assert!(racing_architecture(vec![6, 8, 3]).is_err());
        assert!(racing_architecture(vec![6, 2]).is_err());
        assert!(racing_architecture(vec![6, 0, 2]).is_err());
    }

    #[test]
    fn rng_round_trip_preserves_next_values() {
        let mut rng = ChaCha12Rng::seed_from_u64(123);
        let _: u64 = rng.random();
        let encoded = bincode::serde::encode_to_vec(&rng, bincode::config::standard()).unwrap();
        let (mut restored, _): (ChaCha12Rng, usize) =
            bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();
        assert_eq!(rng.random::<u64>(), restored.random::<u64>());
        assert_eq!(rng.random::<u64>(), restored.random::<u64>());
    }

    #[test]
    fn checkpoint_round_trip_retention_and_corrupt_fallback_work() {
        let directory = temp_directory("checkpoint");
        let checkpoint_directory = directory.join("training_checkpoint");
        let library = TrackLibrary::load_default().unwrap();
        let state = TrainingState::with_architecture(
            3,
            &library,
            EvaluationConfig::default(),
            vec![6, 8, 2],
            99,
        )
        .unwrap();
        let mut snapshots = Vec::new();
        for generation in 1..=3 {
            let mut snapshot = state.training_checkpoint();
            snapshot.generation = generation;
            save_training_checkpoint(&checkpoint_directory, &snapshot, &library).unwrap();
            snapshots.push(snapshot);
        }
        let paths = checkpoint_paths_newest_first(&checkpoint_directory).unwrap();
        assert_eq!(paths.len(), 2);
        assert!(paths[0].ends_with("generation_000003.bin"));
        assert!(paths[1].ends_with("generation_000002.bin"));
        let manifest = test_manifest("6_8_2", 99, 3, 10, 0.8, 0.1, &library);
        let (newest, _) =
            load_latest_training_checkpoint(&checkpoint_directory, &manifest, &library).unwrap();
        assert_eq!(newest.generation, 3);
        fs::write(&paths[0], b"corrupt").unwrap();
        let (loaded, messages) =
            load_latest_training_checkpoint(&checkpoint_directory, &manifest, &library).unwrap();
        assert_eq!(loaded.generation, 2);
        assert!(messages.iter().any(|message| message.contains("rejected")));
        let restored = TrainingState::from_training_checkpoint(loaded, &library).unwrap();
        assert_eq!(restored.generation(), 2);
        assert_eq!(restored.population().individuals().len(), 3);
        assert_eq!(
            restored
                .population()
                .individuals()
                .iter()
                .map(|individual| individual.genome().genes().to_vec())
                .collect::<Vec<_>>(),
            snapshots[1]
                .individuals
                .iter()
                .map(|individual| individual.genome.clone())
                .collect::<Vec<_>>()
        );
        fs::remove_dir_all(directory).unwrap();
    }

    fn deterministic_result(index: usize, generation: usize) -> EpisodeResult {
        let score = 0.1 + index as f32 * 0.01 + generation as f32 * 0.001;
        EpisodeResult {
            score,
            normalized_progress: score,
            useful_progress_speed: 10.0 + index as f32,
            normalized_progress_speed: 0.1,
            elapsed: 1.0,
            finish_reason: FinishReason::EliminatedByLaser,
        }
    }

    fn complete_one_generation(state: &mut TrainingState) {
        loop {
            match state.phase() {
                TrainingPhase::TrainingTrack { .. } => {
                    let results = (0..state.population().len())
                        .map(|index| deterministic_result(index, state.generation()))
                        .collect::<Vec<_>>();
                    let _ = state.record_training_results(&results).unwrap();
                }
                TrainingPhase::Validation { .. } => {
                    assert_eq!(
                        state
                            .record_validation_result(deterministic_result(0, state.generation()))
                            .unwrap(),
                        TrackAdvance::ReadyToEvolve
                    );
                }
                TrainingPhase::Evolving => {
                    state.evolve_generation().unwrap();
                    return;
                }
            }
        }
    }

    #[test]
    fn save_restore_then_continue_matches_continuous_evolution() {
        let library = TrackLibrary::load_default().unwrap();
        let mut continuous = TrainingState::with_architecture(
            6,
            &library,
            EvaluationConfig::default(),
            vec![6, 5, 2],
            2026,
        )
        .unwrap();
        let mut resumed = TrainingState::with_architecture(
            6,
            &library,
            EvaluationConfig::default(),
            vec![6, 5, 2],
            2026,
        )
        .unwrap();
        complete_one_generation(&mut continuous);
        complete_one_generation(&mut resumed);
        let selected_at_checkpoint = resumed.selected_training_tracks().to_vec();
        resumed = TrainingState::from_training_checkpoint(resumed.training_checkpoint(), &library)
            .unwrap();
        assert_eq!(resumed.selected_training_tracks(), selected_at_checkpoint);
        for _ in 1..4 {
            complete_one_generation(&mut continuous);
            complete_one_generation(&mut resumed);
            assert_eq!(
                continuous.selected_training_tracks(),
                resumed.selected_training_tracks()
            );
        }
        assert_eq!(continuous.generation(), resumed.generation());
        assert_eq!(
            continuous
                .population()
                .individuals()
                .iter()
                .map(|individual| (individual.genome().genes().to_vec(), individual.fitness()))
                .collect::<Vec<_>>(),
            resumed
                .population()
                .individuals()
                .iter()
                .map(|individual| (individual.genome().genes().to_vec(), individual.fitness()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn stray_temporary_file_does_not_destroy_previous_snapshot() {
        let directory = temp_directory("temporary");
        let library = TrackLibrary::load_default().unwrap();
        let state = TrainingState::with_architecture(
            2,
            &library,
            EvaluationConfig::default(),
            vec![6, 3, 2],
            7,
        )
        .unwrap();
        let checkpoint_directory = directory.join("training_checkpoint");
        let mut checkpoint = state.training_checkpoint();
        checkpoint.generation = 1;
        let stable =
            save_training_checkpoint(&checkpoint_directory, &checkpoint, &library).unwrap();
        fs::write(
            checkpoint_directory.join("generation_000002.tmp"),
            b"partial",
        )
        .unwrap();
        assert_eq!(decode_training_checkpoint(&stable).unwrap().generation, 1);
        assert_eq!(
            checkpoint_paths_newest_first(&checkpoint_directory).unwrap(),
            vec![stable]
        );
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn incompatible_manifest_is_rejected_but_target_may_increase() {
        let library = TrackLibrary::load_default().unwrap();
        let original = test_manifest("run", 1, 3, 500, 0.8, 0.1, &library);
        let larger_target = test_manifest("run", 1, 3, 800, 0.8, 0.1, &library);
        assert!(original.validate_compatible(&larger_target).is_ok());
        let wrong_seed = test_manifest("run", 2, 3, 800, 0.8, 0.1, &library);
        assert!(original.validate_compatible(&wrong_seed).is_err());
        let wrong_crossover = test_manifest("run", 1, 3, 800, 0.6, 0.1, &library);
        assert!(original.validate_compatible(&wrong_crossover).is_err());
        let wrong_mutation = test_manifest("run", 1, 3, 800, 0.8, 0.2, &library);
        assert!(original.validate_compatible(&wrong_mutation).is_err());
    }

    #[test]
    fn incompatible_resume_does_not_overwrite_existing_manifest() {
        let directory = temp_directory("manifest");
        let library = TrackLibrary::load_default().unwrap();
        let architecture = vec![6, 8, 2];
        let run_directory = directory.join(architecture_slug(&architecture));
        let manifest_path = run_directory.join("run.ron");
        let original = test_manifest("6_8_2", 11, 3, 5, 0.8, 0.1, &library);
        write_run_manifest(&manifest_path, &original).unwrap();
        let before = fs::read(&manifest_path).unwrap();
        let options = WorkerOptions {
            name: "6_8_2".into(),
            architecture,
            target_generation: 10,
            seed: 12,
            results_root: directory.clone(),
            population_size: 3,
            crossover_probability: 0.8,
            mutation_probability: 0.1,
            worker_threads: 1,
            resume: true,
        };
        assert!(prepare_worker(&options, &library).is_err());
        assert_eq!(fs::read(&manifest_path).unwrap(), before);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn summary_uses_each_runs_real_final_generation_and_failure_has_no_fitness() {
        let directory = temp_directory("summary");
        let library = TrackLibrary::load_default().unwrap();
        let runs = vec![
            test_run("first", 0.8, 0.1),
            test_run("second", 0.6, 0.1),
            test_run("failed", 0.8, 0.2),
        ];
        for (run, fitness) in runs[..2].iter().zip([1.25_f32, 2.5_f32]) {
            let run_directory = run.directory(&directory);
            let mut manifest = test_manifest(
                &run.name,
                run.seed,
                run.population_size,
                run.target_generation,
                run.crossover_probability,
                run.mutation_probability,
                &library,
            );
            manifest.status = RunStatus::Completed;
            manifest.completed_generation = Some(1);
            write_run_manifest(&run_directory.join("run.ron"), &manifest).unwrap();
            let mut saved = test_saved_network(run.architecture.clone(), 0);
            saved.training_metadata.champion_training_fitness = fitness;
            let checkpoint_directory = run_directory.join("bests_by_gen");
            fs::create_dir_all(&checkpoint_directory).unwrap();
            fs::write(
                checkpoint_directory.join("generation_000000_test.ron"),
                ron::ser::to_string_pretty(&saved, PrettyConfig::default()).unwrap(),
            )
            .unwrap();
        }
        let failed = &runs[2];
        let mut manifest = test_manifest(
            &failed.name,
            failed.seed,
            failed.population_size,
            failed.target_generation,
            failed.crossover_probability,
            failed.mutation_probability,
            &library,
        );
        manifest.status = RunStatus::Failed;
        let failed_directory = failed.directory(&directory);
        write_run_manifest(&failed_directory.join("run.ron"), &manifest).unwrap();
        let checkpoint_directory = failed_directory.join("bests_by_gen");
        fs::create_dir_all(&checkpoint_directory).unwrap();
        let mut misleading = test_saved_network(failed.architecture.clone(), 0);
        misleading.training_metadata.champion_training_fitness = 999.0;
        fs::write(
            checkpoint_directory.join("misleading.ron"),
            ron::ser::to_string_pretty(&misleading, PrettyConfig::default()).unwrap(),
        )
        .unwrap();

        write_batch_summary(&directory, &runs).unwrap();
        let summary = fs::read_to_string(directory.join("summary.csv")).unwrap();
        assert!(summary.contains("1,first,12345,6-8-2,0.8,0.1,3,1,1.250000000,completed"));
        assert!(summary.contains("2,second,12345,6-8-2,0.6,0.1,3,1,2.500000000,completed"));
        assert!(summary.contains("3,failed,12345,6-8-2,0.8,0.2,3,1,,failed"));
        fs::remove_dir_all(directory).unwrap();
    }
}
