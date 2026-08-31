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
    TrackLibrary, TrainingCheckpoint, TrainingState, racing_architecture,
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
        architecture: Vec<usize>,
        seed: u64,
        population_size: usize,
        target_generation: usize,
        library: &TrackLibrary,
    ) -> Result<Self, String> {
        let architecture_model = racing_architecture(architecture.clone())?;
        let core_genetic_config = GeneticConfig {
            population_size,
            genome_length: architecture_model.parameter_count(),
            seed,
            ..GeneticConfig::default()
        };
        let genetic_config = SavedGeneticConfig::from_config(&core_genetic_config);
        let simulation_config = SimulationConfig {
            population_size,
            ..SimulationConfig::default()
        };
        Ok(Self {
            format_version: RUN_FORMAT_VERSION,
            architecture,
            parameter_count: architecture_model.parameter_count(),
            seed,
            population_size,
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
    pub base_seed: u64,
    #[serde(default = "default_population_size")]
    pub population_size: usize,
    #[serde(default = "default_results_root")]
    pub results_root: PathBuf,
    #[serde(default)]
    pub worker_threads: Option<usize>,
    pub architectures: Vec<Vec<usize>>,
}

fn default_population_size() -> usize {
    500
}

fn default_results_root() -> PathBuf {
    PathBuf::from("results")
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
        if self.architectures.is_empty() {
            return Err("architectures must not be empty".into());
        }
        let mut slugs = std::collections::BTreeSet::new();
        for architecture in &self.architectures {
            racing_architecture(architecture.clone())?;
            if !slugs.insert(architecture_slug(architecture)) {
                return Err(format!(
                    "duplicate architecture {}",
                    architecture_slug(architecture)
                ));
            }
        }
        Ok(())
    }
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
    slug: String,
    child: Child,
}

pub fn run_batch(config_path: &Path) -> Result<(), String> {
    let config = ExperimentConfig::load(config_path)?;
    let library = TrackLibrary::load_default().map_err(|error| error.to_string())?;
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
    for architecture in config.architectures.clone() {
        let slug = architecture_slug(&architecture);
        println!("[QUEUED ] {slug}");
        queued.push_back(architecture);
    }
    let mut running: Vec<RunningWorker> = Vec::new();
    let mut failures = 0usize;
    while !queued.is_empty() || !running.is_empty() {
        while running.len() < config.max_parallel_runs && !queued.is_empty() {
            let architecture = queued.pop_front().unwrap();
            let slug = architecture_slug(&architecture);
            let run_directory = config.results_root.join(&slug);
            let manifest_path = run_directory.join("run.ron");
            if manifest_path.exists() {
                if !config.resume_existing {
                    return Err(format!(
                        "{} already contains an experiment and resume_existing is false",
                        run_directory.display()
                    ));
                }
                let existing = load_run_manifest(&manifest_path)?;
                let requested = RunManifest::new(
                    architecture.clone(),
                    architecture_seed(config.base_seed, &architecture),
                    config.population_size,
                    config.target_generation,
                    &library,
                )?;
                existing.validate_compatible(&requested)?;
            }
            fs::create_dir_all(&run_directory).map_err(|error| {
                format!("failed to create {}: {error}", run_directory.display())
            })?;
            let log = OpenOptions::new()
                .create(true)
                .append(true)
                .open(run_directory.join("worker.log"))
                .map_err(|error| format!("failed to open worker log for {slug}: {error}"))?;
            let stderr = log
                .try_clone()
                .map_err(|error| format!("failed to clone worker log: {error}"))?;
            let mut command = Command::new(&executable);
            command
                .arg("worker")
                .arg("--architecture")
                .arg(
                    architecture
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                )
                .arg("--target-generation")
                .arg(config.target_generation.to_string())
                .arg("--seed")
                .arg(architecture_seed(config.base_seed, &architecture).to_string())
                .arg("--results-root")
                .arg(&config.results_root)
                .arg("--population-size")
                .arg(config.population_size.to_string())
                .arg("--worker-threads")
                .arg(worker_threads.to_string())
                .stdout(Stdio::from(log))
                .stderr(Stdio::from(stderr));
            if config.resume_existing && manifest_path.exists() {
                command.arg("--resume");
            }
            let child = command
                .spawn()
                .map_err(|error| format!("failed to start worker {slug}: {error}"))?;
            println!("[RUNNING] {slug:<18} pid={}", child.id());
            running.push(RunningWorker { slug, child });
        }

        let mut index = 0;
        while index < running.len() {
            match running[index].child.try_wait() {
                Ok(Some(status)) => {
                    let finished = running.swap_remove(index);
                    print_worker_exit(&finished.slug, status, &config.results_root);
                    if !status.success() {
                        failures += 1;
                    }
                }
                Ok(None) => index += 1,
                Err(error) => {
                    let failed = running.swap_remove(index);
                    eprintln!("[FAILED ] {} poll error={error}", failed.slug);
                    failures += 1;
                }
            }
        }
        if !running.is_empty() {
            thread::sleep(Duration::from_millis(200));
        }
    }
    if failures == 0 {
        Ok(())
    } else {
        Err(format!("{failures} worker(s) failed"))
    }
}

fn print_worker_exit(slug: &str, status: ExitStatus, results_root: &Path) {
    if status.success() {
        let generation = load_run_manifest(&results_root.join(slug).join("run.ron"))
            .ok()
            .and_then(|manifest| manifest.completed_generation)
            .map_or_else(|| "?".into(), |value| value.to_string());
        println!("[DONE   ] {slug:<18} generation={generation}");
    } else {
        let manifest_path = results_root.join(slug).join("run.ron");
        if let Ok(mut manifest) = load_run_manifest(&manifest_path) {
            manifest.status = RunStatus::Failed;
            let _ = write_run_manifest(&manifest_path, &manifest);
        }
        println!("[FAILED ] {slug:<18} exit={status}");
    }
}

#[derive(Clone, Debug)]
pub struct WorkerOptions {
    pub architecture: Vec<usize>,
    pub target_generation: usize,
    pub seed: u64,
    pub results_root: PathBuf,
    pub population_size: usize,
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
    racing_architecture(options.architecture.clone())?;
    let slug = architecture_slug(&options.architecture);
    let run_directory = options.results_root.join(&slug);
    let manifest_path = run_directory.join("run.ron");
    let requested = RunManifest::new(
        options.architecture.clone(),
        options.seed,
        options.population_size,
        options.target_generation,
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
            "worker start architecture={} seed={} target={} status={:?}",
            architecture_slug(&prepared.manifest.architecture),
            prepared.manifest.seed,
            prepared.manifest.target_generation,
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
    ron::from_str(&contents).map_err(|error| format!("failed to parse {}: {error}", path.display()))
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
    use crate::simulation::{EpisodeResult, FinishReason, TrackAdvance, TrainingPhase};
    use rand::{RngExt, SeedableRng};
    use rand_chacha::ChaCha12Rng;

    fn temp_directory(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "neuroevolution-experiment-{label}-{}-{}",
            std::process::id(),
            unix_timestamp().unwrap()
        ))
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
        let manifest = RunManifest::new(vec![6, 8, 2], 99, 3, 10, &library).unwrap();
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
        let original = RunManifest::new(vec![6, 8, 2], 1, 3, 500, &library).unwrap();
        let larger_target = RunManifest::new(vec![6, 8, 2], 1, 3, 800, &library).unwrap();
        assert!(original.validate_compatible(&larger_target).is_ok());
        let wrong_seed = RunManifest::new(vec![6, 8, 2], 2, 3, 800, &library).unwrap();
        assert!(original.validate_compatible(&wrong_seed).is_err());
    }

    #[test]
    fn incompatible_resume_does_not_overwrite_existing_manifest() {
        let directory = temp_directory("manifest");
        let library = TrackLibrary::load_default().unwrap();
        let architecture = vec![6, 8, 2];
        let run_directory = directory.join(architecture_slug(&architecture));
        let manifest_path = run_directory.join("run.ron");
        let original = RunManifest::new(architecture.clone(), 11, 3, 5, &library).unwrap();
        write_run_manifest(&manifest_path, &original).unwrap();
        let before = fs::read(&manifest_path).unwrap();
        let options = WorkerOptions {
            architecture,
            target_generation: 10,
            seed: 12,
            results_root: directory.clone(),
            population_size: 3,
            worker_threads: 1,
            resume: true,
        };
        assert!(prepare_worker(&options, &library).is_err());
        assert_eq!(fs::read(&manifest_path).unwrap(), before);
        fs::remove_dir_all(directory).unwrap();
    }
}
