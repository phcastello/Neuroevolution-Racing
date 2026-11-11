//! Preparation and execution of reproducible research runs. Preparation never opens a device.
use clap::{Parser, Subcommand, ValueEnum};
use racing_rl::{
    config::{Config, CurriculumConfig, CurriculumKind, DeviceChoice, SensorLayout},
    persistence,
};
use serde::{Deserialize, Serialize};
use std::{
    ffi::OsString,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum BackendChoice {
    Cpu,
    Gpu,
}
impl BackendChoice {
    fn device(self) -> DeviceChoice {
        match self {
            Self::Cpu => DeviceChoice::Cpu,
            Self::Gpu => DeviceChoice::Gpu { adapter: 0 },
        }
    }
}
#[derive(Subcommand)]
pub enum ResearchCommand {
    /// Prepare full PPO training benchmark from historical E reward/PPO; execution is explicit.
    Benchmark {
        #[arg(long, value_enum, default_value = "cpu")]
        backend: BackendChoice,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 100)]
        rollouts: usize,
        #[arg(long)]
        execute: bool,
        #[arg(long, conflicts_with = "execute")]
        dry_run: bool,
    },
    /// Generate the factorial matrix or a progressive stage; execution is explicit.
    Matrix {
        #[arg(long, default_value = "experiments/factorial.ron")]
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        execute: bool,
        #[arg(long, conflicts_with = "execute")]
        dry_run: bool,
        /// Comma separated cell IDs. Essential original/no-anticipation baseline is retained.
        #[arg(long, value_delimiter = ',')]
        include: Vec<String>,
        /// Screening uses screening_rollouts; full uses rollouts. Finalists are manually selected.
        #[arg(long)]
        screening: bool,
        /// Continue selected cells from a prepared screening directory, restoring Adam/RNGs.
        #[arg(long)]
        continue_from: Option<PathBuf>,
    },
    /// Explicitly execute a previously prepared immutable plan.
    Execute {
        #[arg(long)]
        plan: PathBuf,
    },
    /// Consolidate recorded runs (possibly collected on different dates), never rerun training.
    Compare {
        #[arg(required = true)]
        runs: Vec<PathBuf>,
        #[arg(long)]
        output: PathBuf,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MatrixConfig {
    pub base: PathBuf,
    pub rollouts: usize,
    pub screening_rollouts: usize,
    pub seeds: Vec<u64>,
    pub repetitions: usize,
    pub max_parallel: usize,
    pub backend: String,
    pub evaluate_every: usize,
    /// None follows evaluate_every, or 100 when evaluation is disabled.
    pub checkpoint_every: Option<usize>,
    pub save_first: usize,
    pub save_last: usize,
    pub save_rollouts: bool,
}
impl Default for MatrixConfig {
    fn default() -> Self {
        Self {
            base: "configs/rl/ppo_e_continue_8h.ron".into(),
            rollouts: 500,
            screening_rollouts: 100,
            seeds: vec![11],
            repetitions: 1,
            max_parallel: 1,
            backend: "cpu".into(),
            evaluate_every: 100,
            checkpoint_every: None,
            save_first: 0,
            save_last: 0,
            save_rollouts: false,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlannedRun {
    pub cell: String,
    pub seed: u64,
    pub config: PathBuf,
    pub receipt: PathBuf,
    pub resume: Option<PathBuf>,
    pub evaluate_every: usize,
    #[serde(default)]
    pub final_evaluation: bool,
    #[serde(default)]
    pub checkpoint_every: Option<usize>,
    #[serde(default)]
    pub save_first: usize,
    #[serde(default)]
    pub save_last: usize,
    #[serde(default)]
    pub save_rollouts: bool,
}
impl PlannedRun {
    fn training_args(&self) -> Vec<OsString> {
        let every = self.checkpoint_every.unwrap_or(if self.evaluate_every > 0 {
            self.evaluate_every
        } else {
            100
        });
        let mut args = vec![
            "rl".into(),
            "train".into(),
            "--config".into(),
            self.config.as_os_str().to_owned(),
            "--save-every".into(),
            every.to_string().into(),
            "--save-first".into(),
            self.save_first.to_string().into(),
            "--save-last".into(),
            self.save_last.to_string().into(),
            "--run-path-file".into(),
            self.receipt.as_os_str().to_owned(),
        ];
        if !self.save_rollouts {
            args.push("--no-save-rollouts".into());
        }
        if let Some(path) = &self.resume {
            args.extend(["--resume".into(), path.as_os_str().to_owned()]);
        }
        if self.evaluate_every > 0 {
            args.extend([
                "--evaluate-every".into(),
                self.evaluate_every.to_string().into(),
            ]);
        }
        args
    }
    fn validate_persistence(&self, total_rollouts: usize) -> Result<(), String> {
        // Parse and validate the exact training arguments, sharing the existing
        // persistence rules, including final checkpoints and evaluation alignment.
        let cli = crate::Cli::try_parse_from(
            std::iter::once(OsString::from("neuroevolution-racing")).chain(self.training_args()),
        )
        .map_err(|e| e.to_string())?;
        let Some(crate::CommandLine::Rl {
            command: crate::rl::RlCommand::Train { persistence, .. },
        }) = cli.command
        else {
            return Err("invalid planned training command".into());
        };
        persistence
            .validate_evaluation(total_rollouts)
            .map_err(|e| e.to_string())
    }
}
#[derive(Serialize, Deserialize)]
struct Plan {
    runs: Vec<PlannedRun>,
    max_parallel: usize,
    phase: String,
}
fn cells() -> Vec<(String, SensorLayout, bool, Option<CurriculumKind>)> {
    let mut out = vec![];
    for (sensor, layout) in [
        ("original", SensorLayout::Original),
        ("front", SensorLayout::FrontDense),
    ] {
        for anticipation in [false, true] {
            for (name, kind) in [
                ("original", None),
                ("random", Some(CurriculumKind::Random)),
                ("difficulty", Some(CurriculumKind::Difficulty)),
                ("aco", Some(CurriculumKind::Aco)),
            ] {
                out.push((
                    format!(
                        "{sensor}-{}-{name}",
                        if anticipation { "future" } else { "rays" }
                    ),
                    layout,
                    anticipation,
                    kind,
                ));
            }
        }
    }
    out
}
fn project_path(path: &Path) -> PathBuf {
    if path.is_absolute() || path.is_file() {
        path.to_owned()
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(path)
    }
}
fn base() -> Result<Config, String> {
    Config::load(&project_path(Path::new("configs/rl/ppo_e_continue_8h.ron")))
        .map_err(|e| e.to_string())
}
fn new_output(path: &Path) -> Result<(), String> {
    fs::create_dir(path).map_err(|e| format!("new output {}: {e}", path.display()))
}
fn save<T: Serialize>(p: &Path, value: &T) -> Result<(), String> {
    persistence::save(p, value).map_err(|e| e.to_string())
}
pub fn run(command: ResearchCommand) -> Result<(), String> {
    match command {
        ResearchCommand::Benchmark {
            backend,
            output,
            rollouts,
            execute,
            dry_run,
        } => {
            let mut c = base()?;
            c.device = backend.device();
            c.total_rollouts = rollouts;
            c.curriculum = None;
            c.results_root = output.join("runs").to_string_lossy().into_owned();
            c = crate::rl::resolve_config(c).map_err(|e| e.to_string())?;
            c.validate().map_err(|e| e.to_string())?;
            if dry_run {
                println!(
                    "benchmark preparation: backend={:?} rollouts={rollouts} N={} T={} M={} E={} parameters={} output={} (no execution)",
                    c.device,
                    c.num_envs,
                    c.rollout_length,
                    c.minibatch_size,
                    c.update_epochs,
                    c.parameter_count(),
                    output.display()
                );
                return Ok(());
            }
            new_output(&output)?;
            let path = output.join("config.ron");
            save(&path, &c)?;
            let plan = Plan {
                runs: vec![PlannedRun {
                    cell: "hardware-fixed-E".into(),
                    seed: c.network_seed,
                    config: path,
                    receipt: output.join("run-path.txt"),
                    resume: None,
                    evaluate_every: 0,
                    final_evaluation: false,
                    checkpoint_every: None,
                    save_first: 0,
                    save_last: 0,
                    save_rollouts: false,
                }],
                max_parallel: 1,
                phase: "hardware benchmark; fresh policy; no periodic evaluation; no curriculum"
                    .into(),
            };
            save(&output.join("plan.ron"), &plan)?;
            println!(
                "Prepared {}; explicit --execute starts training",
                output.display()
            );
            if execute {
                execute_plan(&plan)?;
            }
            Ok(())
        }
        ResearchCommand::Matrix {
            config,
            output,
            execute,
            dry_run,
            include,
            screening,
            continue_from,
        } => {
            let m: MatrixConfig =
                persistence::load(&project_path(&config)).map_err(|e| e.to_string())?;
            if m.max_parallel == 0
                || m.repetitions == 0
                || m.seeds.is_empty()
                || m.rollouts == 0
                || m.rollouts > 500
                || m.screening_rollouts == 0
                || m.screening_rollouts > m.rollouts
                || !["cpu", "gpu"].contains(&m.backend.as_str())
            {
                return Err("invalid matrix budgets/seeds/backend".into());
            }
            if screening && continue_from.is_some() {
                return Err("continuation requires full phase".into());
            }
            let budget = if screening {
                m.screening_rollouts
            } else {
                m.rollouts
            };
            let policy = PlannedRun {
                cell: String::new(),
                seed: 0,
                config: "config.ron".into(),
                receipt: "run-path.txt".into(),
                resume: None,
                evaluate_every: m.evaluate_every,
                final_evaluation: true,
                checkpoint_every: m.checkpoint_every,
                save_first: m.save_first,
                save_last: m.save_last,
                save_rollouts: m.save_rollouts,
            };
            policy.validate_persistence(budget)?;
            validate_paired_seeds(&m.seeds, m.repetitions)?;
            let b = crate::rl::resolve_config(
                Config::load(&project_path(&m.base)).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())?;
            let all = cells();
            for id in &include {
                if !all.iter().any(|(x, ..)| x == id) {
                    return Err(format!("unknown cell {id}"));
                }
            }
            let chosen: Vec<_> = all
                .into_iter()
                .filter(|(id, ..)| {
                    include.is_empty() || include.contains(id) || id == "original-rays-original"
                })
                .collect();
            let total = chosen.len() * m.repetitions * m.seeds.len();
            if dry_run {
                println!(
                    "matrix: {total} runs, {} cells; paired seeds {:?}; parallel {}; rollouts {}; no training",
                    chosen.len(),
                    m.seeds,
                    m.max_parallel,
                    if screening {
                        m.screening_rollouts
                    } else {
                        m.rollouts
                    }
                );
                return Ok(());
            }
            new_output(&output)?;
            let mut plan = Plan {
                runs: vec![],
                max_parallel: m.max_parallel,
                phase: if screening {
                    "screening"
                } else {
                    "full/manual finalists"
                }
                .into(),
            };
            for (id, layout, future, kind) in chosen {
                for seed in &m.seeds {
                    for rep in 0..m.repetitions {
                        let paired = seed.wrapping_add((rep as u64).wrapping_mul(1_000_003));
                        let name = format!("{id}-s{paired}");
                        let dir = output.join(&name);
                        fs::create_dir(&dir).map_err(|e| e.to_string())?;
                        let mut c = b.clone();
                        c.device = if m.backend == "cpu" {
                            DeviceChoice::Cpu
                        } else {
                            DeviceChoice::Gpu { adapter: 0 }
                        };
                        c.total_rollouts = if screening {
                            m.screening_rollouts
                        } else {
                            m.rollouts
                        };
                        c.observation = c
                            .observation
                            .with_sensor_layout(layout)
                            .with_anticipation(future);
                        c.curriculum = kind.map(|kind| CurriculumConfig {
                            kind,
                            seed: paired.wrapping_add(660),
                            ..Default::default()
                        });
                        c.network_seed = paired;
                        c.environment_seed = paired.wrapping_add(11);
                        c.action_seed = paired.wrapping_add(22);
                        c.shuffle_seed = paired.wrapping_add(33);
                        c.evaluation_seed = 55;
                        c.results_root = dir.join("runs").to_string_lossy().into_owned();
                        c.validate().map_err(|e| e.to_string())?;
                        let resume = if let Some(previous) = &continue_from {
                            let receipt =
                                fs::read_to_string(previous.join(&name).join("run-path.txt"))
                                    .map_err(|e| {
                                        format!("missing screening receipt for {name}: {e}")
                                    })?;
                            let p = PathBuf::from(receipt.trim())
                                .join(format!("checkpoint-{:06}.ron", m.screening_rollouts));
                            if !p.is_file() {
                                return Err(format!(
                                    "missing screening checkpoint {}",
                                    p.display()
                                ));
                            }
                            Some(p)
                        } else {
                            None
                        };
                        let path = dir.join("config.ron");
                        save(&path, &c)?;
                        plan.runs.push(PlannedRun {
                            cell: id.clone(),
                            seed: paired,
                            config: path,
                            receipt: dir.join("run-path.txt"),
                            resume,
                            evaluate_every: m.evaluate_every,
                            final_evaluation: true,
                            checkpoint_every: m.checkpoint_every,
                            save_first: m.save_first,
                            save_last: m.save_last,
                            save_rollouts: m.save_rollouts,
                        });
                    }
                }
            }
            save(&output.join("plan.ron"), &plan)?;
            println!("Prepared {} runs at {}", plan.runs.len(), output.display());
            if execute {
                execute_plan(&plan)?;
            }
            Ok(())
        }
        ResearchCommand::Execute { plan } => {
            let p: Plan = persistence::load(&plan).map_err(|e| e.to_string())?;
            execute_plan(&p)
        }
        ResearchCommand::Compare { runs, output } => compare(&runs, &output),
    }
}
fn validate_paired_seeds(seeds: &[u64], repetitions: usize) -> Result<(), String> {
    let mut seen = std::collections::BTreeSet::new();
    for seed in seeds {
        for rep in 0..repetitions {
            let paired = seed.wrapping_add((rep as u64).wrapping_mul(1_000_003));
            if !seen.insert(paired) {
                return Err(format!(
                    "duplicate paired seed {paired}; seeds/repetitions would share an output directory"
                ));
            }
        }
    }
    Ok(())
}
fn execute_plan(plan: &Plan) -> Result<(), String> {
    if plan.max_parallel == 0 || plan.runs.is_empty() {
        return Err("empty plan or zero parallelism".into());
    }
    let mut receipts = std::collections::BTreeSet::new();
    for run in &plan.runs {
        let c = Config::load(&run.config).map_err(|e| e.to_string())?;
        run.validate_persistence(c.total_rollouts)?;
        if run.receipt.exists() || !receipts.insert(run.receipt.clone()) {
            return Err(format!(
                "existing/duplicate receipt: {}",
                run.receipt.display()
            ));
        }
        if run.resume.as_ref().is_some_and(|p| !p.is_file()) {
            return Err("missing resume checkpoint".into());
        }
    }
    // Bounded independent subprocesses; shared training command, no duplicate trainer.
    for batch in plan.runs.chunks(plan.max_parallel) {
        let mut children = vec![];
        for run in batch {
            let mut cmd = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
            cmd.args(run.training_args());
            let log = fs::File::create(run.config.parent().unwrap().join("launcher.log"))
                .map_err(|e| e.to_string())?;
            cmd.stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?))
                .stderr(Stdio::from(log));
            children.push(cmd.spawn().map_err(|e| e.to_string())?);
        }
        let mut failures = vec![];
        for mut child in children {
            let status = child.wait().map_err(|e| e.to_string())?;
            if !status.success() {
                failures.push(status.to_string());
            }
        }
        if !failures.is_empty() {
            return Err(format!("training failed: {}", failures.join(", ")));
        }
        for run in batch.iter().filter(|r| r.final_evaluation) {
            let c = Config::load(&run.config).map_err(|e| e.to_string())?;
            let dir = fs::read_to_string(&run.receipt).map_err(|e| e.to_string())?;
            let checkpoint =
                PathBuf::from(dir.trim()).join(format!("checkpoint-{:06}.ron", c.total_rollouts));
            let log_path = run.config.parent().unwrap().join("validation.log");
            let log = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&log_path)
                .map_err(|e| e.to_string())?;
            let status = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
                .args([
                    "rl",
                    "evaluate",
                    "--cpu",
                    "--tracks",
                    "validation",
                    "--checkpoint",
                ])
                .arg(checkpoint)
                .stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?))
                .stderr(Stdio::from(log))
                .status()
                .map_err(|e| e.to_string())?;
            if !status.success() {
                return Err(format!("held-out evaluation failed for {}", run.cell));
            }
            let evaluation =
                evaluation_directory(&fs::read_to_string(&log_path).map_err(|e| e.to_string())?)?;
            link_evaluation(Path::new(dir.trim()), &evaluation)?;
        }
    }
    Ok(())
}
// Keep evaluation artifacts independent while retaining the exact association to
// their training run. The existing evaluator prints this marker only on success.
fn evaluation_directory(log: &str) -> Result<PathBuf, String> {
    log.lines()
        .rev()
        .find_map(|line| {
            let rest = line.strip_prefix("evaluation=")?;
            let (path, _) = rest.rsplit_once(" episodes=")?;
            Some(PathBuf::from(path))
        })
        .ok_or_else(|| "successful evaluation did not report its artifact directory".into())
}
fn link_evaluation(training: &Path, evaluation: &Path) -> Result<(), String> {
    let training = fs::canonicalize(training).map_err(|e| e.to_string())?;
    let evaluation = fs::canonicalize(evaluation).map_err(|e| e.to_string())?;
    for name in ["evaluation_summary.json", "evaluation.csv"] {
        if !evaluation.join(name).is_file() {
            return Err(format!("missing evaluation artifact {name}"));
        }
    }
    // Sibling runs are collected together. A path relative to the training run
    // survives transferring that tree to another host, independent of cwd.
    let target = if training.parent() == evaluation.parent() {
        PathBuf::from("..").join(
            evaluation
                .file_name()
                .ok_or("invalid evaluation directory")?,
        )
    } else {
        evaluation
    };
    let mut receipt = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(training.join("evaluation-path.txt"))
        .map_err(|e| e.to_string())?;
    writeln!(receipt, "{}", target.display()).map_err(|e| e.to_string())?;
    receipt.sync_all().map_err(|e| e.to_string())
}
fn artifact_path(run: &Path, name: &str) -> Result<PathBuf, String> {
    let direct = run.join(name);
    if direct.is_file() || !["evaluation_summary.json", "evaluation.csv"].contains(&name) {
        return Ok(direct);
    }
    let receipt = run.join("evaluation-path.txt");
    if !receipt.exists() {
        return Ok(direct);
    }
    let linked = fs::read_to_string(&receipt).map_err(|e| e.to_string())?;
    let directory = PathBuf::from(linked.trim());
    let directory = if directory.is_absolute() {
        directory
    } else {
        run.join(directory)
    };
    let artifact = directory.join(name);
    if !artifact.is_file() {
        return Err(format!(
            "broken evaluation receipt {}: missing {}",
            receipt.display(),
            artifact.display()
        ));
    }
    Ok(artifact)
}
fn compare(runs: &[PathBuf], output: &Path) -> Result<(), String> {
    new_output(output)?;
    let mut csv = String::from("run,artifact,record\n");
    let mut report = String::from(
        "# Recorded research results\n\nNo training or hardware probe was executed by this report. Missing measurements remain absent.\n\n",
    );
    let mut json = vec![];
    let mut performance_rows = Vec::new();
    let mut summary_csv = String::from(
        "run,status,backend,rollouts,transitions,total_seconds,rollout_mean_seconds,rollout_median_seconds,rollout_p90_seconds,rollout_p95_seconds,transitions_per_second,peak_resident_kib\n",
    );
    let mut validation_csv = String::from(
        "run,track,episodes,completion_rate,progress_mean,progress_median,progress_p10,progress_p90,lap_seconds_mean\n",
    );
    for run in runs {
        if !run.is_dir() {
            return Err(format!("not a run directory: {}", run.display()));
        }
        report.push_str(&format!("## {}\n\n", run.display()));
        for name in [
            "performance_summary.json",
            "hardware.json",
            "evaluation_summary.json",
            "evaluation.csv",
            "periodic_evaluations.csv",
            "track_metrics.csv",
            "update_metrics.csv",
        ] {
            let p = artifact_path(run, name)?;
            if p.is_file() {
                let text = fs::read_to_string(&p).map_err(|e| e.to_string())?;
                csv.push_str(&format!(
                    "{},{},{}\n",
                    quote(&run.display().to_string()),
                    name,
                    quote(&text)
                ));
                if name.ends_with(".json") {
                    let value: serde_json::Value =
                        serde_json::from_str(&text).map_err(|e| e.to_string())?;
                    if name == "performance_summary.json" {
                        let fields = [
                            value.get("status"),
                            value.get("backend"),
                            value.get("completed_rollouts_in_this_run"),
                            value.get("transitions"),
                            value.get("total_seconds"),
                            value.get("rollout_mean_seconds"),
                            value.get("rollout_median_seconds"),
                            value.get("rollout_p90_seconds"),
                            value.get("rollout_p95_seconds"),
                            value.get("global_transitions_per_second"),
                            value.pointer("/memory/peak_resident_kib"),
                        ];
                        summary_csv.push_str(&quote(&run.display().to_string()));
                        for field in fields {
                            summary_csv.push(',');
                            summary_csv.push_str(&quote(
                                &field
                                    .map(|v| {
                                        v.as_str()
                                            .map(str::to_owned)
                                            .unwrap_or_else(|| v.to_string())
                                    })
                                    .unwrap_or_default(),
                            ));
                        }
                        summary_csv.push('\n');
                        performance_rows.push((run.clone(), value.clone()));
                    }
                    if name == "evaluation_summary.json" {
                        if let Some(tracks) = value.get("tracks").and_then(|v| v.as_array()) {
                            for t in tracks {
                                validation_csv.push_str(&quote(&run.display().to_string()));
                                for key in [
                                    "track",
                                    "episodes",
                                    "completion_rate",
                                    "progress_mean",
                                    "progress_median",
                                    "progress_p10",
                                    "progress_p90",
                                    "completed_lap_seconds_mean",
                                ] {
                                    validation_csv.push(',');
                                    validation_csv.push_str(&quote(
                                        &t.get(key)
                                            .map(|v| {
                                                v.as_str()
                                                    .map(str::to_owned)
                                                    .unwrap_or_else(|| v.to_string())
                                            })
                                            .unwrap_or_default(),
                                    ));
                                }
                                validation_csv.push('\n');
                            }
                        }
                    }
                    report.push_str(&format!("- {name}: included in comparison.json\n"));
                    json.push(
                        serde_json::json!({"run":run,"artifact":name,"source":p,"data":value}),
                    );
                } else {
                    report.push_str(&format!(
                        "- [{}]({})\n",
                        name,
                        fs::canonicalize(p).map_err(|e| e.to_string())?.display()
                    ));
                }
            }
        }
    }
    report.push_str("\n## Training performance\n\n| Run | Backend | Rollouts | Seconds | Transitions/s | P95 rollout |\n|---|---|---:|---:|---:|---:|\n");
    for (path, row) in &performance_rows {
        report.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} |\n",
            path.display(),
            row["backend"].as_str().unwrap_or("unknown"),
            row["completed_rollouts_in_this_run"],
            row["total_seconds"],
            row["global_transitions_per_second"],
            row["rollout_p95_seconds"]
        ));
    }
    if let Some((base_path, base)) = performance_rows.first() {
        for (path, row) in performance_rows.iter().skip(1) {
            let clean = |v: &serde_json::Value| {
                let mut c = v.get("config").cloned().unwrap_or(serde_json::Value::Null);
                if let Some(c) = c.as_object_mut() {
                    c.remove("device");
                    c.remove("results_root");
                }
                c
            };
            if clean(base) == clean(row)
                && base.get("transitions") == row.get("transitions")
                && base.get("status") == row.get("status")
            {
                if let (Some(a), Some(b)) = (
                    base["total_seconds"].as_f64(),
                    row["total_seconds"].as_f64(),
                ) {
                    if b > 0. {
                        report.push_str(&format!("\nElapsed-time ratio {} / {} = {:.3}. Verify hardware.json commit, affinity and thread limits before attributing this ratio to hardware.\n",base_path.display(),path.display(),a/b));
                    }
                }
            } else {
                report.push_str(&format!("\n{} and {} have different configurations, status or transition budgets; no speed ratio computed.\n",base_path.display(),path.display()));
            }
        }
    }
    fs::write(output.join("performance.csv"), summary_csv).map_err(|e| e.to_string())?;
    fs::write(output.join("validation.csv"), validation_csv).map_err(|e| e.to_string())?;
    fs::write(output.join("comparison.csv"), csv).map_err(|e| e.to_string())?;
    fs::write(
        output.join("comparison.json"),
        serde_json::to_string_pretty(&json).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    fs::write(output.join("report.md"), report).map_err(|e| e.to_string())?;
    Ok(())
}
fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_factorial_unique() {
        let c = cells();
        assert_eq!(c.len(), 16);
        assert_eq!(
            c.iter()
                .map(|(id, ..)| id)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            16
        );
        for (_, layout, future, _) in c {
            let c = Config::default();
            let spec = c
                .observation
                .with_sensor_layout(layout)
                .with_anticipation(future);
            assert_eq!(spec.angles().len(), 21);
            assert_eq!(spec.dimension(), if future { 31 } else { 25 });
        }
    }
    #[test]
    fn hardware_preserves_science() {
        let a = base().unwrap();
        let mut b = a.clone();
        b.device = DeviceChoice::Cpu;
        b.total_rollouts = 100;
        b.results_root = "benchmark".into();
        assert_eq!(a.observation, b.observation);
        assert_eq!(a.reward, b.reward);
        assert_eq!(b.parameter_count(), 204805);
        assert_eq!(b.rollout_size(), 8192);
    }
    #[test]
    fn comparison_includes_validation_linked_to_training_run() {
        let root = racing_rl::persistence::run_directory(
            &std::env::temp_dir().join("research-comparison-tests"),
        )
        .unwrap();
        let training = root.join("training");
        // Spaces in output roots must survive the evaluator's success marker.
        let evaluation = root.join("held out evaluation");
        fs::create_dir(&training).unwrap();
        fs::create_dir(&evaluation).unwrap();
        let summary = serde_json::json!({"tracks": [{
            "track": "unknown-track", "episodes": 4, "completion_rate": 0.75,
            "progress_mean": 0.9, "progress_median": 1.0, "progress_p10": 0.7,
            "progress_p90": 1.0, "completed_lap_seconds_mean": 12.0
        }]});
        fs::write(
            evaluation.join("evaluation_summary.json"),
            summary.to_string(),
        )
        .unwrap();
        fs::write(
            evaluation.join("evaluation.csv"),
            "track,episodes\nunknown-track,4\n",
        )
        .unwrap();
        let marker = format!(
            "diagnostics\nevaluation={} episodes=4 mode=Deterministic trained=true\n",
            evaluation.display()
        );
        let reported = evaluation_directory(&marker).unwrap();
        assert_eq!(reported, evaluation);
        assert!(evaluation_directory("failed before evaluation artifacts").is_err());
        link_evaluation(&training, &reported).unwrap();
        let receipt = training.join("evaluation-path.txt");
        let stored = fs::read_to_string(&receipt).unwrap();
        assert!(!Path::new(stored.trim()).is_absolute());
        assert_eq!(
            fs::canonicalize(training.join(stored.trim())).unwrap(),
            fs::canonicalize(&evaluation).unwrap()
        );
        assert!(link_evaluation(&training, &reported).is_err());
        assert_eq!(fs::read_to_string(&receipt).unwrap(), stored);

        // Simulate collecting this tree on a different host: the old absolute
        // locations no longer exist, but the training/evaluation link still works.
        let moved = root.with_file_name(format!(
            "{}-moved",
            root.file_name().unwrap().to_string_lossy()
        ));
        fs::rename(&root, &moved).unwrap();
        let training = moved.join("training");
        let evaluation = moved.join("held out evaluation");
        let output = moved.join("comparison");
        compare(std::slice::from_ref(&training), &output).unwrap();
        let csv = fs::read_to_string(output.join("validation.csv")).unwrap();
        assert!(csv.contains(&format!(
            "{},\"unknown-track\",\"4\",\"0.75\"",
            quote(&training.display().to_string())
        )));
        let records: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(output.join("comparison.json")).unwrap())
                .unwrap();
        assert_eq!(records[0]["run"], training.to_string_lossy().as_ref());
        assert_eq!(records[0]["data"], summary);
        assert_eq!(
            Path::new(records[0]["source"].as_str().unwrap()),
            training.join("../held out evaluation/evaluation_summary.json")
        );
        assert!(compare(std::slice::from_ref(&training), &output).is_err());
        fs::remove_file(evaluation.join("evaluation_summary.json")).unwrap();
        assert!(
            artifact_path(&training, "evaluation_summary.json")
                .unwrap_err()
                .contains("broken evaluation receipt")
        );
        fs::remove_dir_all(moved).unwrap();
    }
    #[test]
    fn matrix_policy_rejects_invalid_outputs_and_persists_real_training_args() {
        let root = racing_rl::persistence::run_directory(
            &std::env::temp_dir().join("research-policy-tests"),
        )
        .unwrap();
        let config = root.join("matrix.ron");
        let output = root.join("plan");
        let prepare = || {
            run(ResearchCommand::Matrix {
                config: config.clone(),
                output: output.clone(),
                execute: false,
                dry_run: false,
                include: vec!["original-rays-original".into()],
                screening: false,
                continue_from: None,
            })
        };
        let mut m = MatrixConfig {
            checkpoint_every: Some(0),
            ..Default::default()
        };
        save(&config, &m).unwrap();
        assert!(prepare().is_err());
        assert!(!output.exists());
        m.checkpoint_every = Some(75);
        fs::remove_file(&config).unwrap();
        save(&config, &m).unwrap();
        assert!(
            prepare()
                .unwrap_err()
                .contains("every periodic evaluation must have a persisted checkpoint")
        );
        assert!(!output.exists());
        m.checkpoint_every = Some(25);
        m.seeds = vec![11, 1_000_014];
        m.repetitions = 2;
        fs::remove_file(&config).unwrap();
        save(&config, &m).unwrap();
        assert!(prepare().unwrap_err().contains("duplicate paired seed"));
        assert!(!output.exists());

        m.seeds = vec![11];
        m.repetitions = 1;
        m.save_first = 2;
        m.save_last = 1;
        m.save_rollouts = true;
        fs::remove_file(&config).unwrap();
        save(&config, &m).unwrap();
        prepare().unwrap();
        let plan: Plan = persistence::load(&output.join("plan.ron")).unwrap();
        assert_eq!(plan.runs.len(), 1);
        let planned = &plan.runs[0];
        assert_eq!(planned.checkpoint_every, Some(25));
        assert_eq!(planned.save_first, 2);
        assert_eq!(planned.save_last, 1);
        assert!(planned.save_rollouts);
        planned.validate_persistence(500).unwrap();
        let args = planned.training_args();
        let argument = |name: &str| {
            let index = args.iter().position(|arg| arg == name).unwrap();
            args[index + 1].to_str().unwrap()
        };
        assert_eq!(argument("--save-every"), "25");
        assert_eq!(argument("--save-first"), "2");
        assert_eq!(argument("--save-last"), "1");
        assert_eq!(argument("--evaluate-every"), "100");
        assert!(!args.iter().any(|arg| arg == "--no-save-rollouts"));
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn legacy_plans_preserve_checkpoint_defaults() {
        let mut run: PlannedRun = ron::from_str(
            r#"(cell:"legacy",seed:11,config:"config.ron",receipt:"run-path.txt",resume:None,evaluate_every:50)"#,
        )
        .unwrap();
        assert_eq!(run.checkpoint_every, None);
        assert_eq!(run.save_first, 0);
        assert_eq!(run.save_last, 0);
        assert!(!run.save_rollouts);
        let args = run.training_args();
        let index = args.iter().position(|a| a == "--save-every").unwrap();
        assert_eq!(args[index + 1], "50");
        assert!(args.iter().any(|a| a == "--no-save-rollouts"));
        run.validate_persistence(101).unwrap();
        run.evaluate_every = 0;
        let args = run.training_args();
        let index = args.iter().position(|a| a == "--save-every").unwrap();
        assert_eq!(args[index + 1], "100");
        run.validate_persistence(101).unwrap();
    }
}

#[cfg(test)]
mod preparation_tests {
    use super::*;
    #[test]
    fn dry_runs_never_create_artifacts_or_initialize_gpu() {
        let root = std::env::temp_dir().join(format!("racing-preparation-{}", std::process::id()));
        assert!(!root.exists());
        run(ResearchCommand::Benchmark {
            backend: BackendChoice::Gpu,
            output: root.clone(),
            rollouts: 100,
            execute: false,
            dry_run: true,
        })
        .unwrap();
        run(ResearchCommand::Matrix {
            config: "experiments/factorial.ron".into(),
            output: root.clone(),
            execute: false,
            dry_run: true,
            include: vec![],
            screening: false,
            continue_from: None,
        })
        .unwrap();
        assert!(!root.exists());
    }
}
