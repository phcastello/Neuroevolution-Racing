//! CLI orchestration. GA commands and all historical formats remain separate.
mod persistence_policy;
pub(crate) mod viewer;
use crate::simulation::{
    SimulationConfig, TrackLibrary,
    rl_environment::{self as env, RacingEnvironment},
};
use clap::Subcommand;
use persistence_policy::PersistenceOptions;
use racing_rl::{
    Result,
    backend::{self, Backend, Cpu, Tensor, matrix, read},
    config::{Config, DeviceChoice},
    distribution, ensure, manual,
    network::Network,
    optimizer::Optimizer,
    persistence::{self, Checkpoint, Rngs},
    ppo::{self, LossInputs},
    rollout::{self, EndReason, Environment, Rollout},
};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use serde::Serialize;
use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Subcommand)]
pub enum RlCommand {
    /// Collect one real rollout with frozen, explicitly untrained parameters.
    Collect {
        #[arg(long)]
        config: PathBuf,
    },
    /// Validate a saved time-major rollout and summarize episode boundaries.
    InspectRollout { file: PathBuf },
    /// Train PPO; fails before environment/device setup while manual hooks are pending.
    Train {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[command(flatten)]
        persistence: PersistenceOptions,
    },
    /// Deterministic evaluation on held-out tracks, with independent seeds.
    Evaluate {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value_t = 4)]
        episodes: usize,
        #[arg(long)]
        cpu: bool,
    },
    /// Warmed, synchronized inference and real-physics rollout benchmarks.
    Benchmark {
        #[arg(long)]
        config: PathBuf,
    },
    /// List cases or run real hooks against frozen central finite differences (CPU f64).
    CheckGradients {
        #[arg(long, default_value = "all")]
        case: String,
    },
    /// Inspect a frozen PPO policy with the original sprites, camera and dashboard.
    View {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value_t = 4)]
        cars: usize,
        /// Initial track ID; defaults to the first held-out evaluation track.
        #[arg(long)]
        track: Option<String>,
        /// Close after this many rendered frames (useful for a bounded preview/smoke).
        #[arg(long)]
        frames: Option<u64>,
    },
}
pub fn run(command: RlCommand) -> std::result::Result<(), String> {
    run_inner(command).map_err(|e| e.to_string())
}
fn run_inner(command: RlCommand) -> Result<()> {
    match command {
        RlCommand::InspectRollout { file } => {
            let r: Rollout = persistence::load(&file)?;
            r.validate()?;
            println!(
                "VALIDO [T={},N={},D={}] policy_version={} transitions={}\n{}",
                r.rollout_length,
                r.num_envs,
                r.observation_spec.dimension(),
                r.policy_version,
                r.transitions.len(),
                summary(&r)
            );
            Ok(())
        }
        RlCommand::CheckGradients { case } => {
            if case == "list" {
                for c in racing_rl::verification::CASES {
                    println!("{c}");
                }
                for name in manual::FUNCTIONS {
                    println!("hook {name}");
                }
                return Ok(());
            }
            racing_rl::verification::check_case(&case)?;
            println!("APROVADO: {case} (CPU f64)");
            Ok(())
        }
        RlCommand::Collect { config } => {
            let c = resolve_config(Config::load(&config)?)?;
            dispatch(&c, Job::Collect)
        }
        RlCommand::Train {
            config,
            resume,
            persistence,
        } => {
            let c = Config::load(&config)?;
            manual::preflight()?;
            let c = resolve_config(c)?;
            dispatch(&c, Job::Train(resume, persistence))
        }
        RlCommand::Benchmark { config } => {
            let c = resolve_config(Config::load(&config)?)?;
            dispatch(&c, Job::Benchmark)
        }
        RlCommand::Evaluate {
            checkpoint,
            episodes,
            cpu,
        } => {
            ensure(
                episodes > 0 && episodes <= 10000,
                "evaluation episode count",
            )?;
            let saved: Checkpoint = persistence::load(&checkpoint)?;
            saved.validate()?;
            let mut c = saved.config.clone();
            if cpu {
                c.device = DeviceChoice::Cpu;
            }
            dispatch(&c, Job::Evaluate(Box::new(saved), episodes))
        }
        RlCommand::View {
            checkpoint,
            cars,
            track,
            frames,
        } => {
            ensure((1..=16).contains(&cars), "viewer cars 1..16")?;
            ensure(frames.is_none_or(|f| f > 0), "viewer frames > 0")?;
            let saved: Checkpoint = persistence::load(&checkpoint)?;
            saved.validate()?;
            viewer::view(saved, checkpoint, cars, track, frames)
        }
    }
}
enum Job {
    Collect,
    Train(Option<PathBuf>, PersistenceOptions),
    Benchmark,
    Evaluate(Box<Checkpoint>, usize),
}
fn dispatch(c: &Config, job: Job) -> Result<()> {
    match c.device {
        DeviceChoice::Cpu => execute::<Cpu>(
            c,
            job,
            &Default::default(),
            &format!(
                "Burn 0.20.1 NdArray f32; CPU={}; OS={}",
                std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| "unreported".into()),
                std::env::consts::OS
            ),
        ),
        DeviceChoice::Gpu { adapter } => {
            #[cfg(feature = "rl-gpu")]
            {
                let (device, label) = backend::gpu_device(adapter)?;
                execute::<backend::Gpu>(c, job, &device, &label)
            }
            #[cfg(not(feature = "rl-gpu"))]
            {
                let _ = (job, adapter);
                Err(racing_rl::Error::Device(
                    "GPU requested; rebuild app with --features rl-gpu".into(),
                ))
            }
        }
    }
}
fn library() -> Result<TrackLibrary> {
    TrackLibrary::load_default().map_err(|e| racing_rl::Error::Invalid(e.to_string()))
}
fn resolve_config(mut c: Config) -> Result<Config> {
    let library = library()?;
    if c.training_tracks.is_empty() {
        c.training_tracks = library.training_tracks().map(|d| d.id.clone()).collect();
    }
    if c.evaluation_tracks.is_empty() {
        c.evaluation_tracks = library.validation_tracks().map(|d| d.id.clone()).collect();
    }
    c.validate()?;
    Ok(c)
}
fn physics() -> SimulationConfig {
    SimulationConfig::default()
}
fn validate_physics(p: &SimulationConfig) -> Result<()> {
    ensure(
        [
            p.sensor_max_distance,
            p.acceleration_rate,
            p.braking_rate,
            p.coasting_deceleration,
            p.acceleration_falloff_speed,
            p.speed_normalization_scale,
            p.turn_rate,
            p.max_lateral_acceleration,
        ]
        .iter()
        .all(|v| v.is_finite() && *v > 0.)
            && p.progress_search_radius > 0,
        "checkpoint physics",
    )
}
fn restore<B: Backend>(
    saved: &Checkpoint,
    device: &B::Device,
) -> Result<(Network<B>, Optimizer<B>, SimulationConfig)> {
    saved.validate()?;
    let mut n = Network::new(&saved.config, device)?;
    ensure(
        n.parameters().iter().map(|t| t.dims()).collect::<Vec<_>>() == saved.parameter_shapes,
        "checkpoint architecture shapes",
    )?;
    let last = saved.parameters.last().unwrap();
    ensure(
        last.iter()
            .all(|v| *v >= saved.config.log_std_bounds[0] && *v <= saved.config.log_std_bounds[1]),
        "checkpoint log_std outside projection bounds",
    )?;
    n.restore(
        &saved.parameters,
        saved.policy_version,
        saved.config.log_std_bounds,
    )?;
    let optimizer = Optimizer::restore(&saved.optimizer, &n.parameters())?;
    ensure(
        optimizer.kind == saved.config.optimizer,
        "checkpoint optimizer config",
    )?;
    let p: SimulationConfig =
        ron::from_str(&saved.physics_ron).map_err(|e| racing_rl::Error::Invalid(e.to_string()))?;
    validate_physics(&p)?;
    Ok((n, optimizer, p))
}
struct SaveContext<'a> {
    config: &'a Config,
    physics: &'a SimulationConfig,
    rngs: &'a Rngs,
    backend: &'a str,
    completed_rollouts: u64,
    decisions: u64,
    physical_ticks: u64,
}
fn checkpoint<B: Backend>(
    network: &Network<B>,
    optimizer: &Optimizer<B>,
    context: SaveContext<'_>,
) -> Result<Checkpoint> {
    let SaveContext {
        config,
        physics,
        rngs,
        backend,
        completed_rollouts,
        decisions,
        physical_ticks,
    } = context;
    Ok(Checkpoint{format:"racing-ppo".into(),version:1,config:config.clone(),features:config.observation.features(),parameter_shapes:network.parameters().iter().map(|p|p.dims()).collect(),parameters:network.snapshot()?,policy_version:network.version(),optimizer:optimizer.snapshot()?,rngs:rngs.clone(),completed_rollouts,decisions,physical_ticks,rendered_frames:0,backend:backend.into(),trained:optimizer.step>0,physics_ron:ron::to_string(physics).map_err(|e|racing_rl::Error::Invalid(e.to_string()))?,resume_policy:"restore parameters/optimizer/RNG; restart every environment; no exact trajectory continuation".into()})
}
#[derive(Serialize)]
struct Manifest<'a> {
    version: u32,
    config: &'a Config,
    backend: &'a str,
    physics: &'a SimulationConfig,
    training_tracks: Vec<String>,
    evaluation_tracks: Vec<String>,
    manual_functions: &'a [&'a str],
    gradient_status: &'a str,
    laser_enabled: bool,
    precision: &'a str,
}
fn manifest(
    dir: &Path,
    c: &Config,
    p: &SimulationConfig,
    label: &str,
    library: &TrackLibrary,
) -> Result<()> {
    persistence::save(
        &dir.join("manifest.ron"),
        &Manifest {
            version: 1,
            config: c,
            backend: label,
            physics: p,
            training_tracks: if c.training_tracks.is_empty() {
                library.training_tracks().map(|d| d.id.clone()).collect()
            } else {
                c.training_tracks.clone()
            },
            evaluation_tracks: if c.evaluation_tracks.is_empty() {
                library.validation_tracks().map(|d| d.id.clone()).collect()
            } else {
                c.evaluation_tracks.clone()
            },
            manual_functions: manual::FUNCTIONS,
            gradient_status: if manual::preflight().is_ok() {
                "implemented; no track learning claim"
            } else {
                "PENDENTE; gradient/update/learned-performance metrics unavailable"
            },
            laser_enabled: false,
            precision: "f32",
        },
    )
}
fn execute<B: Backend>(c: &Config, job: Job, device: &B::Device, label: &str) -> Result<()> {
    println!(
        "{label}\nN={} T={} N*T={} M={} E={} D={} estimated={:.3} MiB",
        c.num_envs,
        c.rollout_length,
        c.rollout_size(),
        c.minibatch_size,
        c.update_epochs,
        c.observation.dimension(),
        c.estimated_bytes() as f64 / (1024. * 1024.)
    );
    if matches!(job, Job::Benchmark) {
        return benchmark::<B>(c, device, label);
    }
    if let Job::Evaluate(saved, episodes) = job {
        return evaluate::<B>(*saved, episodes, device, label);
    }
    let library = library()?;
    let (mut network, mut optimizer, p, mut rngs, mut completed, mut decisions, mut ticks) =
        if let Job::Train(Some(ref path), _) = job {
            let saved: Checkpoint = persistence::load(path)?;
            let mut requested = c.clone();
            requested.device = saved.config.device.clone();
            requested.total_rollouts = saved.config.total_rollouts;
            ensure(requested == saved.config, "resume configuration mismatch")?;
            let (n, o, p) = restore(&saved, device)?;
            (
                n,
                o,
                p,
                saved.rngs,
                saved.completed_rollouts,
                saved.decisions,
                saved.physical_ticks,
            )
        } else {
            let n = Network::<B>::new(c, device)?;
            let o = Optimizer::new(c.optimizer.clone(), &n.parameters());
            (n, o, physics(), Rngs::new(c), 0, 0, 0)
        };
    let scenes = env::scenes(&library, &c.training_tracks, false)?;
    let mut envs = env::environments(scenes, c, &p, &rngs.environments)?;
    // Validate held-out suite too, before creating a run.
    let _evaluation = env::scenes(&library, &c.evaluation_tracks, true)?;
    let dir = persistence::run_directory(Path::new(&c.results_root))?;
    manifest(&dir, c, &p, label, &library)?;
    let mut log = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(dir.join("run.log"))?;
    writeln!(
        log,
        "{label}\n{:?}\nmanual={:?}",
        c.device,
        manual::FUNCTIONS
    )?;
    let persistence = match &job {
        Job::Train(_, options) => options.clone(),
        _ => PersistenceOptions::default(),
    };
    writeln!(
        log,
        "artifact_persistence={persistence:?}; resumed_completed_rollouts={completed}; target_rollouts={}",
        c.total_rollouts
    )?;
    let mut metrics = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(dir.join("metrics.csv"))?;
    writeln!(
        metrics,
        "rollout,policy_version,decisions,physical_ticks,rendered_frames,reward,completed,collision,timeout,seconds,transitions_per_second,gradient_status"
    )?;
    let train = matches!(job, Job::Train(..));
    let count = if train {
        c.total_rollouts.saturating_sub(completed as usize)
    } else {
        1
    };
    ensure(count > 0, "target rollouts already completed")?;
    for _ in 0..count {
        let next_completed = completed + u64::from(train);
        // This decision gates files/snapshots only. Collection, every update,
        // RNG advancement, counters and telemetry still run for every rollout.
        let save_heavy = !train || persistence.should_save(next_completed, c.total_rollouts);
        let r = rollout::collect(&network, &mut envs, c, &mut rngs.action, device)?;
        println!("{}", summary(&r));
        writeln!(log, "rollout={next_completed} {}", summary(&r))?;
        if train {
            let start = Instant::now();
            let losses = racing_rl::trainer::update(
                &mut network,
                &mut optimizer,
                &r,
                c,
                &mut rngs.shuffle,
                device,
            )?;
            backend::sync::<B>(device)?;
            if save_heavy {
                let index = persistence.losses_index(next_completed);
                persistence::save(&dir.join(format!("losses-{index:06}.ron")), &losses)?;
            }
            writeln!(log, "update_seconds={}", start.elapsed().as_secs_f64())?;
        } else {
            // Reevaluate the SAME collected u; loss forward works without derivative hooks.
            let loss = preview_loss(&network, &r, c, device)?;
            println!("loss forward (untrained): {loss:?}");
            persistence::save(&dir.join("loss-forward.ron"), &loss)?;
        }
        decisions += r.transitions.len() as u64;
        ticks += r.transitions.iter().map(|s| s.physical_ticks).sum::<u64>();
        if train {
            completed += 1;
        }
        rngs.environments = envs.iter().map(|e| e.rng.clone()).collect();
        if save_heavy {
            let saved = checkpoint(
                &network,
                &optimizer,
                SaveContext {
                    config: c,
                    physics: &p,
                    rngs: &rngs,
                    backend: label,
                    completed_rollouts: completed,
                    decisions,
                    physical_ticks: ticks,
                },
            )?;
            persistence::save(&dir.join(format!("checkpoint-{completed:06}.ron")), &saved)?;
            persistence::save(&dir.join(format!("rollout-{completed:06}.ron")), &r)?;
        }
        let counts = counts(&r);
        writeln!(
            metrics,
            "{completed},{},{},{},0,{},{},{},{},{},{},{}",
            network.version(),
            decisions,
            ticks,
            r.transitions
                .iter()
                .map(|s| s.reward.total() as f64)
                .sum::<f64>(),
            counts[0],
            counts[1],
            counts[2],
            r.timings.total_seconds,
            r.transitions.len() as f64 / r.timings.total_seconds,
            if train { "available" } else { "PENDENTE" }
        )?;
        metrics.flush()?;
        log.flush()?;
    }
    println!("run={}", dir.display());
    Ok(())
}
fn preview_loss<B: Backend>(
    n: &Network<B>,
    r: &Rollout,
    c: &Config,
    d: &B::Device,
) -> Result<ppo::LossMetrics> {
    let count = r.transitions.len().min(c.minibatch_size);
    let steps = &r.transitions[..count];
    let x = matrix(
        &steps
            .iter()
            .flat_map(|s| s.observation.iter().copied())
            .collect::<Vec<_>>(),
        count,
        c.observation.dimension(),
        d,
    )?;
    let out = n.infer(x)?;
    let u = matrix(
        &steps.iter().flat_map(|s| s.u).collect::<Vec<_>>(),
        count,
        2,
        d,
    )?;
    let dist = distribution::evaluate(out.mean, n.log_std.clone(), u)?;
    let inputs = LossInputs {
        old_log_prob: matrix(
            &steps.iter().map(|s| s.old_log_prob).collect::<Vec<_>>(),
            count,
            1,
            d,
        )?,
        advantage: matrix(&r.policy_advantages[..count], count, 1, d)?,
        value_target: matrix(&r.value_targets[..count], count, 1, d)?,
    };
    Ok(ppo::forward(dist, out.value, inputs, c)?.1)
}
fn counts(r: &Rollout) -> [usize; 3] {
    [
        EndReason::Completed,
        EndReason::Collision,
        EndReason::Timeout,
    ]
    .map(|reason| {
        r.transitions
            .iter()
            .filter(|s| s.reason == Some(reason))
            .count()
    })
}
fn summary(r: &Rollout) -> String {
    let count = counts(r);
    let episodes = r
        .transitions
        .iter()
        .map(|s| (s.env_id, s.episode_id))
        .collect::<std::collections::HashSet<_>>()
        .len();
    let parts = [
        r.transitions
            .iter()
            .map(|s| s.reward.progress as f64)
            .sum::<f64>(),
        r.transitions
            .iter()
            .map(|s| s.reward.completion as f64)
            .sum::<f64>(),
        r.transitions
            .iter()
            .map(|s| s.reward.collision as f64)
            .sum::<f64>(),
    ];
    format!(
        "episodes={episodes} terminated={} truncated={} reward_parts[progress,completion,collision]={parts:?} transitions={} decisions={} physical_ticks={} rendered_frames=0 completed={} collision={} timeout={} reward={} seconds={:.6} transitions/s={:.2} timings={:?}",
        count[0] + count[1],
        count[2],
        r.transitions.len(),
        r.transitions.len(),
        r.transitions.iter().map(|s| s.physical_ticks).sum::<u64>(),
        count[0],
        count[1],
        count[2],
        r.transitions
            .iter()
            .map(|s| s.reward.total() as f64)
            .sum::<f64>(),
        r.timings.total_seconds,
        r.transitions.len() as f64 / r.timings.total_seconds,
        r.timings
    )
}
#[derive(Serialize, Debug)]
struct EvaluationEpisode {
    track: String,
    seed: u64,
    reward: f64,
    completed: bool,
    collision: bool,
    timeout: bool,
    progress: f32,
    decisions: u64,
    physical_ticks: u64,
    seconds: f64,
    trained: bool,
}
fn evaluate<B: Backend>(
    saved: Checkpoint,
    episodes: usize,
    device: &B::Device,
    label: &str,
) -> Result<()> {
    let (n, _, p) = restore::<B>(&saved, device)?;
    let c = &saved.config;
    let library = library()?;
    let scenes = env::scenes(&library, &c.evaluation_tracks, true)?;
    let mut results = vec![];
    for scene in scenes {
        for i in 0..episodes {
            let seed = c.evaluation_seed.wrapping_add(i as u64);
            let mut e = RacingEnvironment::new(
                0,
                vec![scene.clone()],
                c,
                &p,
                ChaCha12Rng::seed_from_u64(seed),
            )?;
            let mut reward = 0.;
            let mut decisions = 0;
            let mut ticks = 0;
            let start = Instant::now();
            let final_step = loop {
                let x = matrix(&e.observe()?, 1, c.observation.dimension(), device)?;
                let action = read(n.infer(x)?.mean.tanh())?;
                let s = e.step([action[0], action[1]])?;
                reward += s.reward.total() as f64;
                decisions += 1;
                ticks += s.physical_ticks;
                if s.terminated || s.truncated {
                    break s;
                }
            };
            let result = EvaluationEpisode {
                track: scene.track.definition.id.clone(),
                seed,
                reward,
                completed: final_step.reason == Some(EndReason::Completed),
                collision: final_step.reason == Some(EndReason::Collision),
                timeout: final_step.truncated,
                progress: final_step.progress,
                decisions,
                physical_ticks: ticks,
                seconds: start.elapsed().as_secs_f64(),
                trained: saved.trained,
            };
            println!("{result:?}");
            results.push(result);
        }
    }
    let dir = persistence::run_directory(Path::new(&c.results_root))?;
    manifest(&dir, c, &p, label, &library)?;
    persistence::save(&dir.join("evaluation.ron"), &results)?;
    println!(
        "evaluation={} episodes={} label={}",
        dir.display(),
        results.len(),
        if saved.trained {
            "updated policy (no convergence claim)"
        } else {
            "UNTRAINED random initialization"
        }
    );
    Ok(())
}
#[derive(Serialize, Debug)]
struct BenchmarkRow {
    num_envs: usize,
    backend: String,
    inference_seconds: f64,
    inference_latency_ms: f64,
    inference_rows_per_second: f64,
    upload_seconds: f64,
    download_seconds: f64,
    max_cpu_difference: f32,
    rollout: rollout::Timings,
    transitions_per_second: f64,
    update_status: String,
}
fn benchmark<B: Backend>(c: &Config, device: &B::Device, label: &str) -> Result<()> {
    let library = library()?;
    let p = physics();
    let scenes = env::scenes(&library, &c.training_tracks, false)?;
    let mut rows = vec![];
    for &num_envs in &c.benchmark_envs {
        let mut bc = c.clone();
        bc.num_envs = num_envs;
        bc.minibatch_size = bc.minibatch_size.min(bc.rollout_size());
        bc.validate()?;
        let n = Network::<B>::new(&bc, device)?;
        let cpu = Network::<Cpu>::new(&bc, &Default::default())?;
        let input: Vec<_> = (0..num_envs * c.observation.dimension())
            .map(|i| ((i % 17) as f32 - 8.) / 16.)
            .collect();
        let now = Instant::now();
        let x = matrix(&input, num_envs, c.observation.dimension(), device)?;
        backend::sync::<B>(device)?;
        let upload_seconds = now.elapsed().as_secs_f64();
        let expected = cpu.infer(matrix(
            &input,
            num_envs,
            c.observation.dimension(),
            &Default::default(),
        )?)?;
        let expected = read(Tensor::cat(vec![expected.mean, expected.value], 1))?;
        for _ in 0..c.benchmark_warmup {
            let out = n.infer(x.clone())?;
            backend::validate(&out.mean, "benchmark warmup")?;
        }
        backend::sync::<B>(device)?;
        let now = Instant::now();
        let mut last = None;
        for _ in 0..c.benchmark_iterations {
            last = Some(n.infer(x.clone())?);
            backend::sync::<B>(device)?;
        }
        let seconds = now.elapsed().as_secs_f64();
        let now = Instant::now();
        let out = last.unwrap();
        let actual = read(Tensor::cat(vec![out.mean, out.value], 1))?;
        let download_seconds = now.elapsed().as_secs_f64();
        let max_cpu_difference = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        ensure(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() <= 1e-4 + 1e-3 * b.abs()),
            "CPU/GPU forward parity tolerance",
        )?;
        let mut rng = ChaCha12Rng::seed_from_u64(c.action_seed);
        let mut envs = env::seeded_environments(scenes.clone(), &bc, &p, c.environment_seed)?;
        for _ in 0..c.benchmark_warmup.min(3) {
            let _ = rollout::collect(&n, &mut envs, &bc, &mut rng, device)?;
        }
        let r = rollout::collect(&n, &mut envs, &bc, &mut rng, device)?;
        let row = BenchmarkRow {
            num_envs,
            backend: label.into(),
            inference_seconds: seconds,
            inference_latency_ms: seconds * 1000. / c.benchmark_iterations as f64,
            inference_rows_per_second: num_envs as f64 * c.benchmark_iterations as f64 / seconds,
            upload_seconds,
            download_seconds,
            max_cpu_difference,
            transitions_per_second: r.transitions.len() as f64 / r.timings.total_seconds,
            rollout: r.timings,
            update_status: "UNAVAILABLE until manual backward; not estimated".into(),
        };
        println!("{row:?}");
        rows.push(row);
    }
    let dir = persistence::run_directory(Path::new(&c.results_root))?;
    manifest(&dir, c, &p, label, &library)?;
    persistence::save(&dir.join("benchmark.ron"), &rows)?;
    println!("benchmark={}", dir.display());
    Ok(())
}
