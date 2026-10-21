//! Frozen policy evaluation. Action RNGs never share state with environment or training RNGs.
use super::*;
use crate::simulation::{
    Track,
    rl_environment::{RacingEnvironment, Scene},
};
use clap::{Args, ValueEnum};
use std::{collections::BTreeSet, rc::Rc};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
pub(super) enum EvaluationMode {
    Deterministic,
    Stochastic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, ValueEnum)]
pub(super) enum EvaluationTracks {
    Validation,
    Training,
    Both,
}

#[derive(Args, Clone, Debug, Serialize)]
pub struct EvaluationOptions {
    /// Number of independent episodes per selected track.
    #[arg(long, default_value_t = 4)]
    episodes: usize,
    #[arg(long)]
    pub cpu: bool,
    /// stochastic samples the saved Actor's Gaussian and then applies tanh.
    #[arg(long, value_enum, default_value = "deterministic")]
    mode: EvaluationMode,
    /// Select saved training/validation suites; this never changes learning tracks.
    #[arg(long, value_enum, default_value = "validation")]
    tracks: EvaluationTracks,
    /// Evaluate only these track IDs (repeat this option to select several).
    #[arg(long = "track", conflicts_with = "tracks")]
    track_ids: Vec<String>,
    /// Root for independent action streams; defaults to the saved evaluation_seed.
    #[arg(long)]
    action_seed: Option<u64>,
    /// Root for environment streams; defaults to the saved evaluation_seed.
    #[arg(long)]
    environment_seed: Option<u64>,
}

impl EvaluationOptions {
    pub(super) fn validate(&self) -> Result<()> {
        ensure(
            (1..=10_000).contains(&self.episodes),
            "evaluation episode count",
        )
    }

    fn selected_tracks(&self, c: &Config, library: &TrackLibrary) -> Result<Vec<String>> {
        let suite = |training: bool| {
            let saved = if training {
                &c.training_tracks
            } else {
                &c.evaluation_tracks
            };
            if !saved.is_empty() {
                saved.clone()
            } else if training {
                library.training_tracks().map(|d| d.id.clone()).collect()
            } else {
                library.validation_tracks().map(|d| d.id.clone()).collect()
            }
        };
        let ids = if !self.track_ids.is_empty() {
            self.track_ids.clone()
        } else {
            match self.tracks {
                EvaluationTracks::Validation => suite(false),
                EvaluationTracks::Training => suite(true),
                EvaluationTracks::Both => suite(true).into_iter().chain(suite(false)).collect(),
            }
        };
        ensure(!ids.is_empty(), "empty evaluation track selection")?;
        let mut seen = BTreeSet::new();
        for id in &ids {
            ensure(
                library.definition(id).is_some(),
                &format!("unknown track {id}"),
            )?;
            ensure(seen.insert(id), &format!("duplicate evaluation track {id}"))?;
        }
        Ok(ids)
    }
}

// Stable FNV-1a track hash followed by SplitMix64. No platform-dependent hash state,
// track-list order, policy RNG advancement or environment draws enter action seeds.
fn episode_seed(root: u64, track: &str, episode: usize, domain: u64) -> u64 {
    let hash = track.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    });
    let mut x =
        (root ^ hash ^ domain).wrapping_add((episode as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
const ACTION_DOMAIN: u64 = 0x6163_7469_6f6e_735f;

fn environment_seed(root: u64, episode: usize) -> u64 {
    // Preserve the historical deterministic evaluation environments. Only actions
    // use the new per-track, domain-separated stream.
    root.wrapping_add(episode as u64)
}

fn action<B: Backend>(
    n: &Network<B>,
    observation: &[f32],
    mode: EvaluationMode,
    rng: &mut ChaCha12Rng,
    device: &B::Device,
) -> Result<[f32; 2]> {
    let mean = n.infer(matrix(observation, 1, n.input_size, device)?)?.mean;
    let a = match mode {
        EvaluationMode::Deterministic => mean.tanh(),
        EvaluationMode::Stochastic => distribution::sample(mean, n.log_std.clone(), rng)?.action,
    };
    let values = read(a)?;
    Ok([values[0], values[1]])
}

#[derive(Serialize, Debug)]
struct EvaluationEpisode {
    track: String,
    track_role: String,
    episode: usize,
    mode: EvaluationMode,
    /// Environment seed (the historical field name is retained).
    seed: u64,
    action_seed: u64,
    reward: f64,
    completed: bool,
    collision: bool,
    timeout: bool,
    progress: f32,
    decisions: u64,
    physical_ticks: u64,
    collision_events: CollisionEvents,
    seconds: f64,
    simulated_seconds: f64,
    trained: bool,
}

#[derive(Default, Serialize, Debug)]
struct TrackSummary {
    track: String,
    track_role: String,
    episodes: usize,
    completed: usize,
    collision: usize,
    timeout: usize,
    progress_mean: f64,
    progress_max: f32,
    reward_sum: f64,
    reward_mean: f64,
    decisions: u64,
    physical_ticks: u64,
    collision_events: CollisionEvents,
}
impl TrackSummary {
    fn record(&mut self, e: &EvaluationEpisode) {
        self.track.clone_from(&e.track);
        self.track_role.clone_from(&e.track_role);
        self.episodes += 1;
        self.completed += usize::from(e.completed);
        self.collision += usize::from(e.collision);
        self.timeout += usize::from(e.timeout);
        self.progress_mean += (e.progress as f64 - self.progress_mean) / self.episodes as f64;
        self.progress_max = self.progress_max.max(e.progress);
        self.reward_sum += e.reward;
        self.reward_mean = self.reward_sum / self.episodes as f64;
        self.decisions += e.decisions;
        self.physical_ticks += e.physical_ticks;
        self.collision_events.physical_ticks += e.collision_events.physical_ticks;
        self.collision_events.translation += e.collision_events.translation;
        self.collision_events.rotation += e.collision_events.rotation;
        self.collision_events.rotation_only += e.collision_events.rotation_only;
    }
}

#[derive(Serialize)]
struct EvaluationReport<'a> {
    version: u32,
    options: &'a EvaluationOptions,
    environment_seed_root: u64,
    action_seed_root: u64,
    policy_version: u64,
    tracks: &'a [TrackSummary],
}

fn frozen_episodes<B: Backend>(
    saved: &Checkpoint,
    options: EvaluationOptions,
    device: &B::Device,
) -> Result<(Vec<EvaluationEpisode>, Vec<TrackSummary>)> {
    options.validate()?;
    let (n, _, p) = restore::<B>(saved, device)?;
    let c = &saved.config;
    let library = library()?;
    let ids = options.selected_tracks(c, &library)?;
    let environment_seed_root = options.environment_seed.unwrap_or(c.evaluation_seed);
    let action_seed_root = options.action_seed.unwrap_or(c.evaluation_seed);
    let mut results = vec![];
    let mut tracks = vec![];
    for id in ids {
        // Evaluation may inspect either suite. Training's env::scenes leakage guard
        // and Config.training_tracks remain unchanged.
        let definition = library.definition(&id).expect("validated evaluation track");
        let track = Track::from_definition(definition)
            .map_err(|e| racing_rl::Error::Invalid(e.to_string()))?;
        let scene = Rc::new(Scene::new(track));
        let mut summary = TrackSummary::default();
        for i in 0..options.episodes {
            let seed = environment_seed(environment_seed_root, i);
            let action_seed = episode_seed(action_seed_root, &id, i, ACTION_DOMAIN);
            let mut action_rng = ChaCha12Rng::seed_from_u64(action_seed);
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
                let a = action(&n, &e.observe()?, options.mode, &mut action_rng, device)?;
                let s = e.step(a)?;
                reward += s.reward.total() as f64;
                decisions += 1;
                ticks += s.physical_ticks;
                if s.terminated || s.truncated {
                    break s;
                }
            };
            let result = EvaluationEpisode {
                track: id.clone(),
                track_role: definition.role.label().into(),
                episode: i,
                mode: options.mode,
                seed,
                action_seed,
                reward,
                completed: final_step.reason == Some(EndReason::Completed),
                collision: final_step.reason == Some(EndReason::Collision),
                timeout: final_step.truncated,
                progress: final_step.progress,
                decisions,
                physical_ticks: ticks,
                collision_events: e.collision_events,
                seconds: start.elapsed().as_secs_f64(),
                simulated_seconds: ticks as f64 * c.physical_dt as f64,
                trained: saved.trained,
            };
            summary.record(&result);
            println!("{result:?}");
            results.push(result);
        }
        println!("{summary:?}");
        tracks.push(summary);
    }
    Ok((results, tracks))
}

pub(super) fn evaluate<B: Backend>(
    saved: Checkpoint,
    options: EvaluationOptions,
    device: &B::Device,
    label: &str,
) -> Result<()> {
    let (results, tracks) = frozen_episodes::<B>(&saved, options.clone(), device)?;
    let c = &saved.config;
    let p =
        ron::from_str(&saved.physics_ron).map_err(|e| racing_rl::Error::Invalid(e.to_string()))?;
    let library = library()?;
    let environment_seed_root = options.environment_seed.unwrap_or(c.evaluation_seed);
    let action_seed_root = options.action_seed.unwrap_or(c.evaluation_seed);
    let dir = persistence::run_directory(Path::new(&c.results_root))?;
    manifest(&dir, c, &p, label, &library)?;
    persistence::save(&dir.join("evaluation.ron"), &results)?;
    persistence::save(
        &dir.join("evaluation_summary.ron"),
        &EvaluationReport {
            version: 1,
            options: &options,
            environment_seed_root,
            action_seed_root,
            policy_version: saved.policy_version,
            tracks: &tracks,
        },
    )?;
    println!(
        "evaluation={} episodes={} mode={:?} trained={}",
        dir.display(),
        results.len(),
        options.mode,
        saved.trained
    );
    Ok(())
}

#[cfg(test)]
mod diagnostic;
#[cfg(test)]
mod tests;

pub(super) fn write_periodic_header(w: &mut impl Write) -> Result<()> {
    writeln!(
        w,
        "experiment,rollout,checkpoint,track,completed,collision,timeout,progress,reward,decisions,physical_ticks,simulated_seconds,evaluation_wall_seconds,environment_seed,action_seed"
    )?;
    Ok(())
}

pub(super) fn csv_string(s: &str) -> String {
    format!("\"{}\"", s.replace('"', "\"\""))
}

/// Only the immutable saved snapshot crosses this boundary. Training's network,
/// optimizer, environments, scenes and RNGs cannot be accessed by this evaluator.
#[derive(Clone, Debug)]
pub(super) struct Score {
    pub path: PathBuf,
    pub rollout: u64,
    pub adam_step: u64,
    pub completed: usize,
    pub mean_progress: f64,
}
impl Score {
    pub fn better_than(&self, old: &Self) -> bool {
        self.completed > old.completed
            || (self.completed == old.completed
                && (self.mean_progress > old.mean_progress
                    || (self.mean_progress == old.mean_progress && self.rollout < old.rollout)))
    }
}

pub(super) fn periodic(
    saved: &Checkpoint,
    checkpoint_path: &Path,
    experiment: &str,
    writer: &mut impl Write,
) -> Result<Score> {
    let options = EvaluationOptions {
        episodes: 1,
        cpu: true,
        mode: EvaluationMode::Deterministic,
        tracks: EvaluationTracks::Training,
        track_ids: vec![],
        action_seed: None,
        environment_seed: None,
    };
    let (results, _) = frozen_episodes::<Cpu>(saved, options, &Default::default())?;
    let checkpoint_path = std::fs::canonicalize(checkpoint_path)?;
    let score = Score {
        path: checkpoint_path.clone(),
        rollout: saved.completed_rollouts,
        adam_step: saved.optimizer.step,
        completed: results.iter().filter(|e| e.completed).count(),
        mean_progress: results.iter().map(|e| e.progress as f64).sum::<f64>()
            / results.len() as f64,
    };
    for e in results {
        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            csv_string(experiment),
            saved.completed_rollouts,
            csv_string(&checkpoint_path.to_string_lossy()),
            csv_string(&e.track),
            e.completed,
            e.collision,
            e.timeout,
            e.progress,
            e.reward,
            e.decisions,
            e.physical_ticks,
            e.simulated_seconds,
            e.seconds,
            e.seed,
            e.action_seed
        )?;
    }
    writer.flush()?;
    println!(
        "Periodic evaluation | experiment={experiment} rollout={} | deterministic CPU, one attempt per training track",
        saved.completed_rollouts
    );
    Ok(score)
}
