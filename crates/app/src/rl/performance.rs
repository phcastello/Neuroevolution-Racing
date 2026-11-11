//! Instrumentation shared by normal training and the future hardware benchmark.
//! Collection timings come directly from the collector; PPO update is never duplicated.
use racing_rl::{Result, config::Config, rollout::Timings};
use serde::Serialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug, Default, Serialize)]
pub struct MemorySample {
    pub resident_kib: Option<u64>,
    pub peak_resident_kib: Option<u64>,
}
fn memory_from_status(status: &str) -> MemorySample {
    let kib = |field: &str| {
        status.lines().find_map(|line| {
            line.strip_prefix(field)?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
    };
    MemorySample {
        resident_kib: kib("VmRSS:"),
        peak_resident_kib: kib("VmHWM:"),
    }
}
pub fn memory() -> MemorySample {
    fs::read_to_string("/proc/self/status")
        .map(|s| memory_from_status(&s))
        .unwrap_or_default()
}
#[derive(Serialize)]
struct Hardware {
    cpu_model: Option<String>,
    logical_cpus_available: Option<usize>,
    cpus_allowed_list: Option<String>,
    os: String,
    os_release: Option<String>,
    kernel: Option<String>,
    architecture: String,
    code_version: String,
    code_commit: Option<String>,
    code_dirty: Option<bool>,
    backend: String,
    cpu_parallel_feature: bool,
    rayon_num_threads: Option<String>,
    matrixmultiply_num_threads: Option<String>,
    /// GPU details are supplied only by explicit GPU dispatch. No device probes here.
    device_choice: racing_rl::config::DeviceChoice,
}
impl Hardware {
    fn read(c: &Config, backend: &str) -> Self {
        let cpu_model = fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| {
            s.lines().find_map(|line| {
                line.strip_prefix("model name")?
                    .split_once(':')
                    .map(|(_, model)| model.trim().to_owned())
            })
        });
        let cpus_allowed_list = fs::read_to_string("/proc/self/status").ok().and_then(|s| {
            s.lines().find_map(|line| {
                line.strip_prefix("Cpus_allowed_list:")
                    .map(|v| v.trim().to_owned())
            })
        });
        let code_commit = std::process::Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_owned());
        Self {
            cpu_model,
            logical_cpus_available: std::thread::available_parallelism().ok().map(usize::from),
            cpus_allowed_list,
            os: std::env::consts::OS.into(),
            os_release: fs::read_to_string("/etc/os-release").ok(),
            kernel: fs::read_to_string("/proc/sys/kernel/osrelease")
                .ok()
                .map(|s| s.trim().to_owned()),
            architecture: std::env::consts::ARCH.into(),
            code_version: env!("CARGO_PKG_VERSION").into(),
            code_commit,
            code_dirty: std::process::Command::new("git")
                .args(["status", "--porcelain", "--untracked-files=no"])
                .output()
                .ok()
                .filter(|o| o.status.success())
                .map(|o| !o.stdout.is_empty()),
            backend: backend.into(),
            cpu_parallel_feature: racing_rl::backend::cpu_parallel_enabled(),
            rayon_num_threads: std::env::var("RAYON_NUM_THREADS").ok(),
            matrixmultiply_num_threads: std::env::var("MATMUL_NUM_THREADS").ok(),
            device_choice: c.device.clone(),
        }
    }
}
fn json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer_pretty(&mut file, value)
        .map_err(|e| racing_rl::Error::Invalid(e.to_string()))?;
    writeln!(file)?;
    file.sync_all()?;
    Ok(())
}
#[derive(Serialize)]
pub struct Summary {
    pub status: String,
    pub completed_rollouts_in_this_run: usize,
    pub transitions: u64,
    pub optimizer_updates: u64,
    pub initialization_seconds: f64,
    pub total_seconds: f64,
    pub collect_seconds: f64,
    pub update_seconds: f64,
    pub persistence_seconds: f64,
    pub rollout_mean_seconds: Option<f64>,
    pub rollout_median_seconds: Option<f64>,
    pub rollout_p90_seconds: Option<f64>,
    pub rollout_p95_seconds: Option<f64>,
    pub global_transitions_per_second: f64,
    pub memory: MemorySample,
    pub backend: String,
    pub architecture: Vec<usize>,
    pub parameter_count: usize,
    pub observation: racing_rl::config::ObservationSpec,
    pub config: Config,
    pub search_seconds: f64,
    pub search_overhead_percent_of_ppo: Option<f64>,
    pub curriculum_rollouts_cumulative: u64,
    pub selected_rollouts_cumulative: u64,
    pub distinct_generated_tracks_used: usize,
    pub distinct_selected_tracks_used: usize,
}
/// Only scalar rollout durations are retained in RAM; observations/tensors are not copied.
pub struct Performance {
    path: PathBuf,
    writer: BufWriter<File>,
    started: Instant,
    initialization_seconds: f64,
    durations: Vec<f64>,
    transitions: u64,
    optimizer_updates: u64,
    collect_seconds: f64,
    update_seconds: f64,
    persistence_seconds: f64,
    backend: String,
    config: Config,
    finished: bool,
    search_seconds: f64,
    curriculum: Option<(u64, u64, usize, usize)>,
}
impl Performance {
    pub fn start(dir: &Path, c: &Config, backend: &str, started: Instant) -> Result<Self> {
        json(&dir.join("hardware.json"), &Hardware::read(c, backend))?;
        let mut writer = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(dir.join("performance.csv"))?,
        );
        writeln!(
            writer,
            "rollout,transitions,collect_seconds,observations_seconds,upload_seconds,inference_seconds,download_seconds,physics_seconds,preparation_seconds,update_seconds,persistence_seconds,total_seconds,transitions_per_second,optimizer_updates,resident_kib,peak_resident_kib"
        )?;
        writer.flush()?;
        Ok(Self {
            path: dir.to_owned(),
            writer,
            started,
            initialization_seconds: started.elapsed().as_secs_f64(),
            durations: vec![],
            transitions: 0,
            optimizer_updates: 0,
            collect_seconds: 0.,
            update_seconds: 0.,
            persistence_seconds: 0.,
            backend: backend.into(),
            config: c.clone(),
            finished: false,
            search_seconds: 0.,
            curriculum: None,
        })
    }
    pub fn record(
        &mut self,
        rollout: u64,
        transitions: usize,
        timings: &Timings,
        update_seconds: f64,
        persistence_seconds: f64,
        total_seconds: f64,
        optimizer_updates: u64,
    ) -> Result<()> {
        racing_rl::ensure(
            [
                timings.total_seconds,
                update_seconds,
                persistence_seconds,
                total_seconds,
            ]
            .iter()
            .all(|v| v.is_finite() && *v >= 0.),
            "performance durations must be finite and nonnegative",
        )?;
        let mem = memory();
        let throughput = if total_seconds > 0. {
            transitions as f64 / total_seconds
        } else {
            0.
        };
        writeln!(
            self.writer,
            "{rollout},{transitions},{},{},{},{},{},{},{},{update_seconds},{persistence_seconds},{total_seconds},{throughput},{optimizer_updates},{},{}",
            timings.total_seconds,
            timings.observations_seconds,
            timings.upload_seconds,
            timings.inference_seconds,
            timings.download_seconds,
            timings.simulation_seconds,
            timings.preparation_seconds,
            mem.resident_kib.map(|v| v.to_string()).unwrap_or_default(),
            mem.peak_resident_kib
                .map(|v| v.to_string())
                .unwrap_or_default()
        )?;
        self.writer.flush()?;
        self.durations.push(total_seconds);
        self.transitions += transitions as u64;
        self.optimizer_updates += optimizer_updates;
        self.collect_seconds += timings.total_seconds;
        self.update_seconds += update_seconds;
        self.persistence_seconds += persistence_seconds;
        Ok(())
    }
    pub fn record_final_persistence(&mut self, seconds: f64) -> Result<()> {
        racing_rl::ensure(
            seconds.is_finite() && seconds >= 0.,
            "final persistence duration",
        )?;
        self.persistence_seconds += seconds;
        Ok(())
    }
    pub fn record_search(&mut self, seconds: f64) {
        self.search_seconds += seconds;
    }
    pub fn set_curriculum(&mut self, state: &super::curriculum::CurriculumState) {
        self.curriculum = Some((
            state.curriculum_rollouts,
            state.selected_rollouts,
            state.used_generated_tracks.len(),
            state.used_selected_tracks.len(),
        ));
    }
    pub fn finish(&mut self, status: &str) -> Result<()> {
        self.writer.flush()?;
        let mut times = self.durations.clone();
        times.sort_by(f64::total_cmp);
        let total_seconds = self.started.elapsed().as_secs_f64();
        let summary = Summary {
            status: status.into(),
            completed_rollouts_in_this_run: times.len(),
            transitions: self.transitions,
            optimizer_updates: self.optimizer_updates,
            initialization_seconds: self.initialization_seconds,
            total_seconds,
            collect_seconds: self.collect_seconds,
            update_seconds: self.update_seconds,
            persistence_seconds: self.persistence_seconds,
            rollout_mean_seconds: (!times.is_empty())
                .then(|| times.iter().sum::<f64>() / times.len() as f64),
            rollout_median_seconds: quantile(&times, 0.5),
            rollout_p90_seconds: quantile(&times, 0.9),
            rollout_p95_seconds: quantile(&times, 0.95),
            global_transitions_per_second: if total_seconds > 0. {
                self.transitions as f64 / total_seconds
            } else {
                0.
            },
            memory: memory(),
            backend: self.backend.clone(),
            architecture: self.config.hidden.clone(),
            parameter_count: self.config.parameter_count(),
            observation: self.config.observation.clone(),
            config: self.config.clone(),
            search_seconds: self.search_seconds,
            search_overhead_percent_of_ppo: (self.collect_seconds + self.update_seconds > 0.)
                .then(|| 100. * self.search_seconds / (self.collect_seconds + self.update_seconds)),
            curriculum_rollouts_cumulative: self.curriculum.map_or(0, |v| v.0),
            selected_rollouts_cumulative: self.curriculum.map_or(0, |v| v.1),
            distinct_generated_tracks_used: self.curriculum.map_or(0, |v| v.2),
            distinct_selected_tracks_used: self.curriculum.map_or(0, |v| v.3),
        };
        json(&self.path.join("performance_summary.json"), &summary)?;
        self.finished = true;
        Ok(())
    }
}
impl Drop for Performance {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.finish("failed_or_interrupted");
        }
    }
}
fn quantile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let position = q * (sorted.len() - 1) as f64;
    let index = position.floor() as usize;
    let fraction = position - index as f64;
    Some(sorted[index] * (1. - fraction) + sorted[(index + 1).min(sorted.len() - 1)] * fraction)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn percentiles_empty_single_and_interpolation() {
        assert_eq!(quantile(&[], 0.5), None);
        assert_eq!(quantile(&[2.], 0.95), Some(2.));
        assert_eq!(quantile(&[1., 2., 3., 4.], 0.5), Some(2.5));
        assert!((quantile(&[1., 2., 3., 4.], 0.9).unwrap() - 3.7).abs() < 1e-12);
    }
    #[test]
    fn linux_memory_is_optional_and_reports_kib() {
        let sample = memory_from_status("VmRSS:\t120 kB\nVmHWM:\t160 kB\n");
        assert_eq!(sample.resident_kib, Some(120));
        assert_eq!(sample.peak_resident_kib, Some(160));
        assert!(memory_from_status("").resident_kib.is_none());
    }
    #[test]
    fn records_real_totals_and_keeps_runs_isolated() {
        let dir =
            racing_rl::persistence::run_directory(&std::env::temp_dir().join("performance-tests"))
                .unwrap();
        let c = Config::default();
        let mut recorder = Performance::start(&dir, &c, "CPU NdArray", Instant::now()).unwrap();
        let timings = Timings {
            total_seconds: 1.,
            simulation_seconds: 0.6,
            ..Default::default()
        };
        recorder.record(1, 32, &timings, 0.5, 0.1, 1.6, 4).unwrap();
        recorder.finish("completed").unwrap();
        let summary: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(dir.join("performance_summary.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(summary["transitions"], 32);
        assert_eq!(summary["optimizer_updates"], 4);
        assert_eq!(summary["update_seconds"], 0.5);
        assert!(Performance::start(&dir, &c, "CPU NdArray", Instant::now()).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
