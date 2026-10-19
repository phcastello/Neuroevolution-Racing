//! Read-only policy history. PPO checkpoints are never interpreted as GA individuals.
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use bevy::prelude::Resource;
use bevy_egui::egui;
use racing_rl::persistence::{self, Checkpoint};

#[derive(Debug)]
struct PolicySummary {
    rollout: u64,
    version: u64,
    architecture: String,
    parameters: usize,
    trained: bool,
}

#[derive(Debug)]
struct PolicyEntry {
    path: PathBuf,
    summary: Result<PolicySummary, String>,
}

#[derive(Resource, Default)]
pub(super) struct PpoCheckpointBrowser {
    source: Option<(PathBuf, String)>,
    roots: Vec<PathBuf>,
    entries: Vec<PolicyEntry>,
    directory_input: String,
    status: String,
}

impl PpoCheckpointBrowser {
    /// Discover the configured run root, repository history, and current run.
    /// Discovery is done only on source changes or an explicit refresh.
    pub(super) fn sync_source(&mut self, checkpoint: &Path, results_root: &str) {
        let source = (
            checkpoint.parent().unwrap_or(Path::new(".")).to_path_buf(),
            results_root.to_owned(),
        );
        if self.source.as_ref() == Some(&source) {
            return;
        }
        self.source = Some(source);
        let mut roots = vec![
            PathBuf::from(results_root),
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results_ppo"),
        ];
        if let Some(parent) = checkpoint.parent() {
            roots.push(parent.to_path_buf());
        }
        roots.extend(self.roots.iter().cloned());
        let roots = normalize_roots(roots);
        if roots != self.roots {
            self.roots = roots;
            self.directory_input = results_root.to_owned();
            self.refresh();
        }
    }

    fn refresh(&mut self) {
        let mut paths = BTreeSet::new();
        let mut visited = BTreeSet::new();
        let mut failures = Vec::new();
        for root in &self.roots {
            collect_paths(root, &mut paths, &mut visited, &mut failures);
        }
        self.entries = paths
            .into_iter()
            .map(|path| {
                let summary = load_summary(&path);
                PolicyEntry { path, summary }
            })
            .collect();
        self.entries.sort_by(|a, b| {
            a.path
                .parent()
                .cmp(&b.path.parent())
                .then_with(|| rollout(a).cmp(&rollout(b)))
                .then_with(|| a.path.cmp(&b.path))
        });
        let invalid = self
            .entries
            .iter()
            .filter(|entry| entry.summary.is_err())
            .count();
        self.status = format!("{} policies; {invalid} invalid", self.entries.len());
        if let Some(error) = failures.first() {
            self.status.push_str(&format!(". {error}"));
        }
    }

    fn neighbor(&self, loaded: &Path, direction: isize) -> Option<PathBuf> {
        let loaded = fs::canonicalize(loaded).unwrap_or_else(|_| loaded.to_path_buf());
        let run: Vec<_> = self
            .entries
            .iter()
            .filter(|entry| entry.path.parent() == loaded.parent() && entry.summary.is_ok())
            .collect();
        let index = run.iter().position(|entry| entry.path == loaded)?;
        index
            .checked_add_signed(direction)
            .and_then(|next| run.get(next))
            .map(|entry| entry.path.clone())
    }
}

fn normalize_roots(roots: Vec<PathBuf>) -> Vec<PathBuf> {
    roots
        .into_iter()
        .map(|root| fs::canonicalize(&root).unwrap_or(root))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn rollout(entry: &PolicyEntry) -> u64 {
    entry
        .summary
        .as_ref()
        .map_or(u64::MAX, |summary| summary.rollout)
}

fn collect_paths(
    directory: &Path,
    paths: &mut BTreeSet<PathBuf>,
    visited: &mut BTreeSet<PathBuf>,
    failures: &mut Vec<String>,
) {
    // Do not follow directory symlinks/junctions into unrelated trees.
    let key = fs::canonicalize(directory).unwrap_or_else(|_| directory.to_path_buf());
    if !visited.insert(key) || !directory.exists() {
        return;
    }
    let entries = match fs::read_dir(directory) {
        Ok(entries) => entries,
        Err(error) => {
            failures.push(format!("{}: {error}", directory.display()));
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                failures.push(error.to_string());
                continue;
            }
        };
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let path = entry.path();
        if kind.is_dir() && !kind.is_symlink() {
            collect_paths(&path, paths, visited, failures);
        } else if kind.is_file()
            && path.extension().is_some_and(|extension| extension == "ron")
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("checkpoint-"))
        {
            paths.insert(fs::canonicalize(&path).unwrap_or(path));
        }
    }
}

fn load_summary(path: &Path) -> Result<PolicySummary, String> {
    let saved: Checkpoint = persistence::load(path).map_err(|error| error.to_string())?;
    saved.validate().map_err(|error| error.to_string())?;
    let trunk = saved
        .config
        .hidden
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(" → ");
    Ok(PolicySummary {
        rollout: saved.completed_rollouts,
        version: saved.policy_version,
        architecture: format!(
            "{} → {trunk} → Actor 2 / Critic 1",
            saved.config.observation.dimension()
        ),
        parameters: saved.parameters.iter().map(Vec::len).sum(),
        trained: saved.trained,
    })
}

pub(super) fn draw_policy_history(
    ui: &mut egui::Ui,
    browser: &mut PpoCheckpointBrowser,
    loaded: &Path,
    request: &mut Option<PathBuf>,
) {
    ui.heading("PPO policy history");
    ui.small("Frozen policies by completed rollout/update. Selecting one restarts playback with its saved policy.");
    ui.horizontal(|ui| {
        if ui.button("Refresh").clicked() {
            browser.refresh();
        }
        for (label, direction) in [("← Previous", -1), ("Next →", 1)] {
            let neighbor = browser.neighbor(loaded, direction);
            if ui
                .add_enabled(neighbor.is_some(), egui::Button::new(label))
                .clicked()
            {
                *request = neighbor;
            }
        }
    });
    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(&mut browser.directory_input)
                .desired_width((ui.available_width() - 55.0).max(80.0)),
        );
        if ui.button("Browse").clicked() {
            let directory = PathBuf::from(browser.directory_input.trim());
            if directory.is_dir() {
                browser.roots = normalize_roots(
                    browser
                        .roots
                        .iter()
                        .cloned()
                        .chain(std::iter::once(directory))
                        .collect(),
                );
                browser.refresh();
            } else {
                browser.status = "Enter an existing policy directory".into();
            }
        }
    });
    ui.small(&browser.status);
    let canonical_loaded = fs::canonicalize(loaded).unwrap_or_else(|_| loaded.to_path_buf());
    let mut start = 0;
    while start < browser.entries.len() {
        let folder = browser.entries[start]
            .path
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf();
        let end = browser.entries[start..]
            .iter()
            .position(|entry| entry.path.parent() != Some(folder.as_path()))
            .map_or(browser.entries.len(), |offset| start + offset);
        let title = format!(
            "{} ({})",
            folder.file_name().unwrap_or_default().to_string_lossy(),
            end - start
        );
        egui::CollapsingHeader::new(title)
            .id_salt(("ppo_run", &folder))
            .default_open(canonical_loaded.parent() == Some(folder.as_path()))
            .show(ui, |ui| {
                ui.small(folder.display().to_string());
                for entry in &browser.entries[start..end] {
                    let name = entry.path.file_name().unwrap_or_default().to_string_lossy();
                    match &entry.summary {
                        Ok(summary) => {
                            let response = ui.selectable_label(
                                entry.path == canonical_loaded,
                                format!("{name} · rollout {}", summary.rollout),
                            );
                            if response.clicked() {
                                *request = Some(entry.path.clone());
                            }
                            response.on_hover_text(format!(
                                "{}\n{} parameters · policy version {}\n{}",
                                summary.architecture,
                                summary.parameters,
                                summary.version,
                                if summary.trained {
                                    "Trained PPO"
                                } else {
                                    "Untrained parameters"
                                }
                            ));
                        }
                        Err(error) => {
                            ui.add_enabled(
                                false,
                                egui::Label::new(format!("{name} [invalid PPO]")),
                            )
                            .on_disabled_hover_text(error);
                        }
                    }
                }
            });
        start = end;
    }
    if browser.entries.is_empty() {
        ui.small(
            "No checkpoint-*.ron PPO policies found. Use Browse to include another run directory.",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use racing_rl::{backend::Cpu, config::Config, network::Network, optimizer::Optimizer};

    fn saved_policy(rollout: u64) -> Checkpoint {
        let config = Config {
            hidden: vec![3, 2],
            ..Default::default()
        };
        let network = Network::<Cpu>::new(&config, &Default::default()).unwrap();
        let optimizer = Optimizer::new(config.optimizer.clone(), &network.parameters());
        Checkpoint {
            format: "racing-ppo".into(),
            version: 1,
            features: config.observation.features(),
            parameter_shapes: network
                .parameters()
                .iter()
                .map(|parameter| parameter.dims())
                .collect(),
            parameters: network.snapshot().unwrap(),
            optimizer: optimizer.snapshot().unwrap(),
            rngs: persistence::Rngs::new(&config),
            config,
            policy_version: rollout,
            completed_rollouts: rollout,
            decisions: 0,
            physical_ticks: 0,
            rendered_frames: 0,
            backend: "history test CPU".into(),
            trained: false,
            physics_ron: String::new(),
            resume_policy: "restart environments".into(),
        }
    }

    #[test]
    fn policy_history_uses_checkpoint_rollout_and_keeps_navigation_in_its_run() {
        let directory =
            std::env::temp_dir().join(format!("racing-ppo-policy-history-{}", std::process::id()));
        let early = directory.join("run/checkpoint-000050.ron");
        let late = directory.join("run/checkpoint-000001.ron");
        let other = directory.join("other/checkpoint-000025.ron");
        let saved = saved_policy(1);
        persistence::save(&early, &saved).unwrap();
        persistence::save(&late, &saved_policy(50)).unwrap();
        persistence::save(&other, &saved_policy(25)).unwrap();
        let mut browser = PpoCheckpointBrowser {
            roots: vec![directory.clone()],
            ..Default::default()
        };
        browser.refresh();
        let summary = load_summary(&early).unwrap();
        assert_eq!(summary.rollout, 1);
        assert_eq!(summary.parameters, saved.config.parameter_count());
        assert_eq!(summary.architecture, "25 → 3 → 2 → Actor 2 / Critic 1");
        assert_eq!(
            browser.neighbor(&early, 1),
            Some(fs::canonicalize(&late).unwrap())
        );
        assert_eq!(
            browser.neighbor(&late, -1),
            Some(fs::canonicalize(&early).unwrap())
        );
        assert!(browser.neighbor(&early, -1).is_none());
        assert!(browser.neighbor(&late, 1).is_none());
        assert!(browser.neighbor(&other, 1).is_none());
        for path in [&early, &late, &other] {
            fs::remove_file(path).unwrap();
        }
        fs::remove_dir(directory.join("run")).unwrap();
        fs::remove_dir(directory.join("other")).unwrap();
        fs::remove_dir(directory).unwrap();
    }

    #[test]
    fn discovery_is_recursive_deduplicated_and_excludes_ga_and_other_files() {
        let directory =
            std::env::temp_dir().join(format!("racing-ppo-history-{}", std::process::id()));
        let nested = directory.join("run/subrun");
        fs::create_dir_all(&nested).unwrap();
        fs::write(nested.join("checkpoint-000025.ron"), "invalid PPO").unwrap();
        fs::write(nested.join("generation-25.ron"), "GA checkpoint").unwrap();
        fs::write(nested.join("checkpoint-000025.tmp"), "partial").unwrap();
        let mut browser = PpoCheckpointBrowser {
            roots: normalize_roots(vec![directory.clone(), nested]),
            ..Default::default()
        };
        browser.refresh();
        assert_eq!(browser.entries.len(), 1);
        assert!(browser.entries[0].summary.is_err());
        assert!(browser.status.contains("1 invalid"));
        // Delete only the exact test-owned files, never recursively remove a computed directory.
        fs::remove_file(directory.join("run/subrun/checkpoint-000025.ron")).unwrap();
        fs::remove_file(directory.join("run/subrun/generation-25.ron")).unwrap();
        fs::remove_file(directory.join("run/subrun/checkpoint-000025.tmp")).unwrap();
        fs::remove_dir(directory.join("run/subrun")).unwrap();
        fs::remove_dir(directory.join("run")).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
