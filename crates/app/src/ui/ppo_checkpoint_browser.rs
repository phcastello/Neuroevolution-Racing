//! Shared, asynchronous catalog available before loading the first PPO policy.
use bevy_egui::egui;
use racing_rl::{
    config::Config,
    persistence::{self, Checkpoint},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
};

#[derive(Debug)]
struct PolicySummary {
    rollout: u64,
    version: Option<u64>,
    architecture: String,
    parameters: usize,
    trained: bool,
    metadata_only: bool,
}
#[derive(Debug)]
struct PolicyEntry {
    path: PathBuf,
    summary: Option<Result<PolicySummary, String>>,
}
enum ScanMessage {
    Paths(Vec<PathBuf>),
    Summary(PathBuf, Result<PolicySummary, String>),
    Done(Vec<String>),
}
struct Scan {
    receiver: Mutex<mpsc::Receiver<ScanMessage>>,
    cancel: Arc<AtomicBool>,
}
impl Drop for Scan {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}
#[derive(Default)]
pub(crate) struct PpoCheckpointBrowser {
    source: Option<(PathBuf, String)>,
    roots: Vec<PathBuf>,
    entries: Vec<PolicyEntry>,
    directory_input: String,
    status: String,
    filter: String,
    scan: Option<Scan>,
    started: bool,
    #[cfg(test)]
    pub(crate) row_rects: BTreeMap<PathBuf, egui::Rect>,
}
impl PpoCheckpointBrowser {
    #[cfg(test)]
    pub(crate) fn from_roots(roots: Vec<PathBuf>) -> Self {
        Self {
            roots,
            ..Default::default()
        }
    }
    fn ensure_discovered(&mut self) {
        if !self.started {
            if self.roots.is_empty() {
                self.roots = normalize_roots(vec![
                    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results_ppo"),
                    PathBuf::from("results_ppo"),
                ]);
            }
            self.refresh();
        }
        self.poll();
    }
    pub(super) fn sync_source(&mut self, checkpoint: &Path, results_root: &str) {
        let source = (
            checkpoint.parent().unwrap_or(Path::new(".")).to_path_buf(),
            results_root.to_owned(),
        );
        if self.source.as_ref() == Some(&source) {
            self.ensure_discovered();
            return;
        }
        self.source = Some(source);
        let roots = normalize_roots(
            self.roots
                .iter()
                .cloned()
                .chain([
                    PathBuf::from(results_root),
                    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results_ppo"),
                    checkpoint.parent().unwrap_or(Path::new(".")).to_path_buf(),
                ])
                .collect(),
        );
        if self.roots != roots {
            self.roots = roots;
            self.refresh();
        }
        self.ensure_discovered();
    }
    fn refresh(&mut self) {
        self.scan = None;
        self.started = true;
        self.status = "Buscando checkpoints...".into();
        let roots = self.roots.clone();
        let (sender, receiver) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        match std::thread::Builder::new()
            .name("ppo-catalog".into())
            .spawn(move || {
                let mut paths = BTreeSet::new();
                let mut visited = BTreeSet::new();
                let mut failures = vec![];
                for root in roots {
                    if worker_cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    collect_paths(&root, &mut paths, &mut visited, &mut failures);
                }
                let paths: Vec<_> = paths.into_iter().collect();
                if sender.send(ScanMessage::Paths(paths.clone())).is_err() {
                    return;
                }
                for path in paths {
                    if worker_cancel.load(Ordering::Relaxed) {
                        return;
                    }
                    let summary = load_summary(&path);
                    if sender.send(ScanMessage::Summary(path, summary)).is_err() {
                        return;
                    }
                }
                let _ = sender.send(ScanMessage::Done(failures));
            }) {
            Ok(_) => {
                self.scan = Some(Scan {
                    receiver: Mutex::new(receiver),
                    cancel,
                })
            }
            Err(e) => self.status = format!("Falha ao buscar checkpoints: {e}"),
        }
    }
    fn poll(&mut self) {
        let messages = self.scan.as_ref().map_or_else(Vec::new, |s| {
            s.receiver.lock().unwrap().try_iter().collect()
        });
        if messages.is_empty() {
            return;
        }
        for message in messages {
            match message {
                ScanMessage::Paths(paths) => {
                    self.entries = paths
                        .into_iter()
                        .map(|path| PolicyEntry {
                            path,
                            summary: None,
                        })
                        .collect()
                }
                ScanMessage::Summary(path, summary) => {
                    if let Some(e) = self.entries.iter_mut().find(|e| e.path == path) {
                        e.summary = Some(summary);
                    }
                }
                ScanMessage::Done(failures) => {
                    self.scan = None;
                    let invalid = self
                        .entries
                        .iter()
                        .filter(|e| matches!(e.summary, Some(Err(_))))
                        .count();
                    self.status = format!(
                        "{} checkpoints encontrados; {invalid} com erro",
                        self.entries.len()
                    );
                    if let Some(e) = failures.first() {
                        self.status.push_str(&format!(". {e}"));
                    }
                }
            }
        }
        self.entries.sort_by(|a, b| {
            a.path
                .parent()
                .cmp(&b.path.parent())
                .then_with(|| rollout(a).cmp(&rollout(b)))
                .then_with(|| a.path.cmp(&b.path))
        });
    }
    fn neighbor(&self, loaded: &Path, direction: isize) -> Option<PathBuf> {
        let loaded = fs::canonicalize(loaded).unwrap_or_else(|_| loaded.to_path_buf());
        let run: Vec<_> = self
            .entries
            .iter()
            .filter(|e| e.path.parent() == loaded.parent() && !matches!(e.summary, Some(Err(_))))
            .collect();
        let index = run.iter().position(|e| e.path == loaded)?;
        index
            .checked_add_signed(direction)
            .and_then(|i| run.get(i))
            .map(|e| e.path.clone())
    }
}
fn normalize_roots(roots: Vec<PathBuf>) -> Vec<PathBuf> {
    roots
        .into_iter()
        .map(|r| fs::canonicalize(&r).unwrap_or(r))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
fn rollout(entry: &PolicyEntry) -> u64 {
    entry
        .summary
        .as_ref()
        .and_then(|s| s.as_ref().ok())
        .map_or_else(
            || {
                entry
                    .path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .and_then(|s| s.strip_prefix("checkpoint-"))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0)
            },
            |s| s.rollout,
        )
}
fn collect_paths(
    directory: &Path,
    paths: &mut BTreeSet<PathBuf>,
    visited: &mut BTreeSet<PathBuf>,
    failures: &mut Vec<String>,
) {
    if fs::symlink_metadata(directory).is_ok_and(|m| m.file_type().is_symlink()) {
        return;
    }
    let key = fs::canonicalize(directory).unwrap_or_else(|_| directory.to_path_buf());
    if !visited.insert(key) || !directory.exists() {
        return;
    }
    let entries = match fs::read_dir(directory) {
        Ok(e) => e,
        Err(e) => {
            failures.push(format!("{}: {e}", directory.display()));
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                failures.push(e.to_string());
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
            && path.extension().is_some_and(|e| e == "ron")
            && path
                .file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.starts_with("checkpoint-"))
        {
            paths.insert(fs::canonicalize(&path).unwrap_or(path));
        }
    }
}
fn summary(
    config: &Config,
    rollout: u64,
    version: Option<u64>,
    trained: bool,
    metadata_only: bool,
) -> PolicySummary {
    let trunk = config
        .hidden
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(" -> ");
    PolicySummary {
        rollout,
        version,
        architecture: format!(
            "{} -> {trunk} -> Actor 2 / Critic 1",
            config.observation.dimension()
        ),
        parameters: config.parameter_count(),
        trained,
        metadata_only,
    }
}
/// Pretty RON stores config in the header and counters in the footer. Read only
/// these small blocks to avoid deserializing gigabytes of weights on discovery.
/// Full checkpoint validation still happens on selection, before policy replacement.
fn fast_summary(path: &Path) -> Result<PolicySummary, String> {
    const BLOCK: usize = 65536;
    let mut file = fs::File::open(path).map_err(|e| e.to_string())?;
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    let mut head = vec![0; BLOCK.min(size as usize)];
    file.read_exact(&mut head).map_err(|e| e.to_string())?;
    file.seek(SeekFrom::Start(size.saturating_sub(BLOCK as u64)))
        .map_err(|e| e.to_string())?;
    let mut tail = vec![];
    file.read_to_end(&mut tail).map_err(|e| e.to_string())?;
    let head = String::from_utf8_lossy(&head);
    let tail = String::from_utf8_lossy(&tail);
    fn field<'a>(text: &'a str, name: &str) -> Result<&'a str, String> {
        text.lines()
            .find_map(|line| line.strip_prefix(&format!("    {name}: ")))
            .map(|v| v.trim().trim_end_matches(','))
            .ok_or_else(|| format!("missing {name}"))
    }
    if field(&head, "format")? != "\"racing-ppo\"" || field(&head, "version")? != "1" {
        return Err("PPO checkpoint format".into());
    }
    let begin = head.find("    config: ").ok_or("missing config")? + 12;
    let end = head[begin..]
        .find("\n    features: ")
        .ok_or("missing features")?
        + begin;
    let config: Config =
        ron::from_str(head[begin..end].trim().trim_end_matches(',')).map_err(|e| e.to_string())?;
    config.validate().map_err(|e| e.to_string())?;
    Ok(summary(
        &config,
        ron::from_str(field(&tail, "completed_rollouts")?).map_err(|e| e.to_string())?,
        field(&tail, "policy_version")
            .ok()
            .and_then(|s| ron::from_str(s).ok()),
        ron::from_str(field(&tail, "trained")?).map_err(|e| e.to_string())?,
        true,
    ))
}
fn load_summary(path: &Path) -> Result<PolicySummary, String> {
    if fs::metadata(path).is_ok_and(|m| m.len() > 131072) {
        if let Ok(summary) = fast_summary(path) {
            return Ok(summary);
        }
    }
    // Compatibility for compact/historical RON and unusual field ordering.
    let saved: Checkpoint = persistence::load(path).map_err(|e| e.to_string())?;
    saved.validate().map_err(|e| e.to_string())?;
    Ok(summary(
        &saved.config,
        saved.completed_rollouts,
        Some(saved.policy_version),
        saved.trained,
        false,
    ))
}
fn draw_catalog(
    ui: &mut egui::Ui,
    browser: &mut PpoCheckpointBrowser,
    selected: &Path,
) -> Option<PathBuf> {
    browser.ensure_discovered();
    #[cfg(test)]
    browser.row_rects.clear();
    ui.heading("Checkpoints PPO disponíveis");
    ui.horizontal(|ui| {
        ui.add(
            egui::TextEdit::singleline(&mut browser.filter)
                .hint_text("Filtrar treino / checkpoint")
                .desired_width((ui.available_width() - 80.).max(80.)),
        );
        if ui.button("Atualizar").clicked() {
            browser.refresh();
        }
    });
    ui.small(&browser.status);
    if browser.scan.is_some() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.small("Lendo catálogo em segundo plano...");
        });
        ui.ctx()
            .request_repaint_after(std::time::Duration::from_millis(30));
    }
    let filter = browser.filter.to_lowercase();
    let mut groups: BTreeMap<PathBuf, Vec<&PolicyEntry>> = BTreeMap::new();
    for e in &browser.entries {
        if e.path.to_string_lossy().to_lowercase().contains(&filter) {
            groups
                .entry(e.path.parent().unwrap_or(Path::new(".")).to_owned())
                .or_default()
                .push(e);
        }
    }
    let mut groups: Vec<_> = groups.into_iter().collect();
    groups.sort_by_key(|(_, entries)| {
        std::cmp::Reverse(entries.iter().map(|e| rollout(e)).max().unwrap_or(0))
    });
    let mut request = None;
    let mut draw_row = |ui: &mut egui::Ui, e: &PolicyEntry| {
        let name = e.path.file_name().unwrap_or_default().to_string_lossy();
        let label = match &e.summary {
            Some(Ok(s)) => format!("{name} · rollout {}", s.rollout),
            Some(Err(_)) => format!("{name} [erro]"),
            None => name.to_string(),
        };
        let response = ui.add_enabled(
            !matches!(e.summary, Some(Err(_))),
            egui::Button::selectable(e.path == selected, label),
        );
        #[cfg(test)]
        browser.row_rects.insert(e.path.clone(), response.rect);
        if response.clicked() {
            request = Some(e.path.clone());
        }
        let tooltip = match &e.summary {
            Some(Ok(s)) => format!(
                "{}\n{} parâmetros · versão {}\n{}\n{}",
                e.path.display(),
                s.parameters,
                s.version.map_or_else(|| "-".into(), |v| v.to_string()),
                s.architecture,
                if s.metadata_only {
                    "Prévia; validação completa ao carregar"
                } else if s.trained {
                    "PPO treinado"
                } else {
                    "Política sem treinamento"
                }
            ),
            Some(Err(e)) => e.clone(),
            None => e.path.display().to_string(),
        };
        response.on_hover_text(tooltip);
    };
    egui::ScrollArea::vertical()
        .id_salt("ppo_catalog")
        .max_height(300.)
        .show(ui, |ui| {
            for (folder, entries) in groups {
                ui.separator();
                let relative = folder
                    .strip_prefix(
                        Path::new(env!("CARGO_MANIFEST_DIR"))
                            .join("../..")
                            .canonicalize()
                            .unwrap_or_default(),
                    )
                    .unwrap_or(&folder);
                ui.strong(relative.display().to_string());
                if let Some(latest) = entries.last() {
                    draw_row(ui, latest);
                }
                if entries.len() > 1 {
                    egui::CollapsingHeader::new(format!(
                        "Outros checkpoints ({})",
                        entries.len() - 1
                    ))
                    .id_salt(&folder)
                    .show(ui, |ui| {
                        for e in entries[..entries.len() - 1].iter().rev() {
                            draw_row(ui, e);
                        }
                    });
                }
            }
        });
    if browser.entries.is_empty() && browser.scan.is_none() {
        ui.label("Nenhum checkpoint encontrado em results_ppo.");
    }
    ui.collapsing("Adicionar outra pasta (opcional)", |ui| {
        ui.text_edit_singleline(&mut browser.directory_input);
        if ui.button("Incluir pasta").clicked() {
            let path = PathBuf::from(browser.directory_input.trim());
            if path.is_dir() {
                browser.roots =
                    normalize_roots(browser.roots.iter().cloned().chain([path]).collect());
                browser.refresh();
            } else {
                browser.status = "Pasta não encontrada".into();
            }
        }
    });
    request
}
pub(crate) fn draw_policy_picker(
    ui: &mut egui::Ui,
    browser: &mut PpoCheckpointBrowser,
    selected: &mut String,
) -> bool {
    if let Some(path) = draw_catalog(ui, browser, Path::new(selected)) {
        *selected = path.display().to_string();
        true
    } else {
        false
    }
}
pub(super) fn draw_policy_history(
    ui: &mut egui::Ui,
    browser: &mut PpoCheckpointBrowser,
    loaded: &Path,
    request: &mut Option<PathBuf>,
) {
    ui.horizontal(|ui| {
        for (label, direction) in [("< Anterior", -1), ("Próximo >", 1)] {
            let neighbor = browser.neighbor(loaded, direction);
            if ui
                .add_enabled(neighbor.is_some(), egui::Button::new(label))
                .clicked()
            {
                *request = neighbor;
            }
        }
    });
    if let Some(path) = draw_catalog(ui, browser, loaded) {
        *request = Some(path);
    }
}
#[cfg(test)]
#[path = "ppo_checkpoint_render_test.rs"]
mod render_test;
#[cfg(test)]
#[path = "ppo_checkpoint_browser_tests.rs"]
pub(crate) mod test_support;
