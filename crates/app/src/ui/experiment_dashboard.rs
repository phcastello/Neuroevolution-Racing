//! Bounded read-only browsing of existing run artifacts and their shared CSV metrics.
use crate::visual::Shell;
use bevy::prelude::Resource;
use bevy_egui::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug)]
struct Run {
    directory: PathBuf,
    files: Vec<PathBuf>,
    compare: bool,
}
#[derive(Default)]
struct Csv {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}
#[derive(Resource)]
pub(crate) struct ExperimentDashboard {
    root: String,
    runs: Vec<Run>,
    selected: usize,
    file: Option<PathBuf>,
    text: String,
    csv: Csv,
    metric: usize,
    status: String,
    pub pending_track: Option<crate::simulation::TrackDefinition>,
    compare_cache: BTreeMap<PathBuf, Csv>,
    higher_is_better: bool,
}
impl Default for ExperimentDashboard {
    fn default() -> Self {
        Self {
            root: "results_ppo".into(),
            runs: Vec::new(),
            selected: 0,
            file: None,
            text: String::new(),
            csv: Csv::default(),
            metric: 1,
            status: String::new(),
            pending_track: None,
            compare_cache: BTreeMap::new(),
            higher_is_better: true,
        }
    }
}
impl ExperimentDashboard {
    pub(crate) fn show(&mut self, ctx: &egui::Context, shell: &mut Shell) {
        super::draw_dashboard(ctx, "Experimentos · artefatos persistidos", |ui| {
            ui.heading("Experimentos");
            ui.label("Diretório de resultados / planos");
            ui.text_edit_singleline(&mut self.root);
            if ui.button("Consultar diretório").clicked() {
                match discover(Path::new(&self.root)) {
                    Ok(runs) => {
                        self.runs = runs;
                        self.selected = 0;
                        self.status = format!("{} execuções / diretórios", self.runs.len());
                        self.file = None;
                        self.text.clear();
                        self.csv = Csv::default();
                        self.compare_cache.clear();
                    }
                    Err(error) => self.status = error,
                }
            }
            ui.small(&self.status);
            ui.small("Selecione execuções para comparar a mesma coluna CSV. Não há vencedores sem dados de avaliação.");
            for (index, run) in self.runs.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut run.compare, "");
                    if ui
                        .selectable_label(
                            self.selected == index,
                            run.directory
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy(),
                        )
                        .clicked()
                    {
                        self.selected = index;
                        self.file = None;
                    }
                });
            }
            if let Some(run) = self.runs.get(self.selected) {
                ui.separator();
                ui.strong("Artefatos");
                let mut requested = None;
                for file in &run.files {
                    if ui
                        .selectable_label(
                            self.file.as_ref() == Some(file),
                            file.file_name().unwrap_or_default().to_string_lossy(),
                        )
                        .clicked()
                    {
                        requested = Some(file.clone());
                    }
                }
                if let Some(file) = requested {
                    if is_checkpoint(&file) {
                        shell.policy_path = file.display().to_string();
                        shell.load_policy = true;
                    } else {
                        match read_bounded(&file) {
                            Ok(text) => {
                                self.csv = if file.extension().is_some_and(|e| e == "csv") {
                                    parse_csv(&text)
                                } else {
                                    Csv::default()
                                };
                                self.text = text;
                                self.metric = 1;
                                self.file = Some(file);
                            }
                            Err(error) => self.status = error,
                        }
                    }
                }
            }
            if ron::from_str::<crate::simulation::procedural::GeneratedTrack>(&self.text).is_ok() {
                if ui.button("Abrir pista gerada").clicked() {
                    if let Some(path) = &self.file {
                        match crate::simulation::procedural::GeneratedTrack::load(path) {
                            Ok(generated) => self.pending_track = Some(generated.definition),
                            Err(error) => self.status = error,
                        }
                    }
                }
            } else if let Ok(definition) =
                ron::from_str::<crate::simulation::TrackDefinition>(&self.text)
            {
                if ui.button("Abrir pista").clicked() {
                    self.pending_track = Some(definition);
                }
            }
            if !self.csv.headers.is_empty() {
                self.metric = self.metric.min(self.csv.headers.len() - 1);
                egui::ComboBox::from_label("Métrica")
                    .selected_text(&self.csv.headers[self.metric])
                    .show_ui(ui, |ui| {
                        for (index, name) in self.csv.headers.iter().enumerate() {
                            ui.selectable_value(&mut self.metric, index, name);
                        }
                    });
                ui.label(format!("{} registros", self.csv.rows.len()));
                if let Some(last) = self.csv.rows.last() {
                    ui.monospace(format!(
                        "Último {}: {}",
                        self.csv.headers[self.metric],
                        last.get(self.metric).map_or("—", String::as_str)
                    ));
                }
            }
            if !shell.status.is_empty() {
                ui.colored_label(egui::Color32::LIGHT_RED, &shell.status);
            }
        });
        if let Some(filename) = self.file.as_ref().and_then(|p| p.file_name()) {
            for run in self.runs.iter().filter(|r| r.compare) {
                let path = run.directory.join(filename);
                self.compare_cache.entry(path.clone()).or_insert_with(|| {
                    read_bounded(&path)
                        .map(|text| parse_csv(&text))
                        .unwrap_or_default()
                });
            }
        }
        egui::Window::new("Configuração efetiva / métricas")
            .default_pos([18.0, 58.0])
            .default_width(760.0)
            .default_height(640.0)
            .show(ctx, |ui| {
                if self.file.is_none() {
                    ui.label("Abra um diretório e selecione um artefato. Checkpoints abrem o modo PPO.");
                    return;
                }
                if !self.csv.headers.is_empty() {
                    let metric_name = &self.csv.headers[self.metric];
                    Plot::new("experiment_metric").height(260.0).legend(Legend::default())
                        .show(ui, |plot| {
                            let name = self.runs.get(self.selected)
                                .map_or("selecionado".into(), |r| r.directory.display().to_string());
                            plot.line(Line::new(name, points(&self.csv, metric_name)));
                            if let Some(filename) = self.file.as_ref().and_then(|p|p.file_name()) {
                                for (index, run) in self.runs.iter().enumerate()
                                    .filter(|(i,r)| *i != self.selected && r.compare) {
                                    if let Some(csv) = self.compare_cache.get(&run.directory.join(filename)) {
                                        let name = format!("{index}: {}",run.directory.display());
                                        plot.line(Line::new(name, points(csv, metric_name)));
                                    }
                                }
                            }
                        });
                    ui.checkbox(&mut self.higher_is_better, "Ordenar: maior valor é melhor");
                    ui.label("Ranking do último valor finito na coluna selecionada");
                    ui.small("Selecione uma métrica de avaliação desconhecida para comparar generalização.");
                    let mut ranked = Vec::new();
                    if let Some(value) = last_value(&self.csv, metric_name) {
                        ranked.push((self.selected,value));
                    }
                    if let Some(filename) = self.file.as_ref().and_then(|p|p.file_name()) {
                        for (index,run) in self.runs.iter().enumerate()
                            .filter(|(i,r)| *i != self.selected && r.compare) {
                            if let Some(value) = self.compare_cache.get(&run.directory.join(filename))
                                .and_then(|csv| last_value(csv, metric_name)) {
                                ranked.push((index,value));
                            }
                        }
                    }
                    ranked.sort_by(|a,b| if self.higher_is_better {
                        b.1.total_cmp(&a.1)
                    } else { a.1.total_cmp(&b.1) });
                    for (rank,(index,value)) in ranked.iter().enumerate() {
                        if let Some(run) = self.runs.get(*index) {
                            ui.monospace(format!("{}. {}: {:.6}",rank+1,run.directory.display(),value));
                        }
                    }
                }
                egui::ScrollArea::both().show(ui, |ui| {
                    ui.add(egui::TextEdit::multiline(&mut self.text)
                        .font(egui::TextStyle::Monospace).desired_width(f32::INFINITY)
                        .interactive(false));
                });
            });
    }
}
fn last_value(csv: &Csv, name: &str) -> Option<f64> {
    let column = csv.headers.iter().position(|h| h == name)?;
    csv.rows
        .iter()
        .rev()
        .filter_map(|row| row.get(column)?.parse::<f64>().ok())
        .find(|value| value.is_finite())
}
fn points(csv: &Csv, name: &str) -> PlotPoints<'static> {
    let Some(column) = csv.headers.iter().position(|h| h == name) else {
        return PlotPoints::from(Vec::<[f64; 2]>::new());
    };
    PlotPoints::from_iter(csv.rows.iter().enumerate().filter_map(|(i, row)| {
        let x = row
            .first()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(i as f64);
        let y = row.get(column)?.parse::<f64>().ok()?;
        (x.is_finite() && y.is_finite()).then_some([x, y])
    }))
}
fn read_bounded(path: &Path) -> Result<String, String> {
    if fs::metadata(path).map_err(|e| e.to_string())?.len() > 16 * 1024 * 1024 {
        return Err(
            "Artefato excede limite visual de 16 MiB; consulte-o pelas ferramentas headless."
                .into(),
        );
    }
    fs::read_to_string(path).map_err(|e| e.to_string())
}
fn is_checkpoint(path: &Path) -> bool {
    path.file_name().is_some_and(|n| {
        let n = n.to_string_lossy();
        (n.starts_with("checkpoint_") || n.starts_with("checkpoint-")) && n.ends_with(".ron")
    })
}
fn discover(root: &Path) -> Result<Vec<Run>, String> {
    let mut directories = BTreeSet::new();
    let mut pending = vec![(root.to_owned(), 0)];
    let mut visited = 0;
    while let Some((directory, depth)) = pending.pop() {
        visited += 1;
        if visited > 2048 {
            return Err("Mais de 2048 diretórios; selecione uma raiz menor.".into());
        }
        let mut found = false;
        for entry in
            fs::read_dir(&directory).map_err(|e| format!("{}: {e}", directory.display()))?
        {
            let entry = entry.map_err(|e| e.to_string())?;
            let kind = entry.file_type().map_err(|e| e.to_string())?;
            if kind.is_symlink() {
                continue;
            }
            if kind.is_dir() && depth < 3 {
                pending.push((entry.path(), depth + 1));
            } else if kind.is_file() && artifact(&entry.path()) {
                found = true;
            }
        }
        if found {
            directories.insert(directory);
        }
    }
    directories
        .into_iter()
        .map(|directory| {
            let mut files = fs::read_dir(&directory)
                .map_err(|e| e.to_string())?
                .filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.is_file() && artifact(p))
                .collect::<Vec<_>>();
            files.sort();
            Ok(Run {
                directory,
                files,
                compare: false,
            })
        })
        .collect()
}
fn artifact(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| matches!(e.to_str(), Some("ron" | "json" | "csv" | "md" | "txt")))
}
/// The telemetry uses RFC-style quoted cells (track IDs and optional columns).
fn parse_csv(text: &str) -> Csv {
    let mut records = Vec::new();
    let mut row = Vec::new();
    let mut cell = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if quoted && chars.peek() == Some(&'"') => {
                chars.next();
                cell.push('"');
            }
            '"' => quoted = !quoted,
            ',' if !quoted => row.push(std::mem::take(&mut cell)),
            '\n' if !quoted => {
                row.push(std::mem::take(&mut cell));
                records.push(std::mem::take(&mut row));
            }
            '\r' if !quoted => {}
            _ => cell.push(c),
        }
    }
    if !cell.is_empty() || !row.is_empty() {
        row.push(cell);
        records.push(row);
    }
    if records.is_empty() {
        Csv::default()
    } else {
        let headers = records.remove(0);
        Csv {
            headers,
            rows: records,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn csv_preserves_quoted_cells_empty_fields_and_finite_plot_values() {
        let csv = parse_csv("rollout,track,progress\r\n1,\"curve, left\",0.7\n2,\"a\"\"b\",\n");
        assert_eq!(csv.rows[0][1], "curve, left");
        assert_eq!(csv.rows[1][1], "a\"b");
        assert_eq!(csv.rows[1][2], "");
        assert_eq!(csv.headers.len(), 3);
    }
}
