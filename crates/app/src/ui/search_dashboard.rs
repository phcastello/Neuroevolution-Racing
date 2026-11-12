//! Live execution and archive replay consume the same ACO engine.
use crate::{
    aco::{
        AcoConfig, AntSnapshot, SearchArchive,
        live::{LiveSearch, RunMode, SearchEvent},
    },
    simulation::{Track, TrackDefinition, procedural},
    visual::Shell,
};
use bevy::prelude::Resource;
use bevy_egui::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::{path::PathBuf, time::Instant};
mod inspection_ui;
mod ppo_playback;
pub(crate) mod viewport;
use viewport::AcoView;

#[derive(Resource)]
pub(crate) struct SearchDashboard {
    config: AcoConfig,
    seed: u64,
    checkpoint: String,
    use_policy: bool,
    speed: f32,
    live: Option<LiveSearch>,
    archive: Option<SearchArchive>,
    current: Option<AntSnapshot>,
    current_ants: Vec<AntSnapshot>,
    definition: Option<TrackDefinition>,
    geometry_error: Option<String>,
    pheromones: Vec<Vec<f64>>,
    iteration: usize,
    ant: usize,
    decision: usize,
    replay: bool,
    playing: bool,
    last_advance: Instant,
    tab: usize,
    path: String,
    export_path: String,
    status: String,
    pub pending_track: Option<TrackDefinition>,
    inspected: Option<AntSnapshot>,
    inspected_key: Option<(usize, usize)>,
    inspected_decision: usize,
    candidate_filter: usize,
    reason_filter: String,
    comparison: Option<AntSnapshot>,
    view: AcoView,
    generation_selection: Option<usize>,
    show_generation_tracks: bool,
    ppo_test_track: Option<procedural::GeneratedTrack>,
    ppo_checkpoint: String,
    fit_revision: u64,
    #[cfg(test)]
    buttons: std::collections::BTreeMap<&'static str, egui::Rect>,
    #[cfg(test)]
    candidate_buttons: std::collections::BTreeMap<(usize, usize), egui::Rect>,
}
impl Default for SearchDashboard {
    fn default() -> Self {
        Self {
            config: AcoConfig::default(),
            seed: 47,
            checkpoint: String::new(),
            use_policy: false,
            speed: 5.,
            live: None,
            archive: None,
            current: None,
            current_ants: vec![],
            definition: None,
            geometry_error: None,
            pheromones: vec![],
            iteration: 0,
            ant: 0,
            decision: 0,
            replay: false,
            playing: false,
            last_advance: Instant::now(),
            tab: 0,
            path: "results_aco/search.ron".into(),
            export_path: "results_aco/tracks".into(),
            status: "Configure e clique em Iniciar.".into(),
            pending_track: None,
            inspected: None,
            inspected_key: None,
            inspected_decision: 0,
            candidate_filter: 0,
            reason_filter: String::new(),
            comparison: None,
            view: AcoView::Generation,
            generation_selection: None,
            show_generation_tracks: false,
            ppo_test_track: None,
            ppo_checkpoint: String::new(),
            fit_revision: 0,
            #[cfg(test)]
            buttons: Default::default(),
            #[cfg(test)]
            candidate_buttons: Default::default(),
        }
    }
}
impl SearchDashboard {
    fn save_archive(&self) -> Result<(), String> {
        let mut archive = if let Some(a) = &self.archive {
            a.clone()
        } else {
            SearchArchive::new(self.config.clone(), self.seed)?
        };
        if !archive
            .iterations
            .iter()
            .any(|i| i.iteration == self.iteration)
        {
            archive.in_progress = self.current_ants.clone();
            if let Some(current) = &self.current {
                if !archive.in_progress.iter().any(|a| a.ant == current.ant) {
                    let mut partial = current.clone();
                    if partial.candidate.is_none() && partial.rejection.is_none() {
                        let reason = "Construção parcial salva; geometria ainda não validada";
                        partial.rejection = Some(reason.into());
                        partial.geometry_valid = Some(false);
                        partial.diagnostic =
                            Some(crate::aco::inspection::GeometryDiagnostic::capture(
                                None,
                                &partial.decisions,
                                None,
                                reason,
                            ));
                    }
                    archive.in_progress.push(partial);
                }
            }
        }
        archive.apply_retention();
        let path = PathBuf::from(&self.path);
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        archive.save(path)
    }
    pub(crate) fn suspend(&self) {
        if let Some(live) = &self.live {
            live.control.set_mode(RunMode::Paused);
        }
    }
    fn button(&mut self, ui: &mut egui::Ui, label: &'static str) -> bool {
        let response = ui.button(label);
        #[cfg(test)]
        self.buttons.insert(label, response.rect);
        response.clicked()
    }
    fn start(&mut self) {
        if let Some(live) = self.live.take() {
            live.control.cancel();
        }
        self.config.max_candidates = self.config.ants.saturating_mul(self.config.iterations);
        self.config.policy_label = None;
        let checkpoint = self
            .use_policy
            .then(|| PathBuf::from(self.checkpoint.trim()));
        match LiveSearch::start(
            self.config.clone(),
            self.seed,
            checkpoint,
            RunMode::Running,
            self.speed,
        ) {
            Ok(live) => {
                self.live = Some(live);
                self.archive = None;
                self.current = None;
                self.current_ants.clear();
                self.inspected = None;
                self.inspected_key = None;
                self.comparison = None;
                self.ppo_test_track = None;
                self.view = AcoView::Generation;
                self.generation_selection = None;
                self.definition = None;
                self.geometry_error = None;
                self.pheromones = self
                    .config
                    .generator
                    .choice_counts()
                    .iter()
                    .map(|n| vec![self.config.initial_pheromone; *n])
                    .collect();
                self.iteration = 0;
                self.ant = 0;
                self.decision = 0;
                self.replay = false;
                self.playing = false;
                self.status = "Executando ACO real em tarefa separada.".into();
            }
            Err(e) => self.status = e,
        }
    }
    pub(crate) fn poll(&mut self) {
        for event in self.live.as_ref().map_or_else(Vec::new, LiveSearch::poll) {
            self.receive_event(event);
        }
    }
    fn receive_event(&mut self, event: SearchEvent) {
        match event {
            SearchEvent::AntStarted {
                iteration,
                ant,
                pheromones,
            } => {
                if iteration != self.iteration {
                    self.current_ants.clear();
                }
                self.iteration = iteration;
                self.ant = ant;
                self.decision = 0;
                self.pheromones = pheromones;
                self.current = Some(AntSnapshot {
                    ant,
                    decisions: vec![],
                    candidate: None,
                    rejection: None,
                    fitness: None,
                    rejection_reason: None,
                    evaluation: None,
                    geometry_valid: None,
                    diagnostic: None,
                    retention_evicted: false,
                    candidate_id: None,
                });
                if self.inspected.is_none() {
                    self.definition = None;
                    self.geometry_error = None;
                }
            }
            SearchEvent::Decision(d) => {
                if let Some(c) = &mut self.current {
                    c.decisions.push(d);
                }
                self.decision += 1;
            }
            SearchEvent::Geometry { definition, error } => {
                if self.inspected.is_none() {
                    self.definition = Some(definition);
                    self.geometry_error = error;
                }
            }
            SearchEvent::Candidate(c) => {
                self.current = Some(c.clone());
                self.current_ants.push(c);
            }
            SearchEvent::Iteration { archive } => {
                self.pheromones = archive.state.pheromones.clone();
                self.archive = Some(*archive);
            }
            SearchEvent::Finished { archive } => {
                self.archive = Some(*archive);
                self.status = "Busca concluída. Salvar e exportar disponíveis.".into();
            }
            SearchEvent::Failed(e) => self.status = e,
        }
    }
    fn replay_selection(&mut self) {
        self.view = AcoView::Generation;
        self.inspected = None;
        self.inspected_key = None;
        self.generation_selection = None;
        self.current = self
            .archive
            .as_ref()
            .and_then(|a| a.iterations.get(self.iteration))
            .and_then(|i| i.ants.get(self.ant))
            .cloned();
        self.definition = None;
        self.geometry_error = None;
        if let Some(a) = &self.archive {
            if let Some(i) = a.iterations.get(self.iteration) {
                self.pheromones = i.pheromones_before.clone();
                self.current_ants = i.ants.clone();
            }
            if let Some(c) = &self.current {
                self.decision = self.decision.min(c.decisions.len());
                if self.decision == c.decisions.len() {
                    let choices: Vec<_> = c.decisions.iter().map(|d| d.choice).collect();
                    self.definition =
                        procedural::definition_from_decisions(&a.config.generator, &choices).ok();
                    self.geometry_error =
                        procedural::from_decisions(&a.config.generator, a.state.seed, 0, &choices)
                            .err();
                    self.pheromones = a.iterations[self.iteration].pheromones_after.clone();
                }
            }
        }
    }
    pub(crate) fn show(&mut self, ctx: &egui::Context, shell: &mut Shell) {
        self.poll();
        if self.replay
            && self.playing
            && self.last_advance.elapsed().as_secs_f32() >= 1. / self.speed
        {
            if let Some((a, i)) = self
                .archive
                .as_ref()
                .and_then(|a| a.iterations.get(self.iteration).map(|i| (a, i)))
            {
                if self.decision < self.current.as_ref().map_or(0, |c| c.decisions.len()) {
                    self.decision += 1;
                } else if self.ant + 1 < i.ants.len() {
                    self.ant += 1;
                    self.decision = 0;
                } else if self.iteration + 1 < a.iterations.len() {
                    self.iteration += 1;
                    self.ant = 0;
                    self.decision = 0;
                } else {
                    self.playing = false;
                }
                self.replay_selection();
            } else {
                self.playing = false;
            }
            self.last_advance = Instant::now();
        }
        let best_available = self.has_best();
        let mut requested = None;
        let mut refit = false;
        let can_test = self
            .shown_candidate()
            .is_some_and(|c| c.geometrically_valid() && c.candidate.is_some());
        let mut test_ppo = false;
        #[cfg(test)]
        let mut toolbar_buttons = vec![];
        super::draw_dashboard_with_toolbar(
            ctx,
            "ACO",
            |ui| {
                let best = ui.add_enabled(best_available, egui::Button::new("Ver melhor pista"));
                #[cfg(test)]
                toolbar_buttons.push(("Ver melhor pista", best.rect));
                if best.clicked() {
                    requested = Some(AcoView::Best);
                }
                let generation = ui.button("Ver geração");
                #[cfg(test)]
                toolbar_buttons.push(("Ver geração", generation.rect));
                if generation.clicked() {
                    requested = Some(AcoView::Generation);
                }
                if ui.button("Ajustar visão").clicked() {
                    refit = true;
                }
                let play = ui.add_enabled(can_test, egui::Button::new("Testar com PPO"));
                #[cfg(test)]
                toolbar_buttons.push(("Testar com PPO", play.rect));
                if play.clicked() {
                    test_ppo = true;
                }
            },
            |ui| {
                ui.heading("ACO");
                ui.strong(self.view_caption());
                if !shell.status.is_empty() {
                    ui.colored_label(egui::Color32::LIGHT_RED, &shell.status);
                } else {
                    ui.label(&self.status);
                }
                self.playback_controls(ui, shell);
                let active = self.live.as_ref().is_some_and(|l| !l.is_finished());
                if active {
                    let control = self.live.as_ref().unwrap().control.clone();
                    ui.horizontal(|ui| {
                        if self.button(ui, "Pausar") {
                            control.set_mode(RunMode::Paused);
                        }
                        if self.button(ui, "Continuar") {
                            control.set_mode(RunMode::Running);
                        }
                        if self.button(ui, "Interromper") {
                            control.cancel();
                            self.status = "Interrompendo busca...".into();
                        }
                    });
                    ui.horizontal(|ui| {
                        if self.button(ui, "Uma decisão") {
                            control.set_mode(RunMode::Decision);
                        }
                        if self.button(ui, "Uma iteração") {
                            control.set_mode(RunMode::Iteration);
                        }
                    });
                    ui.label(if control.is_paused() {
                        "Pausado · cálculo interrompido"
                    } else if control.mode() == RunMode::Paused {
                        "Pausando no próximo ponto seguro..."
                    } else {
                        "Motor executando"
                    });
                }
                if self.button(ui, "Reiniciar com mesma seed") {
                    self.start();
                }
                if ui
                    .add(
                        egui::Slider::new(&mut self.speed, 0.5..=120.)
                            .logarithmic(true)
                            .text("Decisões/s"),
                    )
                    .changed()
                {
                    if let Some(l) = &self.live {
                        l.control.set_speed(self.speed);
                    }
                }

                ui.horizontal_wrapped(|ui| {
                    for (i, title) in ["Busca", "Candidatas", "Decisões", "Feromônios", "Histórico"]
                        .iter()
                        .enumerate()
                    {
                        // Preserve internal tab indices used by replay and inspection.
                        let tab = [0, 2, 4, 1, 3][i];
                        ui.selectable_value(&mut self.tab, tab, *title);
                    }
                });
                ui.separator();
                match self.tab {
                    0 => {
                        self.viewport_status(ui);
                        ui.add_enabled_ui(!active, |ui| {
                egui::Grid::new("aco_config").show(ui, |ui| {
                    ui.label("Formigas");
                    ui.add(egui::DragValue::new(&mut self.config.ants).range(1..=256));
                    ui.end_row();
                    ui.label("Iterações");
                    ui.add(egui::DragValue::new(&mut self.config.iterations).range(1..=10000));
                    ui.end_row();
                    ui.label("Seed");
                    ui.add(egui::DragValue::new(&mut self.seed));
                    ui.end_row();
                });
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(self.config.generator.is_sequential(), "Sequencial v2")
                        .clicked()
                    {
                        self.config.generator = procedural::GeneratorConfig::default();
                    }
                    if ui
                        .selectable_label(!self.config.generator.is_sequential(), "Radial legado")
                        .clicked()
                    {
                        self.config.generator = procedural::GeneratorConfig::legacy();
                    }
                });
                ui.collapsing("Parâmetros principais / geometria", |ui| {
                    ui.add(egui::Slider::new(&mut self.config.alpha, 0.0..=8.).text("α"));
                    ui.add(egui::Slider::new(&mut self.config.beta, 0.0..=8.).text("β"));
                    ui.add(
                        egui::Slider::new(&mut self.config.evaporation, 0.01..=0.99)
                            .text("Evaporação"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.config.exploitation_probability, 0.0..=1.)
                            .text("q0"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.config.minimum_diversity, 0.0..=1.)
                            .text("Diversidade mínima"),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.config.stagnation_iterations)
                            .range(1..=1000)
                            .prefix("Estagnação: "),
                    );
                    let sequential = self.config.generator.is_sequential();
                    ui.add(
                        egui::DragValue::new(&mut self.config.generator.sectors)
                            .range(6..=32)
                            .prefix(if sequential {
                                "Segmentos maximos: "
                            } else {
                                "Setores: "
                            }),
                    );
                    if sequential {
                        ui.checkbox(
                            &mut self.config.generator.organic,
                            "Exploração orgânica em três fases",
                        );
                        ui.add(
                            egui::DragValue::new(&mut self.config.generator.primitive_radius)
                                .range(50.0..=250.)
                                .prefix("Raio base: "),
                        );
                        ui.label("La Source / Spa: κ ≈ 0,02094; raio mínimo ≈ 47,75");
                        ui.small("Padrão largo: 96/100/104; hairpin de 180° com R75. R50 exige largura menor.");
                        ui.horizontal(|ui| {
                            ui.label("Larguras:");
                            for width in &mut self.config.generator.widths {
                                ui.add(egui::DragValue::new(width).range(40.0..=120.));
                            }
                        });
                    } else {
                        ui.add(
                            egui::DragValue::new(&mut self.config.generator.radius)
                                .range(100.0..=2000.)
                                .prefix("Raio: "),
                        );
                    }
                    ui.add(
                        egui::DragValue::new(&mut self.config.max_evaluation_steps_per_candidate)
                            .range(2..=100000)
                            .prefix("Passos / candidata: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.config.max_evaluation_steps)
                            .range(2..=10000000)
                            .prefix("Orçamento PPO: "),
                    );
                });
                ui.checkbox(&mut self.use_policy, "Avaliar com PPO congelado");
                if self.use_policy {
                    ui.push_id("aco_policy_picker", |ui| {
                        super::ppo_checkpoint_browser::draw_policy_picker(
                            ui,
                            &mut shell.checkpoint_browser,
                            &mut self.checkpoint,
                        );
                    });
                    if !self.checkpoint.is_empty() {
                        ui.small(format!(
                            "Selecionado: {}",
                            PathBuf::from(&self.checkpoint)
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                        ));
                    }
                } else {
                    ui.small("Sem piloto: fitness geométrico + diversidade.");
                }
                if self.button(ui, "Iniciar") {
                    self.start();
                }
            });
                        ui.label(&self.status);
                        ui.separator();
                        ui.label(format!(
                            "Iteração {} · formiga {} · decisões {}",
                            self.iteration + 1,
                            self.ant + 1,
                            self.decision
                        ));
                        ui.label(format!(
                            "Candidatas nesta iteração: {} · rejeitadas: {}",
                            self.current_ants.len(),
                            self.current_ants
                                .iter()
                                .filter(|a| a.rejection.is_some())
                                .count()
                        ));
                        let best_current = self
                            .current_ants
                            .iter()
                            .filter(|c| c.rejection.is_none())
                            .filter_map(|c| c.fitness.as_ref())
                            .map(|f| f.score)
                            .max_by(f64::total_cmp);
                        ui.label(format!(
                            "Melhor atual: {}",
                            best_current.map_or_else(|| "nenhuma".into(), |v| format!("{v:.4}"))
                        ));
                        if let Some(a) = &self.archive {
                            ui.label(format!(
                                "{} iterações concluídas · {} candidatas",
                                a.state.iteration, a.state.candidates
                            ));
                            ui.label(format!(
                                "Estagnação: {} / {}",
                                a.state.iterations_without_improvement,
                                a.config.stagnation_iterations
                            ));
                            if let Some(i) = a.iterations.last() {
                                ui.label(format!(
                                    "Média {:.4} · diversidade {:.3}",
                                    i.mean_fitness, i.diversity
                                ));
                                if i.stagnation_reset {
                                    ui.colored_label(
                                        egui::Color32::YELLOW,
                                        "Feromônios reiniciados por estagnação",
                                    );
                                }
                            }
                        }
                        if let Some(c) = self.shown_candidate() {
                            if let Some(reason) = &c.rejection {
                                ui.colored_label(
                                    egui::Color32::LIGHT_RED,
                                    format!("Rejeitada: {reason}"),
                                );
                            }
                            if let Some(f) = &c.fitness {
                                ui.label(format!("Fitness: {:.4}", f.score));
                                if let Some(quality) = f.organic_quality {
                                    ui.label(format!("Qualidade orgânica: {quality:.3}"));
                                }
                                ui.monospace(format!("Geometria {:.4}\nDiversidade {:.4}\nDificuldade útil {:.4}\nConclusão {:?}\nProgresso {:?}\nColisões {:?}\nPassos {}", f.geometric_difficulty, f.diversity, f.useful_difficulty, f.completion_rate, f.mean_progress, f.collision_rate, f.simulation_steps));
                            }
                            if let Some(candidate) = &c.candidate {
                                let m = &candidate.geometry;
                                ui.small(format!(
                        "Largura {:.1} = {:.2} carros · faixa entre muros ≈ {:.1} = {:.2} carros",
                        m.width,
                        m.width / crate::simulation::CAR_WIDTH,
                        m.width - crate::simulation::WALL_THICKNESS,
                        (m.width - crate::simulation::WALL_THICKNESS)
                            / crate::simulation::CAR_WIDTH
                    ));
                                if self.config.generator.is_sequential() {
                                    ui.small(format!("L={:.0} | maior reta={:.0} | retas={:.0}% | curvas={} | hairpins={} | chicanes={} | S={} | compacidade={:.3}",m.length,m.longest_straight,m.straight_fraction*100.,m.turns,m.hairpins,m.chicanes,m.s_bends,m.compactness));
                                    if self.config.generator.organic {
                                        let o = &m.organic;
                                        ui.small(format!("Retangularidade {:.3} | eixos {:.3} | diversidade heading {:.3} | eventos {} | maior reta {:.1}%", o.rectangularity,o.axis_concentration,o.heading_entropy,o.geometric_events,100.*o.longest_straight_ratio));
                                    }
                                }
                                if self.button(ui, "Testar pista com PPO") {
                                    self.begin_selected_ppo_test(shell);
                                }
                            }
                        }
                        ui.collapsing("Salvar / exportar / reproduzir arquivo", |ui| {
                            ui.label("Arquivo de busca (.ron)");
                            ui.text_edit_singleline(&mut self.path);
                            ui.add_enabled_ui(!active, |ui| {
                                if ui.button("Abrir replay").clicked() {
                                    match SearchArchive::load(&self.path) {
                                        Ok(a) => {
                                            self.live = None;
                                            self.inspected = None;
                                            self.inspected_key = None;
                                            self.comparison = None;
                                            self.current_ants.clear();
                                            self.current = None;
                                            if !a.in_progress.is_empty() {
                                                self.tab = 2;
                                            }
                                            self.config = a.config.clone();
                                            self.seed = a.state.seed;
                                            self.archive = Some(a);
                                            self.replay = true;
                                            self.playing = false;
                                            self.iteration = 0;
                                            self.ant = 0;
                                            self.decision = 0;
                                            self.status = "Replay de arquivo".into();
                                            self.replay_selection();
                                        }
                                        Err(e) => self.status = e,
                                    }
                                }
                                if ui.button("Salvar busca").clicked() {
                                    self.status = self
                                        .save_archive()
                                        .map_or_else(|e| e, |_| "Busca salva.".into());
                                }
                                ui.label("Diretório para pistas selecionadas");
                                ui.text_edit_singleline(&mut self.export_path);
                                if ui.button("Exportar pistas").clicked() {
                                    self.status = self.archive.as_ref().map_or_else(
                                        || "Sem resultados.".into(),
                                        |a| {
                                            a.export_selected(&self.export_path).map_or_else(
                                                |e| e,
                                                |ids| format!("{} pistas exportadas.", ids.len()),
                                            )
                                        },
                                    );
                                }
                            });
                        });
                        if self.replay {
                            ui.checkbox(&mut self.playing, "Reproduzir arquivo");
                            let count = self.archive.as_ref().map_or(0, |a| a.iterations.len());
                            if count > 0 {
                                let mut changed = ui
                                    .add(
                                        egui::Slider::new(&mut self.iteration, 0..=count - 1)
                                            .text("Iteração do replay"),
                                    )
                                    .changed();
                                let ants = self.archive.as_ref().unwrap().iterations
                                    [self.iteration]
                                    .ants
                                    .len();
                                self.ant = self.ant.min(ants.saturating_sub(1));
                                if ants > 0 {
                                    changed |= ui
                                        .add(
                                            egui::Slider::new(&mut self.ant, 0..=ants - 1)
                                                .text("Formiga do replay"),
                                        )
                                        .changed();
                                }
                                let decisions = self.archive.as_ref().unwrap().iterations
                                    [self.iteration]
                                    .ants
                                    .get(self.ant)
                                    .map_or(0, |a| a.decisions.len());
                                changed |= ui
                                    .add(
                                        egui::Slider::new(&mut self.decision, 0..=decisions)
                                            .text("Decisões visíveis"),
                                    )
                                    .changed();
                                if changed {
                                    self.replay_selection();
                                }
                            }
                        }
                        ui.collapsing("Decisões e probabilidades", |ui| self.alternatives(ui));
                        ui.small("Câmera: roda para zoom · botão direito/meio para mover · Q/E para girar · F11 para tela cheia.");
                    }
                    1 => self.matrix(ui),
                    2 => self.inspect_candidates(ui, shell),
                    4 => {
                        self.viewport_status(ui);
                        egui::ScrollArea::horizontal().show(ui, |ui| self.alternatives(ui));
                    }
                    _ => self.history(ui),
                }
            },
        );
        #[cfg(test)]
        self.buttons.extend(toolbar_buttons);
        if let Some(view) = requested {
            match view {
                AcoView::Best => self.view_best(),
                _ => self.view_generation(),
            };
        }
        if refit {
            self.fit_revision = self.fit_revision.wrapping_add(1);
        }
        if test_ppo {
            self.begin_selected_ppo_test(shell);
        }
        if self.live.as_ref().is_some_and(|l| !l.is_finished()) || self.playing {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
    }
    fn alternatives(&self, ui: &mut egui::Ui) {
        if let Some(c) = self.shown_candidate() {
            let decisions = &c.decisions[..self.shown_decision().min(c.decisions.len())];
            ui.label(format!(
                "Decisões tomadas: {}",
                decisions
                    .iter()
                    .map(|d| format!("{}->{}", d.position, d.choice))
                    .collect::<Vec<_>>()
                    .join(" · ")
            ));
            if let Some(d) = decisions.last() {
                ui.label(format!(
                    "Decisão {} · alternativa escolhida {}",
                    d.position, d.choice
                ));
                let tau = self
                    .archive
                    .as_ref()
                    .and_then(|a| a.iterations.get(self.iteration))
                    .map_or(&self.pheromones, |i| &i.pheromones_before);
                ui.strong(format!(
                    "Escolhida {}: probabilidade {:.2}% · feromônio usado {}",
                    d.choice,
                    d.probabilities[d.choice] * 100.,
                    tau.get(d.position)
                        .and_then(|r| r.get(d.choice))
                        .map_or_else(|| "não retido".into(), |tau| format!("{tau:.5}"))
                ));
                egui::Grid::new("aco_choices").striped(true).show(ui, |ui| {
                    for label in [
                        "Alternativa",
                        "Valor",
                        "Probabilidade real",
                        "Feromônio τ usado",
                        "Viabilidade",
                    ] {
                        ui.strong(label);
                    }
                    ui.end_row();
                    for (choice, p) in d.probabilities.iter().enumerate() {
                        ui.label(format!(
                            "{} {choice}",
                            if choice == d.choice { ">" } else { " " }
                        ));
                        ui.label(
                            if self.config.generator.is_sequential()
                                && d.position < self.config.generator.sectors
                            {
                                procedural::sequential::label(choice).to_string()
                            } else if d.position < self.config.generator.sectors {
                                format!("raio ×{:.2}", self.config.generator.radial_levels[choice])
                            } else {
                                format!("largura {:.1}", self.config.generator.widths[choice])
                            },
                        );
                        ui.label(format!("{:.2}%", p * 100.));
                        ui.monospace(
                            tau.get(d.position)
                                .and_then(|r| r.get(choice))
                                .map_or_else(|| "não retido".into(), |t| format!("{t:.5}")),
                        );
                        if let Some(reason) = d
                            .construction
                            .as_ref()
                            .and_then(|v| v.alternatives.get(choice))
                            .and_then(|a| a.rejection.as_ref())
                        {
                            ui.label("inviável").on_hover_text(reason);
                        } else {
                            ui.label("viável");
                        }
                        ui.end_row();
                    }
                });
            }
        }
    }
    fn matrix(&self, ui: &mut egui::Ui) {
        ui.label("Matriz atual · linha = decisão, coluna = alternativa");
        egui::ScrollArea::both().max_height(500.).show(ui, |ui| {
            egui::Grid::new("aco_matrix").striped(true).show(ui, |ui| {
                for (position, row) in self.pheromones.iter().enumerate() {
                    ui.monospace(position.to_string());
                    for value in row {
                        ui.colored_label(egui::Color32::LIGHT_GREEN, format!("{value:.4}"));
                    }
                    ui.end_row();
                }
            });
            if let Some(i) = self.archive.as_ref().and_then(|a| {
                if self.replay {
                    a.iterations.get(self.iteration)
                } else {
                    a.iterations.last()
                }
            }) {
                ui.separator();
                ui.label("Antes -> evaporado -> reforçado / reiniciado");
                for pos in 0..i.pheromones_before.len() {
                    ui.monospace(format!(
                        "{pos}: {}",
                        (0..i.pheromones_before[pos].len())
                            .map(|c| format!(
                                "{:.3}->{:.3}->{:.3}",
                                i.pheromones_before[pos][c],
                                i.pheromones_evaporated[pos][c],
                                i.pheromones_after[pos][c]
                            ))
                            .collect::<Vec<_>>()
                            .join(" | ")
                    ));
                }
            }
        });
    }
    fn history(&self, ui: &mut egui::Ui) {
        if let Some(a) = &self.archive {
            Plot::new("aco_evolution")
                .height(260.)
                .legend(egui_plot::Legend::default())
                .show(ui, |plot| {
                    plot.line(Line::new(
                        "Fitness médio",
                        PlotPoints::from_iter(
                            a.iterations
                                .iter()
                                .map(|i| [i.iteration as f64 + 1., i.mean_fitness]),
                        ),
                    ));
                    plot.line(Line::new(
                        "Diversidade",
                        PlotPoints::from_iter(
                            a.iterations
                                .iter()
                                .map(|i| [i.iteration as f64 + 1., i.diversity]),
                        ),
                    ));
                });
            for i in &a.iterations {
                ui.label(format!(
                    "Iteração {} · candidatas {} · diversidade {:.3} · melhor {}",
                    i.iteration + 1,
                    i.candidates_built,
                    i.diversity,
                    i.best_iteration.as_deref().unwrap_or("nenhuma")
                ));
            }
        } else {
            ui.label("Aguardando primeira iteração do motor.");
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    fn frame(d: &mut SearchDashboard, ctx: &egui::Context, events: Vec<egui::Event>) {
        let mut output = ctx.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1400., 1000.),
                )),
                events,
                ..Default::default()
            },
            |ui| d.show(ui.ctx(), &mut Shell::default()),
        );
        assert!(!output.shapes.is_empty());
        output.textures_delta.clear();
    }
    fn click(d: &mut SearchDashboard, ctx: &egui::Context, label: &'static str) {
        frame(d, ctx, vec![]);
        let pos = d.buttons[label].center();
        frame(d, ctx, vec![egui::Event::PointerMoved(pos)]);
        for pressed in [true, false] {
            frame(
                d,
                ctx,
                vec![egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                }],
            );
        }
    }
    fn until(
        d: &mut SearchDashboard,
        ctx: &egui::Context,
        condition: impl Fn(&SearchDashboard) -> bool,
    ) {
        // Includes full organic iterations while the workspace runs other physics tests.
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            frame(d, ctx, vec![]);
            if condition(d) {
                return;
            }
            assert!(Instant::now() < deadline, "UI timeout: {}", d.status);
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    #[test]
    fn ui_start_pause_decision_iteration_resume_restart_and_export_real_search() {
        let mut d = SearchDashboard {
            config: AcoConfig {
                ants: 1,
                iterations: 2,
                max_candidates: 2,
                ..Default::default()
            },
            speed: 120.,
            ..Default::default()
        };
        let ctx = egui::Context::default();
        // Pointer events hit the actual visible egui controls, not start() directly.
        frame(&mut d, &ctx, vec![]);
        frame(&mut d, &ctx, vec![]);
        click(&mut d, &ctx, "Iniciar");
        assert!(d.live.is_some(), "Iniciar must create a real worker");
        click(&mut d, &ctx, "Pausar");
        until(&mut d, &ctx, |d| {
            d.live.as_ref().unwrap().control.is_paused()
        });
        let before = d.decision;
        assert!(before < 12);
        std::thread::sleep(Duration::from_millis(30));
        frame(&mut d, &ctx, vec![]);
        assert_eq!(d.decision, before);
        click(&mut d, &ctx, "Uma decisão");
        until(&mut d, &ctx, |d| {
            d.live.as_ref().unwrap().control.is_paused() && d.decision == before + 1
        });
        assert!(
            d.definition.is_none(),
            "partial geometry must never be a valid circuit"
        );
        let c = d.current.as_ref().unwrap();
        let chosen = c.decisions.last().unwrap();
        let p = chosen
            .construction
            .as_ref()
            .and_then(|v| v.points.last())
            .copied()
            .unwrap_or([0., 0.]);
        assert!(p.iter().all(|v| v.is_finite()));
        click(&mut d, &ctx, "Uma iteração");
        until(&mut d, &ctx, |d| {
            d.live.as_ref().unwrap().control.is_paused()
                && d.archive.as_ref().is_some_and(|a| a.state.iteration == 1)
        });
        assert!(d.definition.is_some());
        let first = d.archive.as_ref().unwrap().iterations[0].ants[0]
            .decisions
            .clone();
        assert_ne!(
            d.pheromones,
            d.archive.as_ref().unwrap().iterations[0].pheromones_before
        );
        click(&mut d, &ctx, "Continuar");
        until(&mut d, &ctx, |d| {
            d.archive.as_ref().is_some_and(SearchArchive::is_finished)
        });
        let expected = d.archive.as_ref().unwrap().state.pheromones.clone();
        until(&mut d, &ctx, |d| d.live.as_ref().unwrap().is_finished());
        click(&mut d, &ctx, "Reiniciar com mesma seed");
        until(&mut d, &ctx, |d| {
            d.archive.as_ref().is_some_and(SearchArchive::is_finished)
        });
        assert_eq!(
            first,
            d.archive.as_ref().unwrap().iterations[0].ants[0].decisions
        );
        assert_eq!(expected, d.archive.as_ref().unwrap().state.pheromones);
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/aco-live-ui-validation")
            .join(format!(
                "{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ));
        std::fs::create_dir_all(&dir).unwrap();
        let a = d.archive.as_ref().unwrap();
        a.save(dir.join("search.ron")).unwrap();
        a.export_selected(dir.join("tracks")).unwrap();
        d.path = dir.join("nested/ui-search.ron").display().to_string();
        d.save_archive().unwrap();
        assert!(dir.join("nested/ui-search.ron").is_file());
        let restored = SearchArchive::load(dir.join("search.ron")).unwrap();
        assert_eq!(restored.state.pheromones, expected);
        println!(
            "UI pointer-driven start/pause/step/iteration/continue/restart passed; artifacts={}",
            dir.display()
        );
    }
    #[test]
    fn ui_interrupts_a_paused_search_and_mode_switch_suspends_computation() {
        let mut d = SearchDashboard {
            speed: 120.,
            ..Default::default()
        };
        let ctx = egui::Context::default();
        frame(&mut d, &ctx, vec![]);
        frame(&mut d, &ctx, vec![]);
        click(&mut d, &ctx, "Iniciar");
        d.suspend();
        until(&mut d, &ctx, |d| {
            d.live.as_ref().unwrap().control.is_paused()
        });
        let decision = d.decision;
        std::thread::sleep(Duration::from_millis(25));
        frame(&mut d, &ctx, vec![]);
        assert_eq!(d.decision, decision);
        click(&mut d, &ctx, "Interromper");
        until(&mut d, &ctx, |d| d.live.as_ref().unwrap().is_finished());
        frame(&mut d, &ctx, vec![]);
        assert!(d.status.contains("cancelled"));
    }
    #[test]
    fn aco_frozen_policy_is_selected_by_click_without_copying_a_path() {
        use super::super::ppo_checkpoint_browser::{
            PpoCheckpointBrowser, test_support::saved_policy,
        };
        let root = std::env::temp_dir().join(format!("aco-policy-click-{}", std::process::id()));
        let path = root.join("long-training/checkpoint-004000.ron");
        racing_rl::persistence::save(&path, &saved_policy(4000)).unwrap();
        let path = std::fs::canonicalize(path).unwrap();
        let mut shell = Shell::default();
        shell.checkpoint_browser = PpoCheckpointBrowser::from_roots(vec![root.clone()]);
        let mut d = SearchDashboard {
            use_policy: true,
            ..Default::default()
        };
        let ctx = egui::Context::default();
        fn draw(
            d: &mut SearchDashboard,
            shell: &mut Shell,
            ctx: &egui::Context,
            events: Vec<egui::Event>,
        ) {
            let mut output = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1400., 1000.),
                    )),
                    events,
                    ..Default::default()
                },
                |ui| d.show(ui.ctx(), shell),
            );
            output.textures_delta.clear();
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while !shell.checkpoint_browser.row_rects.contains_key(&path) {
            draw(&mut d, &mut shell, &ctx, vec![]);
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(2));
        }
        draw(&mut d, &mut shell, &ctx, vec![]);
        let pos = shell.checkpoint_browser.row_rects[&path].center();
        draw(
            &mut d,
            &mut shell,
            &ctx,
            vec![egui::Event::PointerMoved(pos)],
        );
        for pressed in [true, false] {
            draw(
                &mut d,
                &mut shell,
                &ctx,
                vec![egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed,
                    modifiers: Default::default(),
                }],
            );
        }
        assert_eq!(PathBuf::from(d.checkpoint), path);
        assert!(
            !shell.load_policy,
            "ACO selection must keep the search mode"
        );
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(root.join("long-training")).unwrap();
        std::fs::remove_dir(root).unwrap();
    }
    #[test]
    fn archive_replay_shows_open_polygon_until_all_decisions_are_visible() {
        let mut a = SearchArchive::new(
            AcoConfig {
                ants: 1,
                iterations: 1,
                max_candidates: 1,
                ..Default::default()
            },
            47,
        )
        .unwrap();
        a.run_iteration(None).unwrap();
        let expected = ron::to_string(&a).unwrap();
        let mut d = SearchDashboard {
            config: a.config.clone(),
            archive: Some(a),
            replay: true,
            decision: 3,
            ..Default::default()
        };
        d.replay_selection();
        assert!(d.definition.is_none());
        frame(&mut d, &egui::Context::default(), vec![]);
        d.decision = d.config.generator.sectors + 1;
        d.replay_selection();
        assert!(d.definition.is_some());
        assert_eq!(
            ron::to_string(d.archive.as_ref().unwrap()).unwrap(),
            expected
        );
        d.archive = Some(SearchArchive::new(AcoConfig::default(), 47).unwrap());
        d.current = None;
        d.playing = true;
        d.last_advance = Instant::now() - Duration::from_secs(1);
        frame(&mut d, &egui::Context::default(), vec![]);
        assert!(!d.playing);
    }
}
#[cfg(test)]
#[path = "search_dashboard_render_test.rs"]
mod render_test;
