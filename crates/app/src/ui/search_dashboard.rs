//! Read-only inspection of the headless ACO's persisted decision graph.
use crate::{
    aco::SearchArchive,
    simulation::TrackDefinition,
    visual::{Shell, policy_input},
};
use bevy::prelude::Resource;
use bevy_egui::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::{path::Path, time::Instant};

#[derive(Resource)]
pub(crate) struct SearchDashboard {
    path: String,
    archive: Option<SearchArchive>,
    iteration: usize,
    ant: usize,
    decision: usize,
    playing: bool,
    show_graph: bool,
    show_comparison: bool,
    show_history: bool,
    last_advance: Instant,
    status: String,
    pub pending_track: Option<TrackDefinition>,
}
impl Default for SearchDashboard {
    fn default() -> Self {
        Self {
            path: "results_aco/search.ron".into(),
            archive: None,
            iteration: 0,
            ant: 0,
            decision: 0,
            playing: false,
            show_graph: true,
            show_comparison: true,
            show_history: true,
            last_advance: Instant::now(),
            status: String::new(),
            pending_track: None,
        }
    }
}
impl SearchDashboard {
    pub(crate) fn show(&mut self, ctx: &egui::Context, shell: &mut Shell) {
        if self.playing && self.last_advance.elapsed().as_secs_f32() >= 0.25 {
            let current = self
                .archive
                .as_ref()
                .and_then(|a| a.iterations.get(self.iteration));
            if let Some(iteration) = current {
                let decisions = iteration
                    .ants
                    .get(self.ant)
                    .map_or(0, |a| a.decisions.len());
                if self.decision + 1 < decisions {
                    self.decision += 1;
                } else if self.ant + 1 < iteration.ants.len() {
                    self.ant += 1;
                    self.decision = 0;
                } else if self.iteration + 1 < self.archive.as_ref().unwrap().iterations.len() {
                    self.iteration += 1;
                    self.ant = 0;
                    self.decision = 0;
                } else {
                    self.playing = false;
                }
            } else {
                self.playing = false;
            }
            self.last_advance = Instant::now();
            self.select_track();
        }
        let mut requested_best = None;
        super::draw_dashboard(ctx, "ACO · reprodução do algoritmo headless", |ui| {
            ui.heading("Busca ACO");
            ui.label("Arquivo de snapshots (.ron)");
            ui.text_edit_singleline(&mut self.path);
            if ui.button("Abrir / atualizar busca").clicked() {
                match SearchArchive::load(Path::new(&self.path)) {
                    Ok(archive) => {
                        self.archive = Some(archive);
                        self.iteration = 0;
                        self.ant = 0;
                        self.decision = 0;
                        self.status.clear();
                        self.select_track();
                    }
                    Err(error) => self.status = error.to_string(),
                }
            }
            if !self.status.is_empty() {
                ui.colored_label(egui::Color32::LIGHT_RED, &self.status);
            }
            let count = self.archive.as_ref().map_or(0, |a| a.iterations.len());
            if count == 0 {
                ui.label("Gere uma busca com o comando aco search e carregue seus snapshots. Não há resultados fabricados.");
                return;
            }
            if let Some(archive) = &self.archive {
                ui.collapsing("Parâmetros / orçamento", |ui| {
                    ui.monospace(format!("Seed {} · {:?}\n{} formigas × {} iterações\nCandidatas {}/{} · passos {}/{}\nα={} β={} evaporação={} q0={}\nGeometria {:?} · {} setores\nEscalas {:?} · larguras {:?}",archive.state.seed,archive.config.strategy,archive.config.ants,archive.config.iterations,archive.state.candidates,archive.config.max_candidates,archive.state.evaluation_steps,archive.config.max_evaluation_steps,archive.config.alpha,archive.config.beta,archive.config.evaporation,archive.config.exploitation_probability,archive.config.generator.shape,archive.config.generator.sectors,archive.config.generator.radial_levels,archive.config.generator.widths));
                });
            }
            ui.checkbox(&mut self.playing, "Reproduzir construção / evolução");
            ui.horizontal_wrapped(|ui| {
                ui.checkbox(&mut self.show_graph, "Grafo");
                ui.checkbox(&mut self.show_comparison, "Candidatas");
                ui.checkbox(&mut self.show_history, "Histórico");
            });
            let previous = (self.iteration, self.ant);
            ui.add(egui::Slider::new(&mut self.iteration, 0..=count - 1).text("Iteração"));
            let iteration = &self.archive.as_ref().unwrap().iterations[self.iteration];
            self.ant = self.ant.min(iteration.ants.len().saturating_sub(1));
            if !iteration.ants.is_empty() {
                ui.add(
                    egui::Slider::new(&mut self.ant, 0..=iteration.ants.len() - 1).text("Formiga"),
                );
            }
            ui.label(format!(
                "Fitness médio {:.4} · diversidade {:.3}",
                iteration.mean_fitness, iteration.diversity
            ));
            ui.label(format!(
                "Melhor iteração: {}",
                iteration.best_iteration.as_deref().unwrap_or("nenhuma")
            ));
            ui.label(format!(
                "Melhor global: {}",
                iteration.best_global.as_deref().unwrap_or("nenhuma")
            ));
            ui.horizontal(|ui| {
                if ui.button("Ver melhor da iteração").clicked() {
                    requested_best = iteration.best_iteration.clone();
                }
                if ui.button("Ver melhor global").clicked() {
                    requested_best = iteration.best_global.clone();
                }
            });
            if iteration.stagnation_reset {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "Feromônios reiniciados por estagnação",
                );
            }
            if let Some(ant) = iteration.ants.get(self.ant) {
                if let Some(reason) = &ant.rejection {
                    ui.colored_label(egui::Color32::LIGHT_RED, format!("Rejeitada: {reason}"));
                }
                if let Some(fitness) = &ant.fitness {
                    ui.label(format!("Fitness {:.4}", fitness.score));
                    ui.monospace(format!("Conclusão {:?}\nProgresso {:?}\nColisões {:?}\nGeometria {:.4}\nDiversidade {:.4}\nPassos {}\nReferência {:?}", fitness.completion_rate, fitness.mean_progress, fitness.collision_rate, fitness.geometric_difficulty, fitness.diversity, fitness.simulation_steps, fitness.reference_completed));
                }
                if let Some(candidate) = &ant.candidate {
                    ui.separator();
                    ui.strong(&candidate.definition.id);
                    ui.label(format!("Largura {:.1}", candidate.definition.width));
                    if ui.button("Mostrar pista selecionada").clicked() {
                        self.pending_track = Some(candidate.definition.clone());
                    }
                    if ui.button("Usar pista no piloto PPO").clicked() {
                        shell.track = candidate.definition.id.clone();
                        shell.status = "Carregue um checkpoint para observar esta pista. A pista será registrada somente como Training.".into();
                        self.pending_track = Some(candidate.definition.clone());
                    }
                }
            }
            if previous != (self.iteration, self.ant) {
                self.select_track();
            }
            ui.separator();
            ui.collapsing("Piloto congelado na pista", |ui| policy_input(ui, shell));
            ui.small("Os feromônios representam decisões discretas do gerador, não posições físicas da pista.");
        });
        if let Some(id) = requested_best {
            let found = self.archive.as_ref().and_then(|archive| {
                archive
                    .iterations
                    .iter()
                    .enumerate()
                    .find_map(|(i, iteration)| {
                        iteration
                            .ants
                            .iter()
                            .position(|ant| {
                                ant.candidate
                                    .as_ref()
                                    .is_some_and(|c| c.definition.id == id)
                            })
                            .map(|ant| (i, ant))
                    })
            });
            if let Some((iteration, ant)) = found {
                self.iteration = iteration;
                self.ant = ant;
                self.decision = 0;
                self.select_track();
            }
        }
        let Some(archive) = &self.archive else {
            return;
        };
        let Some(iteration) = archive.iterations.get(self.iteration) else {
            return;
        };
        if self.show_graph {
            egui::Window::new("Grafo de decisões · probabilidades e feromônios")
                .default_pos([18.0, 58.0])
                .default_width(660.0)
                .default_height(470.0)
                .show(ctx, |ui| {
                    if let Some(ant) = iteration.ants.get(self.ant) {
                        if !ant.decisions.is_empty() {
                            self.decision = self.decision.min(ant.decisions.len() - 1);
                            ui.add(
                                egui::Slider::new(&mut self.decision, 0..=ant.decisions.len() - 1)
                                    .text("Etapa de construção"),
                            );
                        }
                        ui.horizontal_wrapped(|ui| {
                            for (index, decision) in ant.decisions.iter().enumerate() {
                                if ui
                                    .selectable_label(
                                        self.decision == index,
                                        format!("{} → {}", decision.position, decision.choice),
                                    )
                                    .clicked()
                                {
                                    self.decision = index;
                                }
                            }
                        });
                        if let Some(decision) = ant.decisions.get(self.decision) {
                            ui.label(format!(
                                "Posição {} · alternativa sorteada {}",
                                decision.position, decision.choice
                            ));
                            egui::Grid::new("aco_decision_table")
                                .striped(true)
                                .show(ui, |ui| {
                                    for label in [
                                        "Alternativa",
                                        "P(escolha)",
                                        "τ antes",
                                        "τ evaporado",
                                        "τ reforçado",
                                    ] {
                                        ui.strong(label);
                                    }
                                    ui.end_row();
                                    for (choice, probability) in
                                        decision.probabilities.iter().enumerate()
                                    {
                                        let value = if decision.position
                                            < archive.config.generator.sectors
                                        {
                                            archive
                                                .config
                                                .generator
                                                .radial_levels
                                                .get(choice)
                                                .map(|v| format!("×{v:.2}"))
                                        } else {
                                            archive
                                                .config
                                                .generator
                                                .widths
                                                .get(choice)
                                                .map(|v| format!("largura {v:.1}"))
                                        };
                                        ui.label(format!(
                                            "{} {choice}: {}",
                                            if choice == decision.choice {
                                                "●"
                                            } else {
                                                " "
                                            },
                                            value.unwrap_or_default()
                                        ));
                                        ui.add(
                                            egui::ProgressBar::new(*probability as f32)
                                                .text(format!("{:.2}%", probability * 100.0))
                                                .desired_width(110.0),
                                        );
                                        for matrix in [
                                            &iteration.pheromones_before,
                                            &iteration.pheromones_evaporated,
                                            &iteration.pheromones_after,
                                        ] {
                                            ui.monospace(
                                                matrix
                                                    .get(decision.position)
                                                    .and_then(|row| row.get(choice))
                                                    .map_or("—".into(), |v| format!("{v:.5}")),
                                            );
                                        }
                                        ui.end_row();
                                    }
                                });
                        }
                    }
                    ui.separator();
                    ui.label("Matriz τ após reforço · linha = posição; coluna = alternativa");
                    egui::ScrollArea::both().max_height(180.0).show(ui, |ui| {
                        egui::Grid::new("aco_pheromone_matrix")
                            .spacing([3.0, 3.0])
                            .show(ui, |ui| {
                                for (position, row) in iteration.pheromones_after.iter().enumerate()
                                {
                                    ui.monospace(position.to_string());
                                    let max = row.iter().copied().fold(1e-12_f64, f64::max);
                                    for (choice, value) in row.iter().enumerate() {
                                        let intensity = (value / max).clamp(0.0, 1.0);
                                        let (rect, response) = ui.allocate_exact_size(
                                            egui::vec2(36.0, 20.0),
                                            egui::Sense::hover(),
                                        );
                                        ui.painter().rect_filled(
                                            rect,
                                            2.0,
                                            egui::Color32::from_rgb(
                                                20,
                                                (40.0 + 180.0 * intensity) as u8,
                                                105,
                                            ),
                                        );
                                        ui.painter().text(
                                            rect.center(),
                                            egui::Align2::CENTER_CENTER,
                                            choice.to_string(),
                                            egui::FontId::monospace(10.0),
                                            egui::Color32::WHITE,
                                        );
                                        response.on_hover_text(format!(
                                            "posição {position}, alternativa {choice}: τ={value:.7}"
                                        ));
                                    }
                                    ui.end_row();
                                }
                            });
                    });
                });
        }
        if self.show_comparison {
            egui::Window::new("Comparar formigas / candidatas")
                .default_pos([700.0, 58.0])
                .default_width(310.0)
                .default_height(280.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::both().max_height(260.0).show(ui, |ui| {
                        egui::Grid::new("aco_candidates")
                            .striped(true)
                            .show(ui, |ui| {
                                for label in [
                                    "Formiga",
                                    "Fitness",
                                    "Progresso",
                                    "Conclusão",
                                    "Diversidade",
                                    "Status",
                                ] {
                                    ui.strong(label);
                                }
                                ui.end_row();
                                for (index, ant) in iteration.ants.iter().enumerate() {
                                    if ui
                                        .selectable_label(self.ant == index, ant.ant.to_string())
                                        .clicked()
                                    {
                                        self.ant = index;
                                        self.pending_track =
                                            ant.candidate.as_ref().map(|c| c.definition.clone());
                                    }
                                    ui.monospace(
                                        ant.fitness
                                            .as_ref()
                                            .map_or("—".into(), |f| format!("{:.3}", f.score)),
                                    );
                                    ui.monospace(
                                        ant.fitness
                                            .as_ref()
                                            .and_then(|f| f.mean_progress)
                                            .map_or("—".into(), |p| format!("{:.1}%", p * 100.0)),
                                    );
                                    ui.monospace(
                                        ant.fitness
                                            .as_ref()
                                            .and_then(|f| f.completion_rate)
                                            .map_or("—".into(), |p| format!("{:.1}%", p * 100.0)),
                                    );
                                    ui.monospace(
                                        ant.fitness.as_ref().map_or("—".into(), |f| {
                                            format!("{:.3}", f.diversity)
                                        }),
                                    );
                                    ui.label(ant.rejection.as_deref().unwrap_or("válida"));
                                    ui.end_row();
                                }
                            });
                    });
                });
        }
        if self.show_history {
            egui::Window::new("Evolução da busca")
                .default_pos([18.0, 550.0])
                .default_width(660.0)
                .show(ctx, |ui| {
                    Plot::new("aco_evolution")
                        .height(160.0)
                        .legend(egui_plot::Legend::default())
                        .show(ui, |plot| {
                            plot.line(Line::new(
                                "Fitness médio",
                                PlotPoints::from_iter(
                                    archive
                                        .iterations
                                        .iter()
                                        .map(|i| [i.iteration as f64, i.mean_fitness]),
                                ),
                            ));
                            plot.line(Line::new(
                                "Diversidade",
                                PlotPoints::from_iter(
                                    archive
                                        .iterations
                                        .iter()
                                        .map(|i| [i.iteration as f64, i.diversity]),
                                ),
                            ));
                        });
                });
        }
        if self.playing {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
    fn select_track(&mut self) {
        self.pending_track = self
            .archive
            .as_ref()
            .and_then(|a| a.iterations.get(self.iteration))
            .and_then(|i| i.ants.get(self.ant))
            .and_then(|a| a.candidate.as_ref())
            .map(|c| c.definition.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aco::AcoConfig;
    #[test]
    fn real_search_snapshot_is_rendered_without_mutating_search_state() {
        let mut archive = SearchArchive::new(
            AcoConfig {
                ants: 2,
                iterations: 1,
                max_candidates: 2,
                ..Default::default()
            },
            73,
        )
        .unwrap();
        archive.run_iteration(None).unwrap();
        let expected = ron::to_string(&archive).unwrap();
        let mut dashboard = SearchDashboard {
            archive: Some(archive),
            ..Default::default()
        };
        let context = egui::Context::default();
        let output = context.run_ui(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1400.0, 850.0),
                )),
                ..Default::default()
            },
            |ui| dashboard.show(ui.ctx(), &mut Shell::default()),
        );
        assert!(!output.shapes.is_empty());
        assert_eq!(
            ron::to_string(dashboard.archive.as_ref().unwrap()).unwrap(),
            expected
        );
    }
}
