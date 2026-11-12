use super::*;

impl SearchDashboard {
    pub(super) fn shown_candidate(&self) -> Option<&AntSnapshot> {
        self.inspected.as_ref().or(self.current.as_ref())
    }
    pub(super) fn shown_decision(&self) -> usize {
        if self.inspected.is_some() {
            self.inspected_decision
        } else {
            self.decision
        }
    }
    pub(super) fn select_inspection(&mut self, key: (usize, usize), record: AntSnapshot) {
        self.view = AcoView::Candidate;
        self.fit_revision = self.fit_revision.wrapping_add(1);
        self.playing = false;
        self.ppo_test_track = None;
        self.inspected_decision = record.decisions.len();
        self.definition = record
            .candidate
            .as_ref()
            .map(|c| c.definition.clone())
            .or_else(|| {
                record
                    .diagnostic
                    .as_ref()
                    .and_then(|d| d.definition.clone())
            });
        self.geometry_error = (!record.geometrically_valid()).then(|| {
            record
                .rejection
                .clone()
                .unwrap_or_else(|| "Geometria não validada".into())
        });
        self.inspected_key = Some(key);
        self.inspected = Some(record);
    }
    pub(super) fn inspect_candidates(&mut self, ui: &mut egui::Ui, shell: &mut Shell) {
        ui.label("Todas as candidatas: validade geométrica e seleção são estados separados.");
        ui.horizontal_wrapped(|ui| {
            for (i, label) in [
                "Todas",
                "Aceitas",
                "Rejeitadas",
                "Geometria inválida",
                "Válidas rejeitadas",
            ]
            .iter()
            .enumerate()
            {
                ui.selectable_value(&mut self.candidate_filter, i, *label);
            }
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Motivo (nome ou texto):");
            ui.text_edit_singleline(&mut self.reason_filter);
        });
        let archived_current = self
            .archive
            .as_ref()
            .is_some_and(|a| a.iterations.iter().any(|i| i.iteration == self.iteration));
        let rows: Vec<_> = self
            .archive
            .iter()
            .flat_map(|a| &a.iterations)
            .flat_map(|i| i.ants.iter().map(move |c| ((i.iteration, c.ant), c)))
            .chain(self.archive.iter().flat_map(|a| {
                a.in_progress
                    .iter()
                    .map(move |c| ((a.state.iteration, c.ant), c))
            }))
            .chain(
                self.current_ants
                    .iter()
                    .filter(|_| !archived_current)
                    .map(|c| ((self.iteration, c.ant), c)),
            )
            .collect();
        let mut counts = std::collections::BTreeMap::<String, usize>::new();
        for (_, c) in &rows {
            *counts
                .entry(format!("{:?}", c.rejection_reason))
                .or_default() += 1;
        }
        ui.small(format!(
            "{} candidatas · {}",
            rows.len(),
            counts
                .iter()
                .map(|(k, v)| format!("{k}: {v}"))
                .collect::<Vec<_>>()
                .join(" | ")
        ));
        let mut selected = None;
        let mut compared = None;
        #[cfg(test)]
        let mut candidate_hits = vec![];
        egui::ScrollArea::vertical()
            .id_salt("aco_inspection_list")
            .max_height(210.)
            .show(ui, |ui| {
                for (key, c) in rows {
                    let show = match self.candidate_filter {
                        1 => c.rejection.is_none(),
                        2 => c.rejection.is_some(),
                        3 => !c.geometrically_valid(),
                        4 => c.geometrically_valid() && c.rejection.is_some(),
                        _ => true,
                    };
                    let reason = format!(
                        "{:?} {}",
                        c.rejection_reason,
                        c.rejection.as_deref().unwrap_or("aceita")
                    );
                    if !show
                        || !reason
                            .to_lowercase()
                            .contains(&self.reason_filter.to_lowercase())
                    {
                        continue;
                    }
                    ui.push_id(key, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.strong(format!("G{} · formiga {}", key.0 + 1, key.1 + 1));
                            let open = ui.button("Ver pista");
                            #[cfg(test)]
                            candidate_hits.push((key, open.rect));
                            if open.clicked() {
                                selected = Some((key, c.clone()));
                            }
                            if c.retention_evicted {
                                ui.small("Detalhes expirados pela retenção");
                            } else if ui.small_button("Comparar").clicked() {
                                compared = Some(c.clone());
                            }
                        });
                        ui.label(c.classification());
                        ui.small(&reason);
                        ui.separator();
                    });
                }
            });
        #[cfg(test)]
        self.candidate_buttons.extend(candidate_hits);
        if let Some((key, c)) = selected {
            self.select_inspection(key, c);
        }
        if let Some(c) = compared {
            self.comparison = Some(c);
        }
        if ui.button("Voltar à construção atual").clicked() {
            self.view_generation();
        }
        if self.inspected.is_some() {
            ui.separator();
            let c = self.inspected.as_ref().unwrap();
            ui.strong(c.classification());
            if let Some(r) = &c.rejection {
                ui.label(r);
            }
            if c.geometrically_valid()
                && c.evaluation
                    .as_ref()
                    .is_some_and(|e| !e.reference_completed)
            {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    "O piloto de referência falhou; isso não prova impossibilidade da pista.",
                );
            }
            if let Some(e) = &c.evaluation {
                ui.label(format!(
                    "Referência: concluiu={} · colisão={} · progresso {:.1}% · {} decisões",
                    e.reference_completed,
                    e.reference_collision,
                    e.reference_progress * 100.,
                    e.reference_steps
                ));
            }
            ui.label(format!(
                "Larguras configuradas {:?} · raio base {:.1} · folga {:.1}",
                self.config.generator.widths,
                self.config.generator.primitive_radius,
                self.config.generator.minimum_border_gap
            ));
            if c.retention_evicted {
                ui.label("O resumo e as decisões foram preservados; a geometria detalhada expirou. Aumente a retenção em próximas execuções.");
            }
            let moved = ui
                .add(
                    egui::Slider::new(&mut self.inspected_decision, 0..=c.decisions.len())
                        .text("Decisões visíveis"),
                )
                .changed();
            if moved {
                self.definition = if self.inspected_decision == c.decisions.len() {
                    c.candidate
                        .as_ref()
                        .map(|t| t.definition.clone())
                        .or_else(|| c.diagnostic.as_ref().and_then(|d| d.definition.clone()))
                } else {
                    None
                };
            }
            ui.horizontal_wrapped(|ui| {
                if ui.button("Ver decisões").clicked() {
                    self.tab = 4;
                }
                if ui.button("Limpar comparação").clicked() {
                    self.comparison = None;
                }
            });
            egui::Grid::new("aco_compare_metrics")
                .striped(true)
                .show(ui, |ui| {
                    ui.label("Métrica");
                    ui.label("Selecionada");
                    ui.label("Comparação");
                    ui.end_row();
                    let a = c.candidate.as_ref().map(|c| &c.geometry);
                    let b = self
                        .comparison
                        .as_ref()
                        .and_then(|c| c.candidate.as_ref())
                        .map(|c| &c.geometry);
                    let value = |m: Option<&procedural::GeometryMetrics>, metric: usize| {
                        m.map(|m| match metric {
                            0 => m.width,
                            1 => m.width / crate::simulation::CAR_WIDTH,
                            2 => m.length,
                            3 => 1. / m.max_curvature,
                            4 => m.organic.rectangularity,
                            5 => m.organic.heading_entropy,
                            _ => m.longest_straight,
                        })
                    };
                    for (i, label) in [
                        "Largura nominal",
                        "Larguras de carro",
                        "Comprimento",
                        "Raio mínimo medido",
                        "Retangularidade",
                        "Entropia heading",
                        "Maior reta",
                    ]
                    .iter()
                    .enumerate()
                    {
                        ui.label(*label);
                        ui.label(
                            value(a, i)
                                .map_or_else(|| "indisponível".into(), |v| format!("{v:.2}")),
                        );
                        ui.label(
                            value(b, i)
                                .map_or_else(|| "indisponível".into(), |v| format!("{v:.2}")),
                        );
                        ui.end_row();
                    }
                });
            if let Some(f) = &c.fitness {
                ui.label(format!("Score {:.4} · geometria {:.4} · diversidade {:.4} · orgânica {:?} · dificuldade útil {:.4}",f.score,f.geometric_difficulty,f.diversity,f.organic_quality,f.useful_difficulty));
            } else {
                ui.label("Fitness indisponível: não calculado antes desta rejeição.");
            }
            ui.horizontal_wrapped(|ui| {
                ui.label("Diretório de exportação:");
                ui.text_edit_singleline(&mut self.export_path);
            });
            let valid = c.geometrically_valid() && c.candidate.is_some();
            let mut export_valid = false;
            let mut export_diag = false;
            let mut play = false;
            ui.horizontal_wrapped(|ui| {
                export_valid = ui
                    .add_enabled(
                        valid,
                        egui::Button::new("Exportar pista válida selecionada"),
                    )
                    .clicked();
                export_diag = ui
                    .add_enabled(
                        !c.retention_evicted,
                        egui::Button::new("Exportar diagnóstico selecionado"),
                    )
                    .clicked();
                play = ui
                    .add_enabled(valid, egui::Button::new("Testar selecionada com PPO"))
                    .clicked();
            });
            if export_valid || export_diag {
                let result = crate::aco::inspection::export_record(
                    c,
                    &self.config.generator,
                    self.seed,
                    self.inspected_key.unwrap().0,
                    PathBuf::from(self.export_path.trim()).as_path(),
                    export_diag,
                );
                self.status = result.map_or_else(
                    |e| e,
                    |_| "Candidata exportada com metadados de inspeção".into(),
                );
            }
            if play {
                self.begin_selected_ppo_test(shell);
            }
        }
        ui.horizontal_wrapped(|ui| {
            ui.label("Arquivo da busca/inspeção:");
            ui.text_edit_singleline(&mut self.path);
            if ui.button("Salvar snapshot da inspeção").clicked() {
                self.status = self.save_archive().map_or_else(
                    |e| e,
                    |_| "Inspeção salva, incluindo a iteração parcial".into(),
                );
            }
        });
        ui.collapsing("Retenção e exportação automática",|ui|{
            ui.add(egui::DragValue::new(&mut self.config.retention.detailed_candidates).range(0..=100000).prefix("Candidatas detalhadas: "));
            ui.add(egui::DragValue::new(&mut self.config.retention.pheromone_iterations).range(0..=10000).prefix("Iterações com matrizes: "));
            ui.checkbox(&mut self.config.retention.keep_alternatives,"Preservar previews de todas as alternativas (maior arquivo)");
            ui.checkbox(&mut self.config.retention.export_rejected_valid,"CLI: exportar também válidas rejeitadas");
            ui.checkbox(&mut self.config.retention.export_diagnostics,"CLI: exportar diagnósticos inválidos separadamente");
            ui.small("A política vale para novas buscas. Resumos, decisões e motivos permanecem após expirar o detalhe. Melhores selecionadas são preservadas à parte.");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inspector_renders_rejected_valid_and_invalid_snapshots_without_training_leak() {
        use crate::aco::{PilotEvaluation, RejectionReason};
        let mut archive = SearchArchive::new(
            AcoConfig {
                ants: 1,
                iterations: 1,
                max_candidates: 1,
                policy_label: Some("unit evaluator".into()),
                ..Default::default()
            },
            47,
        )
        .unwrap();
        let mut evaluator = |_: &procedural::GeneratedTrack, _: u64, _: usize| {
            Ok(PilotEvaluation {
                reference_episodes: 1,
                reference_steps: 1,
                simulation_steps: 1,
                ..Default::default()
            })
        };
        archive.run_iteration(Some(&mut evaluator)).unwrap();
        let record = archive.iterations[0].ants[0].clone();
        assert_eq!(
            record.rejection_reason,
            Some(RejectionReason::ReferenceTimeout)
        );
        let mut d = SearchDashboard {
            archive: Some(archive),
            ..Default::default()
        };
        d.select_inspection((0, 0), record);
        assert!(d.definition.is_some());
        assert!(d.geometry_error.is_none());
        let ctx = egui::Context::default();
        let mut shell = Shell::default();
        d.tab = 2;
        let frame = |d: &mut SearchDashboard, shell: &mut Shell| {
            let mut result = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1400., 1000.),
                    )),
                    ..Default::default()
                },
                |ui| {
                    d.show(ui.ctx(), shell);
                },
            );
            result.append(ctx.end_pass());
            result.textures_delta.clear();
        };
        frame(&mut d, &mut shell);
        d.tab = 0;
        frame(&mut d, &mut shell);
        let mut invalid = d.inspected.clone().unwrap();
        invalid.candidate = None;
        invalid.geometry_valid = Some(false);
        invalid.rejection_reason = Some(RejectionReason::SelfIntersection);
        invalid.diagnostic = Some(crate::aco::inspection::GeometryDiagnostic::capture(
            None,
            &invalid.decisions,
            None,
            "controlled invalid geometry",
        ));
        d.select_inspection((0, 0), invalid);
        assert!(d.definition.is_none());
        assert!(d.geometry_error.is_some());
        frame(&mut d, &mut shell);
        // A save during the very first, unfinished iteration preserves its trace.
        let mut partial = d.inspected.clone().unwrap();
        partial.retention_evicted = false;
        partial.diagnostic = None;
        partial.rejection = None;
        partial.geometry_valid = None;
        d.archive = None;
        d.current = Some(partial);
        d.current_ants.clear();
        d.iteration = 0;
        d.path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join(format!(
                "../../target/aco-partial-inspection-{}.ron",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ))
            .display()
            .to_string();
        d.save_archive().unwrap();
        let saved = SearchArchive::load(&d.path).unwrap();
        assert_eq!(saved.in_progress.len(), 1);
        assert!(saved.in_progress[0].diagnostic.is_some());
        assert!(saved.iterations.is_empty());
        assert!(d.pending_track.is_none());
        d.inspected.as_mut().unwrap().retention_evicted = true;
        d.inspected
            .as_mut()
            .unwrap()
            .decisions
            .iter_mut()
            .for_each(|x| x.construction = None);
        frame(&mut d, &mut shell);
    }
}
