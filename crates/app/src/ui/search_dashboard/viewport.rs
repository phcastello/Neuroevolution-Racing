//! One explicit selection drives the main scene, captions, decisions and exports.
use super::*;
use crate::aco::inspection::GeometryIssue;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) enum AcoView {
    #[default]
    Generation,
    Best,
    Candidate,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum StrokeKind {
    Path,
    Selected,
    Completion,
    Alternative,
    Rejected,
    Population,
    Comparison,
}
#[derive(Clone, Debug)]
pub(crate) struct PreviewStroke {
    pub points: Vec<[f32; 2]>,
    pub kind: StrokeKind,
    pub closed: bool,
}
#[derive(Clone, Debug, Default)]
pub(crate) struct AcoScene {
    pub road: Option<TrackDefinition>,
    pub strokes: Vec<PreviewStroke>,
    pub issues: Vec<GeometryIssue>,
    pub fit_revision: u64,
}
impl AcoScene {
    pub fn signature(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.fit_revision.hash(&mut h);
        if let Some(d) = &self.road {
            d.id.hash(&mut h);
            d.width.to_bits().hash(&mut h);
        }
        for line in &self.strokes {
            line.kind.hash(&mut h);
            line.closed.hash(&mut h);
            for x in line.points.iter().flatten() {
                x.to_bits().hash(&mut h);
            }
        }
        for issue in &self.issues {
            for x in issue.points.iter().flatten() {
                x.to_bits().hash(&mut h);
            }
        }
        h.finish()
    }
    fn line(&mut self, points: Vec<[f32; 2]>, kind: StrokeKind, closed: bool) {
        self.strokes.push(PreviewStroke {
            points,
            kind,
            closed,
        });
    }
}
impl SearchDashboard {
    pub(super) fn has_best(&self) -> bool {
        self.archive
            .as_ref()
            .is_some_and(|a| !a.state.selected.is_empty())
            || self
                .current_ants
                .iter()
                .any(|a| a.rejection.is_none() && a.candidate.is_some() && a.fitness.is_some())
    }
    pub(super) fn view_best(&mut self) {
        let best = self
            .archive
            .iter()
            .flat_map(|a| &a.state.selected)
            .map(|r| (&r.track, &r.fitness))
            .chain(
                self.current_ants
                    .iter()
                    .filter(|a| a.rejection.is_none())
                    .filter_map(|a| a.candidate.as_ref().zip(a.fitness.as_ref())),
            )
            .max_by(|a, b| a.1.score.total_cmp(&b.1.score));
        let Some((track, fitness)) = best else {
            return;
        };
        let id = track.definition.id.clone();
        let saved = self
            .archive
            .iter()
            .flat_map(|a| &a.iterations)
            .flat_map(|i| i.ants.iter().map(move |a| ((i.iteration, a.ant), a)))
            .chain(
                self.current_ants
                    .iter()
                    .map(|a| ((self.iteration, a.ant), a)),
            )
            .find(|(_, a)| a.candidate.as_ref().is_some_and(|c| c.definition.id == id));
        let (key, mut record) = if let Some((key, record)) = saved {
            (key, record.clone())
        } else {
            let key = (
                track.provenance.attempt / self.config.ants,
                track.provenance.attempt % self.config.ants,
            );
            (
                key,
                AntSnapshot {
                    ant: key.1,
                    decisions: vec![],
                    candidate: None,
                    rejection: None,
                    fitness: Some(fitness.clone()),
                    rejection_reason: None,
                    evaluation: None,
                    geometry_valid: Some(true),
                    diagnostic: None,
                    retention_evicted: false,
                    candidate_id: Some(id),
                },
            )
        };
        record.candidate = Some(track.clone());
        record.retention_evicted = false;
        self.select_inspection(key, record);
        self.view = AcoView::Best;
        self.playing = false;
        self.comparison = None;
    }
    pub(super) fn view_generation(&mut self) {
        self.ppo_test_track = None;
        self.generation_selection = None;
        self.view = AcoView::Generation;
        self.inspected = None;
        self.inspected_key = None;
        self.comparison = None;
        self.restore_current_definition();
        self.fit_revision = self.fit_revision.wrapping_add(1);
        if self.current.is_none() {
            self.select_generation(self.iteration);
        }
    }
    fn restore_current_definition(&mut self) {
        let c = self.shown_candidate();
        let definition = c.and_then(|c| {
            c.candidate
                .as_ref()
                .map(|t| t.definition.clone())
                .or_else(|| c.diagnostic.as_ref().and_then(|d| d.definition.clone()))
        });
        let error = c
            .filter(|c| c.diagnostic.is_some())
            .and_then(|c| c.rejection.clone());
        self.definition = definition;
        self.geometry_error = error;
    }
    fn select_generation(&mut self, index: usize) {
        self.playing = false;
        self.generation_selection = Some(index);
        self.view = AcoView::Generation;
        self.inspected = None;
        self.inspected_key = None;
        if index == self.iteration && self.live.as_ref().is_some_and(|l| !l.is_finished()) {
            self.generation_selection = None;
        } else if let Some(c) = self.generation_records().last().copied().cloned() {
            self.inspected_decision = c.decisions.len();
            self.inspected_key = Some((index, c.ant));
            self.inspected = Some(c);
        }
        self.restore_current_definition();
        self.fit_revision = self.fit_revision.wrapping_add(1);
    }
    fn generation_records(&self) -> Vec<&AntSnapshot> {
        let index = self.generation_selection.unwrap_or(self.iteration);
        if let Some(i) = self.archive.as_ref().and_then(|a| a.iterations.get(index)) {
            return i.ants.iter().collect();
        }
        if index == self.iteration && !self.current_ants.is_empty() {
            return self.current_ants.iter().collect();
        }
        self.archive
            .as_ref()
            .filter(|a| a.state.iteration == index)
            .map_or_else(Vec::new, |a| a.in_progress.iter().collect())
    }
    pub(super) fn view_caption(&self) -> String {
        match self.view {
            AcoView::Best => "Melhor pista · visualização fixa".into(),
            AcoView::Candidate => self.inspected_key.map_or_else(
                || "Candidata selecionada".into(),
                |(i, a)| format!("Candidata · geração {} · formiga {}", i + 1, a + 1),
            ),
            AcoView::Generation => format!(
                "Geração {} · {} candidatas",
                self.generation_selection.unwrap_or(self.iteration) + 1,
                self.generation_records().len()
            ),
        }
    }
    pub(super) fn viewport_status(&mut self, ui: &mut egui::Ui) {
        if self.view == AcoView::Generation {
            let max = self.archive.as_ref().map_or(self.iteration, |a| {
                a.iterations
                    .len()
                    .saturating_sub(1)
                    .max(self.iteration)
                    .max(if a.in_progress.is_empty() {
                        0
                    } else {
                        a.state.iteration
                    })
            });
            let mut selected = self.generation_selection.unwrap_or(self.iteration) + 1;
            if ui
                .add(
                    egui::DragValue::new(&mut selected)
                        .range(1..=max + 1)
                        .prefix("Ver geração: "),
                )
                .changed()
            {
                self.select_generation(selected - 1);
            }
            ui.checkbox(
                &mut self.show_generation_tracks,
                "Mostrar outras pistas da geração",
            );
        }
        if let Some(c) = self.shown_candidate() {
            if self.definition.is_some() && self.geometry_error.is_none() {
                ui.label("Circuito completo · geometria validada");
                return;
            }
            if let Some(d) = &c.diagnostic {
                ui.colored_label(
                    egui::Color32::LIGHT_RED,
                    "Diagnóstico: geometria inválida/incompleta, não apta para treinamento",
                );
                for i in &d.issues {
                    ui.label(&i.message);
                }
            }
            let view = c
                .diagnostic
                .as_ref()
                .and_then(|d| d.terminal.as_ref())
                .or_else(|| {
                    c.decisions
                        .get(self.shown_decision().saturating_sub(1))
                        .and_then(|d| d.construction.as_ref())
                });
            if let Some(e) = view.and_then(|v| v.exploration.as_ref()) {
                ui.label(format!("Fase: {}", e.phase.label()));
                ui.label(&e.reason);
                if e.chose_extra_exploration && c.diagnostic.is_none() {
                    ui.colored_label(
                        egui::Color32::LIGHT_GREEN,
                        "Escolheu passear mais; havia fechamento viável",
                    );
                }
            }
        }
    }
    pub(crate) fn viewport_scene(&self) -> AcoScene {
        let mut scene = AcoScene {
            fit_revision: self.fit_revision,
            ..Default::default()
        };
        if self.view == AcoView::Generation && self.show_generation_tracks {
            for c in self.generation_records() {
                if let Some(t) = &c.candidate {
                    if self
                        .definition
                        .as_ref()
                        .is_some_and(|d| d.id == t.definition.id)
                    {
                        continue;
                    }
                    if let Ok([points, _, _]) = Track::preview_geometry(&t.definition) {
                        scene.line(
                            points.iter().map(|p| p.to_array()).collect(),
                            StrokeKind::Population,
                            true,
                        );
                    }
                }
            }
        }
        if let Some(d) = &self.definition {
            if self.geometry_error.is_none() {
                scene.road = Some(d.clone());
            } else if let Ok(preview) = Track::preview_geometry(d) {
                for points in preview {
                    scene.line(
                        points.iter().map(|p| p.to_array()).collect(),
                        StrokeKind::Rejected,
                        true,
                    );
                }
            }
        }
        if let Some(c) = self.shown_candidate() {
            if let Some(d) = &c.diagnostic {
                scene.issues = d.issues.clone();
            }
            let view = c
                .diagnostic
                .as_ref()
                .filter(|_| self.shown_decision() == c.decisions.len())
                .and_then(|d| d.terminal.as_ref())
                .or_else(|| {
                    c.decisions
                        .get(self.shown_decision().saturating_sub(1))
                        .and_then(|d| d.construction.as_ref())
                });
            if scene.road.is_none() {
                if let Some(v) = view {
                    scene.line(v.points.clone(), StrokeKind::Path, false);
                    scene.line(
                        v.points[v.selected_start.min(v.points.len())..].to_vec(),
                        StrokeKind::Selected,
                        false,
                    );
                    scene.line(v.completion.clone(), StrokeKind::Completion, false);
                    for a in &v.alternatives {
                        scene.line(
                            a.points.clone(),
                            if a.rejection.is_some() {
                                StrokeKind::Rejected
                            } else {
                                StrokeKind::Alternative
                            },
                            false,
                        );
                    }
                } else if !self.config.generator.is_sequential() {
                    scene.line(
                        c.decisions
                            .iter()
                            .take(self.shown_decision())
                            .filter(|d| d.position < self.config.generator.sectors)
                            .map(|d| {
                                procedural::control_point(
                                    &self.config.generator,
                                    d.position,
                                    d.choice,
                                )
                            })
                            .collect(),
                        StrokeKind::Path,
                        false,
                    );
                }
            }
        }
        if let Some(t) = self.comparison.as_ref().and_then(|a| a.candidate.as_ref()) {
            if let Ok([points, _, _]) = Track::preview_geometry(&t.definition) {
                scene.line(
                    points.iter().map(|p| p.to_array()).collect(),
                    StrokeKind::Comparison,
                    true,
                );
            }
        }
        scene
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn frame(d: &mut SearchDashboard, ctx: &egui::Context, events: Vec<egui::Event>) {
        let mut out = ctx.run_ui(
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
        out.append(ctx.end_pass());
        out.textures_delta.clear();
    }
    fn click(d: &mut SearchDashboard, ctx: &egui::Context, rect: egui::Rect) {
        frame(d, ctx, vec![egui::Event::PointerMoved(rect.center())]);
        for pressed in [true, false] {
            frame(
                d,
                ctx,
                vec![
                    egui::Event::PointerMoved(rect.center()),
                    egui::Event::PointerButton {
                        pos: rect.center(),
                        button: egui::PointerButton::Primary,
                        pressed,
                        modifiers: Default::default(),
                    },
                ],
            );
        }
    }
    #[test]
    fn best_and_generation_buttons_pin_preview_against_live_events() {
        let mut archive = SearchArchive::new(
            AcoConfig {
                generator: procedural::GeneratorConfig {
                    radial_levels: vec![0.98, 1., 1.02],
                    ..procedural::GeneratorConfig::legacy()
                },
                ants: 2,
                iterations: 1,
                max_candidates: 2,
                max_selected: 2,
                minimum_diversity: 0.,
                ..Default::default()
            },
            17,
        )
        .unwrap();
        archive.run_iteration(None).unwrap();
        assert!(
            archive.iterations[0]
                .ants
                .iter()
                .all(|a| a.candidate.is_some())
        );
        let best = archive.state.selected[0].track.definition.id.clone();
        let other = archive.iterations[0]
            .ants
            .iter()
            .find(|a| a.candidate.as_ref().unwrap().definition.id != best)
            .unwrap()
            .clone();
        let mut d = SearchDashboard {
            config: archive.config.clone(),
            archive: Some(archive),
            ..Default::default()
        };
        d.view_generation();
        let ctx = egui::Context::default();
        frame(&mut d, &ctx, vec![]);
        frame(&mut d, &ctx, vec![]);
        let rect = d.buttons["Ver melhor pista"];
        click(&mut d, &ctx, rect);
        assert_eq!(d.view, AcoView::Best);
        assert_eq!(d.viewport_scene().road.as_ref().unwrap().id, best);
        let pinned = d.viewport_scene().signature();
        d.receive_event(SearchEvent::AntStarted {
            iteration: 1,
            ant: 0,
            pheromones: vec![],
        });
        d.receive_event(SearchEvent::Geometry {
            definition: other.candidate.as_ref().unwrap().definition.clone(),
            error: None,
        });
        d.receive_event(SearchEvent::Candidate(other.clone()));
        assert_eq!(
            d.viewport_scene().signature(),
            pinned,
            "live construction must not replace a pinned best/candidate"
        );
        let rect = d.buttons["Ver geração"];
        click(&mut d, &ctx, rect);
        assert_eq!(d.view, AcoView::Generation);
        assert_eq!(
            d.viewport_scene().road.as_ref().unwrap().id,
            other.candidate.as_ref().unwrap().definition.id
        );
        d.select_generation(0);
        assert!(
            !d.viewport_scene()
                .strokes
                .iter()
                .any(|s| s.kind == StrokeKind::Population)
        );
        d.show_generation_tracks = true;
        assert!(
            d.viewport_scene()
                .strokes
                .iter()
                .any(|s| s.kind == StrokeKind::Population)
        );
    }
    #[test]
    fn invalid_preview_never_produces_a_road_mesh_definition() {
        let mut archive = SearchArchive::new(
            AcoConfig {
                generator: procedural::GeneratorConfig {
                    min_length: 1.,
                    max_length: 2.,
                    ..Default::default()
                },
                ants: 1,
                iterations: 1,
                max_candidates: 1,
                ..Default::default()
            },
            17,
        )
        .unwrap();
        archive.run_iteration(None).unwrap();
        let record = archive.iterations[0].ants[0].clone();
        let mut d = SearchDashboard {
            config: archive.config.clone(),
            archive: Some(archive),
            ..Default::default()
        };
        d.select_inspection((0, 0), record);
        let scene = d.viewport_scene();
        assert!(scene.road.is_none());
        assert!(!scene.issues.is_empty());
        assert!(scene.strokes.iter().any(|s| s.kind == StrokeKind::Rejected));
    }
}
