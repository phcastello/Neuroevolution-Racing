//! Diagnostic artifacts are deliberately distinct from exportable training tracks.
use super::*;
use crate::simulation::{Track, TrackDefinition};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeometryIssue {
    pub message: String,
    pub points: Vec<[f32; 2]>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometryDiagnostic {
    pub version: u32,
    pub definition: Option<TrackDefinition>,
    pub partial: Vec<[f32; 2]>,
    pub terminal: Option<procedural::sequential::ConstructionView>,
    pub issues: Vec<GeometryIssue>,
}
impl GeometryDiagnostic {
    pub fn capture(
        definition: Option<TrackDefinition>,
        decisions: &[DecisionSnapshot],
        terminal: Option<procedural::sequential::ConstructionView>,
        reason: &str,
    ) -> Self {
        let partial = terminal
            .as_ref()
            .or_else(|| decisions.last().and_then(|d| d.construction.as_ref()))
            .map_or_else(Vec::new, |v| v.points.clone());
        let mut issues: Vec<_> = terminal
            .as_ref()
            .into_iter()
            .flat_map(|v| &v.alternatives)
            .filter_map(|a| a.issue.clone())
            .take(16)
            .collect();
        if let Some(d) = &definition {
            let [p, left, right] =
                Track::preview_geometry(d).unwrap_or_else(|_| [vec![], vec![], vec![]]);
            // Raw sampled definitions retain exact edges even when Track rejects them.
            for i in 0..p.len() {
                for j in i + 2..p.len() {
                    if i == 0 && j + 1 == p.len() {
                        continue;
                    }
                    let (a, b, c, e) = (p[i], p[(i + 1) % p.len()], p[j], p[(j + 1) % p.len()]);
                    if crate::simulation::track::segments_intersect(a, b, c, e) && issues.len() < 16
                    {
                        issues.push(GeometryIssue {
                            message: "Interseção da linha central".into(),
                            points: vec![a.to_array(), b.to_array(), c.to_array(), e.to_array()],
                        });
                    }
                }
            }
            // Offset edges can fail even when the centerline is simple.
            let borders = [left, right];
            'cross: for i in 0..p.len() {
                for j in 0..p.len() {
                    if issues.len() >= 16 {
                        break 'cross;
                    }
                    if crate::simulation::track::segments_intersect(
                        borders[0][i],
                        borders[0][(i + 1) % p.len()],
                        borders[1][j],
                        borders[1][(j + 1) % p.len()],
                    ) {
                        issues.push(GeometryIssue {
                            message: "Interseção entre bordas esquerda e direita".into(),
                            points: vec![
                                borders[0][i].to_array(),
                                borders[0][(i + 1) % p.len()].to_array(),
                                borders[1][j].to_array(),
                                borders[1][(j + 1) % p.len()].to_array(),
                            ],
                        });
                    }
                }
            }
            for side in 0..2 {
                for i in 0..p.len() {
                    for j in i + 2..p.len() {
                        if i == 0 && j + 1 == p.len() {
                            continue;
                        }
                        if crate::simulation::track::segments_intersect(
                            borders[side][i],
                            borders[side][(i + 1) % p.len()],
                            borders[side][j],
                            borders[side][(j + 1) % p.len()],
                        ) && issues.len() < 16
                        {
                            issues.push(GeometryIssue {
                                message: format!("Interseção da borda {}", side + 1),
                                points: vec![
                                    borders[side][i].to_array(),
                                    borders[side][(i + 1) % p.len()].to_array(),
                                    borders[side][j].to_array(),
                                    borders[side][(j + 1) % p.len()].to_array(),
                                ],
                            });
                        }
                    }
                }
            }
            if reason.contains("curvature") && p.len() > 2 {
                let i = (0..p.len())
                    .max_by(|i, j| {
                        let k = |n: usize| {
                            let a = p[n] - p[(n + p.len() - 1) % p.len()];
                            let b = p[(n + 1) % p.len()] - p[n];
                            2. * a.perp_dot(b).abs()
                                / (a.length() * b.length() * (a + b).length()).max(1e-6)
                        };
                        k(*i).total_cmp(&k(*j))
                    })
                    .unwrap();
                issues.push(GeometryIssue {
                    message: reason.into(),
                    points: vec![
                        p[(i + p.len() - 1) % p.len()].to_array(),
                        p[i].to_array(),
                        p[(i + 1) % p.len()].to_array(),
                    ],
                });
            }
        }
        if issues.is_empty() {
            issues.push(GeometryIssue {
                message: reason.into(),
                points: partial.last().copied().into_iter().collect(),
            });
        }
        Self {
            version: 1,
            definition,
            partial,
            terminal,
            issues,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RetentionPolicy {
    pub detailed_candidates: usize,
    pub pheromone_iterations: usize,
    pub keep_alternatives: bool,
    pub export_rejected_valid: bool,
    pub export_diagnostics: bool,
}
impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            detailed_candidates: 128,
            pheromone_iterations: 16,
            keep_alternatives: false,
            export_rejected_valid: false,
            export_diagnostics: false,
        }
    }
}
impl AntSnapshot {
    pub fn geometrically_valid(&self) -> bool {
        self.geometry_valid.unwrap_or(self.candidate.is_some())
    }
    pub fn classification(&self) -> &'static str {
        if !self.geometrically_valid() {
            "Geometria inválida / incompleta"
        } else if self.rejection.is_some() {
            "Válida · rejeitada pela seleção"
        } else {
            "Válida · aceita"
        }
    }
}
impl SearchArchive {
    pub(crate) fn apply_retention(&mut self) {
        let p = &self.config.retention;
        let mut keep = p.detailed_candidates;
        for ant in self.in_progress.iter_mut().rev() {
            if keep > 0 {
                keep -= 1;
                if !p.keep_alternatives {
                    for d in &mut ant.decisions {
                        if let Some(v) = &mut d.construction {
                            v.alternatives.clear();
                        }
                    }
                }
            } else {
                ant.geometry_valid = Some(ant.geometrically_valid());
                ant.candidate = None;
                ant.diagnostic = None;
                ant.retention_evicted = true;
                for d in &mut ant.decisions {
                    d.construction = None;
                }
            }
        }
        let iteration_count = self.iterations.len();
        for (offset, it) in self.iterations.iter_mut().rev().enumerate() {
            if offset >= p.pheromone_iterations {
                it.pheromones_before.clear();
                it.pheromones_evaporated.clear();
                it.pheromones_after.clear();
            }
            for ant in it.ants.iter_mut().rev() {
                if keep == 0 {
                    ant.geometry_valid = Some(ant.geometrically_valid());
                    ant.candidate = None;
                    ant.diagnostic = None;
                    ant.retention_evicted = true;
                    for d in &mut ant.decisions {
                        d.construction = None;
                    }
                } else {
                    keep -= 1;
                    if !p.keep_alternatives {
                        for d in &mut ant.decisions {
                            if let Some(v) = &mut d.construction {
                                v.alternatives.clear();
                            }
                        }
                    }
                }
            }
        }
        debug_assert_eq!(iteration_count, self.iterations.len());
    }
    pub fn candidate_at(&self, iteration: usize, ant: usize) -> GenerationResult<&AntSnapshot> {
        self.iterations
            .get(iteration)
            .and_then(|i| i.ants.iter().find(|a| a.ant == ant))
            .or_else(|| {
                (iteration == self.state.iteration)
                    .then(|| self.in_progress.iter().find(|a| a.ant == ant))
                    .flatten()
            })
            .ok_or_else(|| "candidate not found".into())
    }
    pub fn export_candidate(
        &self,
        iteration: usize,
        ant: usize,
        directory: impl AsRef<Path>,
    ) -> GenerationResult<()> {
        export_record(
            self.candidate_at(iteration, ant)?,
            &self.config.generator,
            self.state.seed,
            iteration,
            directory.as_ref(),
            false,
        )
    }
    pub fn export_diagnostic(
        &self,
        iteration: usize,
        ant: usize,
        directory: impl AsRef<Path>,
    ) -> GenerationResult<()> {
        export_record(
            self.candidate_at(iteration, ant)?,
            &self.config.generator,
            self.state.seed,
            iteration,
            directory.as_ref(),
            true,
        )
    }
    pub fn export_inspection(&self, root: impl AsRef<Path>) -> GenerationResult<()> {
        let root = root.as_ref();
        if root.exists() {
            return Err("inspection destination already exists".into());
        }
        std::fs::create_dir_all(root).map_err(|e| e.to_string())?;
        let mut manifest = vec![];
        for (iteration, ants) in self
            .iterations
            .iter()
            .map(|i| (i.iteration, &i.ants))
            .chain(std::iter::once((self.state.iteration, &self.in_progress)))
        {
            for ant in ants {
                manifest.push(serde_json::json!({"iteration":iteration,"ant":ant.ant,"class":ant.classification(),"reason":ant.rejection,"details_retained":!ant.retention_evicted}));
                if ant.retention_evicted {
                    continue;
                }
                let path = root.join(format!("iteration-{}-ant-{}", iteration, ant.ant));
                if ant.geometrically_valid() {
                    if ant.rejection.is_none() || self.config.retention.export_rejected_valid {
                        self.export_candidate(iteration, ant.ant, path)?;
                    }
                } else if self.config.retention.export_diagnostics {
                    self.export_diagnostic(iteration, ant.ant, path)?;
                }
            }
        }
        std::fs::write(
            root.join("manifest.json"),
            serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }
}

pub fn export_record(
    record: &AntSnapshot,
    config: &GeneratorConfig,
    seed: u64,
    iteration: usize,
    root: &Path,
    diagnostic: bool,
) -> GenerationResult<()> {
    if record.retention_evicted {
        return Err("candidate details expired under retention policy".into());
    }
    if !diagnostic && !record.geometrically_valid() {
        return Err(
            "invalid geometry: only diagnostic export is allowed, never training export".into(),
        );
    }
    let track = if diagnostic {
        None
    } else {
        Some(
            record
                .candidate
                .as_ref()
                .ok_or("complete validated track unavailable")?,
        )
    };
    if let Some(track) = track {
        track.validate()?;
    }
    if root.exists() {
        return Err("export destination already exists".into());
    }
    std::fs::create_dir_all(root).map_err(|e| e.to_string())?;
    if let Some(track) = track {
        track.save(root.join("track.generated.ron"))?;
        track.export_definition(root.join("track.ron"))?;
    }
    std::fs::write(root.join(if diagnostic {"diagnostic.json"}else{"inspection.json"}),serde_json::to_vec_pretty(&serde_json::json!({"metadata_version":1,"training_eligible":!diagnostic,"seed":seed,"iteration":iteration,"config":config,"snapshot":record})).map_err(|e|e.to_string())?).map_err(|e|e.to_string())
}
