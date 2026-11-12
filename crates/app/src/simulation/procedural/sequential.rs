//! Pose-based construction. A feasible completion witness uses the same bounded
//! straight/arc primitives as the ant; no spline is invented to join endpoints.
use super::*;
use std::collections::{HashMap, VecDeque};
pub const CLOSE: usize = 13;
pub const LEGACY_CHOICES: usize = 20;
pub const CHOICES: usize = 36;
mod organic;
pub use organic::{ConstructionPhase, ExplorationView, OrganicMetrics, organic_metrics};
pub fn label(choice: usize) -> &'static str {
    [
        "Reta curta",
        "Reta media",
        "Reta longa",
        "Esquerda 45 R*2",
        "Direita 45 R*2",
        "Esquerda 90 R*2",
        "Direita 90 R*2",
        "Hairpin esquerdo R",
        "Hairpin direito R",
        "S esquerda",
        "S direita",
        "Chicane esquerda",
        "Chicane direita",
        "Fechar: proximo segmento",
        "Esquerda 90 R*4",
        "Direita 90 R*4",
        "Esquerda 135 R*2",
        "Direita 135 R*2",
        "Curva rapida esquerda 45 R*6",
        "Curva rapida direita 45 R*6",
        "Bend esquerdo 15 R*8",
        "Bend direito 15 R*8",
        "Curva longa esquerda 60 R*6",
        "Curva longa direita 60 R*6",
        "Compound esquerda 30+45",
        "Compound direita 30+45",
        "Kink esquerdo + curva direita",
        "Kink direito + curva esquerda",
        "Deriva esquerda 15 R*16",
        "Deriva direita 15 R*16",
        "S assimetrico esquerda 30/45",
        "S assimetrico direita 30/45",
        "Curva progressiva esquerda 15/30/15",
        "Curva progressiva direita 15/30/15",
        "Bend longo esquerdo 30 R*10",
        "Bend longo direito 30 R*10",
    ][choice]
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AlternativeView {
    pub choice: usize,
    pub points: Vec<[f32; 2]>,
    pub rejection: Option<String>,
    #[serde(default)]
    pub issue: Option<crate::aco::inspection::GeometryIssue>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConstructionView {
    pub points: Vec<[f32; 2]>,
    pub selected_start: usize,
    pub alternatives: Vec<AlternativeView>,
    pub closing: bool,
    pub closed: bool,
    pub completion: Vec<[f32; 2]>,
    pub heading: f32,
    #[serde(default)]
    pub exploration: Option<ExplorationView>,
}
#[derive(Clone, Debug)]
pub enum Segment {
    Straight(f32),
    Arc(f32, i32),
    /// Fifteen-degree units; the original Arc retains its 45-degree semantics.
    Bend(f32, i32),
    /// Four gentle equal-radius arcs (+a,-a,-a,+a): same endpoint and tangent
    /// as a straight chord, with a structured S transition rather than drift noise.
    Link(f32, i32),
}
impl Segment {
    fn length(&self) -> f32 {
        match self {
            Self::Straight(l) => *l,
            Self::Arc(r, q) => r * (*q as f32).abs() * std::f32::consts::FRAC_PI_4,
            Self::Bend(r, q) => r * (*q as f32).abs() * std::f32::consts::PI / 12.,
            Self::Link(chord, q) => {
                let a = (*q as f32).abs() * std::f32::consts::PI / 12.;
                chord * a / a.sin()
            }
        }
    }
}
fn direction(h: i32) -> Vec2 {
    match h.rem_euclid(24) {
        0 => Vec2::X,
        6 => Vec2::Y,
        12 => Vec2::NEG_X,
        18 => Vec2::NEG_Y,
        _ if h % 3 == 0 => Vec2::from_angle((h / 3) as f32 * std::f32::consts::FRAC_PI_4),
        _ => Vec2::from_angle(h as f32 * std::f32::consts::PI / 12.),
    }
}
#[derive(Clone, Debug)]
struct PathState {
    points: Vec<Vec2>,
    distances: Vec<f32>,
    cells: HashMap<(i32, i32), Vec<usize>>,
    position: Vec2,
    heading: i32,
    winding: i32,
    length: f32,
    clearance: f32,
    minimum_radius: f32,
    issue: Option<crate::aco::inspection::GeometryIssue>,
}
impl PathState {
    fn new(config: &GeneratorConfig) -> Self {
        let clearance =
            config.widths.iter().copied().fold(0., f32::max) + config.minimum_border_gap + 5.;
        Self {
            points: vec![Vec2::ZERO],
            distances: vec![0.],
            cells: HashMap::from([((0, 0), vec![0])]),
            position: Vec2::ZERO,
            heading: 0,
            winding: 0,
            length: 0.,
            clearance,
            minimum_radius: (1. / config.max_curvature).max(
                config.widths.iter().copied().fold(0., f32::max) * 0.5 + CAR_LENGTH * 0.5 + 3.,
            ),
            issue: None,
        }
    }
    fn closed(&self) -> bool {
        self.points.len() > 8
            && self.position.length() < 0.1
            && self.heading == 0
            && self.winding.abs() == 24
    }
    fn append(&mut self, segment: &Segment, seam: Option<f32>) -> GenerationResult<()> {
        if let Segment::Link(chord, q) = segment {
            let angle = (*q as f32).abs() * std::f32::consts::PI / 12.;
            let radius = *chord / (4. * angle.sin());
            for turn in [*q, -*q, -*q, *q] {
                self.append(&Segment::Bend(radius, turn), seam)?;
            }
            return Ok(());
        }
        if matches!(segment, Segment::Arc(radius,_) | Segment::Bend(radius,_) if *radius+0.001<self.minimum_radius)
        {
            self.issue = Some(crate::aco::inspection::GeometryIssue {
                message: format!(
                    "Raio abaixo do mínimo {:.2} para curvatura/largura",
                    self.minimum_radius
                ),
                points: vec![self.position.to_array()],
            });
            return Err("curvature / road-width limit excludes this primitive".into());
        }
        let begin = self.position;
        let f = direction(self.heading);
        let left = f.perp();
        let steps = (segment.length() / 10.).ceil().max(1.) as usize;
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let mut p = match segment {
                Segment::Straight(l) => begin + f * (*l * t),
                Segment::Arc(r, q) => {
                    let angle = (*q as f32).abs() * std::f32::consts::FRAC_PI_4 * t;
                    begin
                        + f * (*r * angle.sin())
                        + left * (*r * (q.signum() as f32) * (1. - angle.cos()))
                }
                Segment::Bend(r, q) => {
                    let angle = (*q as f32).abs() * std::f32::consts::PI / 12. * t;
                    begin
                        + f * (*r * angle.sin())
                        + left * (*r * q.signum() as f32 * (1. - angle.cos()))
                }
                Segment::Link(..) => unreachable!(),
            };
            if i == steps {
                p = (p * 1000.).round() / 1000.;
            }
            let distance = self.length + self.position.distance(p);
            let key = (
                (p.x / self.clearance).floor() as i32,
                (p.y / self.clearance).floor() as i32,
            );
            for x in key.0 - 1..=key.0 + 1 {
                for y in key.1 - 1..=key.1 + 1 {
                    if let Some(indices) = self.cells.get(&(x, y)) {
                        for j in indices {
                            let arc_gap = distance - self.distances[*j];
                            if arc_gap < self.clearance * 1.7
                                || seam.is_some_and(|total| total - arc_gap < self.clearance * 1.7)
                            {
                                continue;
                            }
                            if self.points[*j].distance(p) < self.clearance {
                                self.issue = Some(crate::aco::inspection::GeometryIssue {
                                    message: format!(
                                        "Corredores a {:.2}; separação mínima {:.2}",
                                        self.points[*j].distance(p),
                                        self.clearance
                                    ),
                                    points: vec![self.points[*j].to_array(), p.to_array()],
                                });
                                return Err(
                                    "segment corridor collision / insufficient border gap".into()
                                );
                            }
                        }
                    }
                }
            }
            self.points.push(p);
            self.distances.push(distance);
            self.cells
                .entry(key)
                .or_default()
                .push(self.points.len() - 1);
            self.position = p;
            self.length = distance;
        }
        if let Segment::Arc(_, q) = segment {
            self.heading = (self.heading + 3 * q).rem_euclid(24);
            self.winding += 3 * q;
        } else if let Segment::Bend(_, q) = segment {
            self.heading = (self.heading + q).rem_euclid(24);
            self.winding += q;
        }
        Ok(())
    }
}
fn primitive(config: &GeneratorConfig, choice: usize) -> Vec<Segment> {
    let r = config.primitive_radius;
    match choice {
        0 => vec![Segment::Straight(2. * r)],
        1 => vec![Segment::Straight(6. * r)],
        2 => vec![Segment::Straight(16. * r)],
        3 => vec![Segment::Arc(2. * r, 1)],
        4 => vec![Segment::Arc(2. * r, -1)],
        5 => vec![Segment::Arc(2. * r, 2)],
        6 => vec![Segment::Arc(2. * r, -2)],
        7 => vec![Segment::Arc(r, 4)],
        8 => vec![Segment::Arc(r, -4)],
        9 => vec![Segment::Arc(3. * r, 1), Segment::Arc(3. * r, -1)],
        10 => vec![Segment::Arc(3. * r, -1), Segment::Arc(3. * r, 1)],
        11 => vec![
            Segment::Arc(2. * r, 1),
            Segment::Arc(2. * r, -2),
            Segment::Arc(2. * r, 1),
        ],
        12 => vec![
            Segment::Arc(2. * r, -1),
            Segment::Arc(2. * r, 2),
            Segment::Arc(2. * r, -1),
        ],
        14 => vec![Segment::Arc(4. * r, 2)],
        15 => vec![Segment::Arc(4. * r, -2)],
        16 => vec![Segment::Arc(2. * r, 3)],
        17 => vec![Segment::Arc(2. * r, -3)],
        18 => vec![Segment::Arc(6. * r, 1)],
        19 => vec![Segment::Arc(6. * r, -1)],
        20 | 21 => vec![Segment::Bend(8. * r, if choice == 20 { 1 } else { -1 })],
        22 | 23 => vec![Segment::Bend(6. * r, if choice == 22 { 4 } else { -4 })],
        24 | 25 => {
            let s = if choice == 24 { 1 } else { -1 };
            vec![Segment::Bend(5. * r, 2 * s), Segment::Arc(2. * r, s)]
        }
        26 | 27 => {
            let s = if choice == 26 { 1 } else { -1 };
            vec![
                Segment::Bend(6. * r, s),
                Segment::Straight(2. * r),
                Segment::Bend(3. * r, -4 * s),
            ]
        }
        28 | 29 => vec![Segment::Bend(16. * r, if choice == 28 { 1 } else { -1 })],
        30 | 31 => {
            let s = if choice == 30 { 1 } else { -1 };
            vec![Segment::Bend(5. * r, 2 * s), Segment::Bend(3. * r, -3 * s)]
        }
        32 | 33 => {
            let s = if choice == 32 { 1 } else { -1 };
            vec![
                Segment::Bend(8. * r, s),
                Segment::Bend(4. * r, 2 * s),
                Segment::Bend(7. * r, s),
            ]
        }
        34 | 35 => vec![Segment::Bend(10. * r, if choice == 34 { 2 } else { -2 })],
        _ => vec![],
    }
}
fn end_pose(mut p: Vec2, mut h: i32, segment: &Segment) -> (Vec2, i32) {
    let f = direction(h);
    match segment {
        Segment::Straight(l) => p += f * *l,
        Segment::Arc(r, q) => {
            let angle = (*q as f32).abs() * std::f32::consts::FRAC_PI_4;
            p += f * (*r * angle.sin()) + f.perp() * (*r * q.signum() as f32 * (1. - angle.cos()));
            h = (h + 3 * q).rem_euclid(24);
        }
        Segment::Bend(r, q) => {
            let angle = (*q as f32).abs() * std::f32::consts::PI / 12.;
            p += f * (*r * angle.sin()) + f.perp() * (*r * q.signum() as f32 * (1. - angle.cos()));
            h = (h + q).rem_euclid(24);
        }
        Segment::Link(chord, _) => p += f * *chord,
    }
    (p, h)
}
/// Solve the two displacement equations for a finite cardinal arc/straight
/// completion. Heading is solved exactly by the number/sign of quarter turns.
/// Added opposite-direction distances cancel algebraically, creating detours
/// around occupied corridors. Every proposed full completion is collision tested.
fn completion(
    path: &PathState,
    config: &GeneratorConfig,
    remaining: usize,
    check: &mut dyn FnMut() -> GenerationResult<()>,
) -> GenerationResult<Vec<Segment>> {
    if config.organic {
        return organic::completion(path, config, remaining, check);
    }
    if path.closed() {
        return Ok(vec![]);
    }
    let mut trials = 0;
    for radius in [
        2. * config.primitive_radius,
        config.primitive_radius,
        4. * config.primitive_radius,
    ] {
        for sign in [1, -1] {
            for turns in 1..=4 {
                let mut p = path.position;
                let mut h = path.heading;
                let mut dirs = vec![direction(h)];
                for _ in 0..turns {
                    (p, h) = end_pose(p, h, &Segment::Arc(radius, 2 * sign));
                    dirs.push(direction(h));
                }
                let final_turn = (-h / 3 + 4).rem_euclid(8) - 4;
                // A simple closed C1 circuit has turning number +1 or -1.
                // Reject impossible winding before any corridor trials.
                if (path.winding + turns * 6 * sign + 3 * final_turn).abs() != 24 {
                    continue;
                }
                if final_turn != 0 {
                    p = end_pose(p, h, &Segment::Arc(radius, final_turn)).0;
                }
                let axes = [direction(path.heading), direction(path.heading).perp()];
                for extra_x in [0., 6., 12., 24.] {
                    for extra_y in [0., 6., 12., 24.] {
                        for last in [true, false] {
                            trials += 1;
                            if trials % 8 == 0 {
                                check()?;
                            }
                            let mut lengths = vec![0.; dirs.len()];
                            let mut possible = true;
                            for axis in 0..2 {
                                let need = -p.dot(axes[axis]);
                                let extra = if axis == 0 { extra_x } else { extra_y }
                                    * config.primitive_radius;
                                for s in [-1., 1.] {
                                    let amount = if need * s > 0. {
                                        need.abs() + extra
                                    } else {
                                        extra
                                    };
                                    if amount < 0.05 {
                                        continue;
                                    }
                                    let index = if last {
                                        dirs.iter().rposition(|d| d.dot(axes[axis]) * s > 0.5)
                                    } else {
                                        dirs.iter().position(|d| d.dot(axes[axis]) * s > 0.5)
                                    };
                                    if let Some(index) = index {
                                        lengths[index] += amount;
                                    } else {
                                        possible = false;
                                    }
                                }
                            }
                            if !possible {
                                continue;
                            }
                            let mut plan = vec![];
                            for (i, length) in lengths.iter().enumerate() {
                                if *length > 0.05 {
                                    plan.push(Segment::Straight(*length));
                                }
                                if i < turns as usize {
                                    plan.push(Segment::Arc(radius, 2 * sign));
                                }
                            }
                            if final_turn != 0 {
                                plan.push(Segment::Arc(radius, final_turn));
                            }
                            if plan.len() > remaining {
                                continue;
                            }
                            let total = path.length + plan.iter().map(Segment::length).sum::<f32>();
                            if total < config.min_length || total > config.max_length {
                                continue;
                            }
                            let mut candidate = path.clone();
                            let mut valid = true;
                            for segment in &plan {
                                if candidate.append(segment, Some(total)).is_err() {
                                    valid = false;
                                    break;
                                }
                            }
                            if valid && candidate.closed() {
                                return Ok(plan);
                            }
                        }
                    }
                }
            }
        }
    }
    Err(
        "no collision-free tangent-continuous closure within remaining segment/length budget"
            .into(),
    )
}
#[derive(Clone, Debug)]
pub struct Construction {
    path: PathState,
    closing: VecDeque<Segment>,
    pub used: usize,
    history: Vec<usize>,
    exploration: Option<ExplorationView>,
    witness: Option<Vec<Segment>>,
}
pub struct Alternative {
    pub next: Option<Construction>,
    pub view: AlternativeView,
    pub eta: f64,
}
impl Construction {
    pub fn new(config: &GeneratorConfig) -> Self {
        Self {
            path: PathState::new(config),
            closing: VecDeque::new(),
            used: 0,
            history: vec![],
            exploration: None,
            witness: None,
        }
    }
    pub fn closed(&self) -> bool {
        self.path.closed()
    }
    pub fn points(&self) -> Vec<[f32; 2]> {
        self.path.points.iter().map(|p| p.to_array()).collect()
    }
    pub fn options(
        &self,
        config: &GeneratorConfig,
        check: &mut dyn FnMut() -> GenerationResult<()>,
    ) -> GenerationResult<Vec<Alternative>> {
        let remaining = config.sectors.saturating_sub(self.used);
        let mut options = vec![];
        let diagnostics = config.organic.then(|| self.exploration_status(config));
        let counts = if config.organic {
            CHOICES
        } else {
            LEGACY_CHOICES
        };
        for choice in 0..counts {
            check()?;
            let mut next = self.clone();
            let before = next.path.points.len();
            let result = (|| {
                if remaining == 0 {
                    return Err("segment budget exhausted".into());
                }
                if !self.closing.is_empty() && choice != CLOSE {
                    return Err("closure committed; follow its verified primitives".into());
                }
                if choice == CLOSE {
                    if next.closing.is_empty() {
                        next.closing = if let Some(plan) = &self.witness {
                            plan.clone()
                        } else {
                            completion(&self.path, config, remaining, check)?
                        }
                        .into();
                    }
                    next.witness = None;
                    let total =
                        next.path.length + next.closing.iter().map(Segment::length).sum::<f32>();
                    let segment = next.closing.pop_front().ok_or("closure already complete")?;
                    next.path.append(&segment, Some(total))?;
                } else {
                    for segment in primitive(config, choice) {
                        next.path.append(&segment, None)?;
                    }
                    if next.path.length > config.max_length {
                        return Err("length budget exhausted".into());
                    }
                    // A witness must exist BEFORE this expansion is made available.
                    next.witness = Some(completion(&next.path, config, remaining - 1, check)?);
                }
                next.used += 1;
                next.history.push(choice);
                Ok::<_, String>(())
            })();
            let rejection = result.err();
            let points = next.path.points[before.saturating_sub(1)..]
                .iter()
                .map(|p| p.to_array())
                .collect();
            let mut eta = if choice == CLOSE {
                0.05 + 3. * (self.used as f64 / config.sectors as f64).powi(3)
            } else {
                let repeats = self.history.iter().filter(|c| **c == choice).count();
                ((if matches!(choice, 18 | 19) {
                    2.0
                } else if matches!(choice, 7 | 8) {
                    0.8
                } else {
                    1.5
                }) / (1. + repeats as f64))
                    * if self.history.last() == Some(&choice) {
                        0.55
                    } else {
                        1.
                    }
            };
            if let Some(status) = &diagnostics {
                eta = if choice == CLOSE {
                    status.closure_eta
                } else {
                    let family = |c: usize| match c {
                        0..=2 => 0,
                        7 | 8 => 1,
                        9..=12 | 30 | 31 => 2,
                        24..=27 | 32 | 33 => 3,
                        20..=23 | 28 | 29 | 34 | 35 => 4,
                        _ => 5,
                    };
                    let repeats = self
                        .history
                        .iter()
                        .filter(|c| family(**c) == family(choice))
                        .count();
                    let novelty = if repeats == 0 { 1.65 } else { 1. };
                    let length = next.path.length - self.path.length;
                    let net_heading =
                        ((next.path.heading - self.path.heading + 12).rem_euclid(24) - 12).abs();
                    let previous_straights =
                        self.history.iter().rev().take_while(|c| **c <= 2).count();
                    let straight_penalty = if choice <= 2 {
                        0.65 / (1. + previous_straights as f64 * 2.)
                    } else {
                        1.
                    };
                    // Prefer bends and compounds without suppressing R50 or long straights.
                    let bend_bonus = if choice >= 20 { 1.4 } else { 1. };
                    let heading_bonus = if net_heading % 6 != 0 { 1.3 } else { 1. };
                    let distance = next.path.position.length();
                    let expansion = if status.phase == ConstructionPhase::FreeExploration
                        && distance > self.path.position.length() + config.primitive_radius
                    {
                        1.2
                    } else {
                        1.
                    };
                    let return_ratio = next
                        .witness
                        .as_ref()
                        .map_or(0., |p| p.iter().map(Segment::length).sum::<f32>())
                        / next.path.length.max(config.min_length);
                    let return_bias = if status.phase == ConstructionPhase::FreeExploration {
                        1. / (1. + 0.5 * (return_ratio as f64).powi(2))
                    } else {
                        1. / (1. + 4. * (return_ratio as f64).powi(3))
                    };
                    eta * return_bias
                        * novelty
                        * straight_penalty
                        * bend_bonus
                        * heading_bonus
                        * expansion
                        / (1. + 0.18 * repeats as f64)
                        / (1. + (length as f64 / 1800.).powi(2))
                };
                let mut view = status.clone();
                view.chose_extra_exploration = choice != CLOSE;
                next.exploration = Some(view);
            }
            let issue = next.path.issue.clone();
            options.push(Alternative {
                next: if rejection.is_none() {
                    Some(next)
                } else {
                    None
                },
                view: AlternativeView {
                    choice,
                    points,
                    rejection,
                    issue,
                },
                eta,
            });
        }
        if config.organic {
            let close_viable = options[CLOSE].next.is_some();
            let expanding = options
                .iter()
                .enumerate()
                .any(|(i, a)| i != CLOSE && a.next.is_some());
            for (choice, alternative) in options.iter_mut().enumerate() {
                if let Some(view) = alternative
                    .next
                    .as_mut()
                    .and_then(|n| n.exploration.as_mut())
                {
                    view.chose_extra_exploration =
                        choice != CLOSE && close_viable && self.closing.is_empty();
                    if !expanding && self.closing.is_empty() {
                        view.reason =
                            "Única continuação viável: preservar fechamento, tangência e folga"
                                .into();
                    }
                    if choice == CLOSE {
                        view.phase = ConstructionPhase::Closure;
                    }
                }
            }
        }
        Ok(options)
    }
    fn exploration_status(&self, config: &GeneratorConfig) -> ExplorationView {
        let m = organic_metrics(&self.path.points, false);
        let remaining = config.sectors.saturating_sub(self.used);
        let target = config.sectors.saturating_sub(6).max(1);
        let free_until = (target as f32 * 0.65).ceil() as usize;
        let phase = if !self.closing.is_empty() || self.history.last() == Some(&CLOSE) {
            ConstructionPhase::Closure
        } else if self.used < free_until {
            ConstructionPhase::FreeExploration
        } else if self.used < target {
            ConstructionPhase::GuidedExploration
        } else {
            ConstructionPhase::Closure
        };
        let desired_length = config
            .min_length
            .max(config.primitive_radius * config.sectors as f32 * 3.);
        let readiness = (self.path.length / desired_length).min(1.)
            * (m.geometric_events as f32 / 10.).min(1.)
            * (m.element_families as f32 / 3.).min(1.);
        let eta = match phase {
            ConstructionPhase::FreeExploration => 0.0001,
            ConstructionPhase::GuidedExploration => 0.012 * readiness as f64,
            ConstructionPhase::Closure => {
                (2. + (self.used.saturating_sub(target) * 3) as f64)
                    * (0.15 + 0.85 * readiness as f64)
            }
        };
        let reason = if !self.closing.is_empty() {
            "Executando continuação já verificada".to_string()
        } else if phase == ConstructionPhase::FreeExploration {
            format!(
                "Passear: {} de {free_until} decisões livres; retorno com pressão mínima",
                self.used
            )
        } else if readiness < 0.95 {
            format!(
                "Passear: riqueza insuficiente ({}/{:.0} unidades, {} eventos, {} famílias); reserva {remaining}",
                self.path.length, desired_length, m.geometric_events, m.element_families
            )
        } else if phase == ConstructionPhase::GuidedExploration {
            format!(
                "Passear com retorno viável: diversificar até {target} decisões; reserva {remaining}"
            )
        } else {
            format!("Fechamento favorecido: exploração cumprida; reserva de {remaining} decisões")
        };
        ExplorationView {
            phase,
            closure_eta: eta.max(1e-6),
            reason,
            chose_extra_exploration: false,
            exploration_decisions: self.history.iter().filter(|c| **c != CLOSE).count(),
            length: self.path.length,
            events: m.geometric_events,
            families: m.element_families,
            remaining_decisions: remaining,
        }
    }
    pub fn view(&self, start: usize, alternatives: Vec<AlternativeView>) -> ConstructionView {
        let mut witness = self.path.clone();
        let total = witness.length + self.closing.iter().map(Segment::length).sum::<f32>();
        for segment in &self.closing {
            let _ = witness.append(segment, Some(total));
        }
        ConstructionView {
            points: self.points(),
            selected_start: start.saturating_sub(1),
            alternatives,
            closing: !self.closing.is_empty() || self.history.last() == Some(&CLOSE),
            closed: self.closed(),
            completion: witness.points[self.path.points.len().saturating_sub(1)..]
                .iter()
                .map(|p| p.to_array())
                .collect(),
            heading: self.path.heading as f32 * std::f32::consts::PI / 12.,
            exploration: self.exploration.clone(),
        }
    }
}
pub fn reconstruct(config: &GeneratorConfig, choices: &[usize]) -> GenerationResult<Construction> {
    let mut state = Construction::new(config);
    // Replay applies the recorded choices, never samples a second RNG stream.
    for choice in choices {
        if *choice
            >= if config.organic {
                CHOICES
            } else {
                LEGACY_CHOICES
            }
            || state.closed()
        {
            return Err("invalid sequential decision after closure".into());
        }
        if *choice == CLOSE {
            if state.closing.is_empty() {
                state.closing = completion(
                    &state.path,
                    config,
                    config.sectors - state.used,
                    &mut || Ok(()),
                )?
                .into();
            }
            let total = state.path.length + state.closing.iter().map(Segment::length).sum::<f32>();
            let segment = state.closing.pop_front().ok_or("empty closure")?;
            state.path.append(&segment, Some(total))?;
        } else {
            if !state.closing.is_empty() {
                return Err("recorded expansion interrupts closure".into());
            }
            for segment in primitive(config, *choice) {
                state.path.append(&segment, None)?;
            }
        }
        state.used += 1;
        state.history.push(*choice);
    }
    Ok(state)
}
pub fn definition(
    config: &GeneratorConfig,
    decisions: &[usize],
) -> GenerationResult<TrackDefinition> {
    if !config.decisions_valid(decisions) {
        return Err("sequential decision shape out of bounds".into());
    }
    let state = reconstruct(config, &decisions[..decisions.len() - 1])?;
    if !state.closed() {
        return Err("circuit did not reach initial position AND tangent".into());
    }
    let mut points = state.points();
    points.pop();
    let width = config.widths[*decisions.last().unwrap()];
    let hash = super::geometry_hash(&points, width, 1);
    Ok(TrackDefinition {
        sampled_centerline: true,
        id: format!("generated_{hash}"),
        name: format!("Sequential {hash}"),
        country: None,
        category: TrackCategory::PermanentCircuit,
        role: TrackRole::Training,
        difficulty: TrackDifficulty::MediumHard,
        width,
        samples_per_control_point: 1,
        start_index: 0,
        control_points: points,
        description: Some(
            "Sequential straight/arc construction with verified closure witness; training only"
                .into(),
        ),
        approximation_note: Some(
            "C1 analytic primitives sampled at <=10 world units; no arbitrary endpoint spline"
                .into(),
        ),
    })
}
pub fn generate(config: &GeneratorConfig, seed: u64) -> GenerationResult<GeneratedTrack> {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let mut last = String::new();
    for attempt in 0..config.max_attempts {
        let mut state = Construction::new(config);
        let mut decisions = vec![];
        while !state.closed() && state.used < config.sectors {
            let mut options = state.options(config, &mut || Ok(()))?;
            let viable: Vec<_> = options
                .iter()
                .enumerate()
                .filter(|(_, a)| a.next.is_some())
                .map(|(i, _)| i)
                .collect();
            if viable.is_empty() {
                break;
            }
            let choice = if config.organic {
                let total = viable.iter().map(|i| options[*i].eta).sum::<f64>();
                let mut sample = rng.random::<f64>() * total;
                *viable
                    .iter()
                    .find(|i| {
                        sample -= options[**i].eta;
                        sample < 0.
                    })
                    .unwrap_or(viable.last().unwrap())
            } else {
                viable[rng.random_range(0..viable.len())]
            };
            state = options[choice].next.take().unwrap();
            decisions.push(choice);
        }
        if !state.closed() {
            last = "closure budget exhausted".into();
            continue;
        }
        decisions.push(rng.random_range(0..config.widths.len()));
        match super::from_decisions(config, seed, attempt, &decisions) {
            Ok(track) => return Ok(track),
            Err(e) => last = e,
        }
    }
    Err(format!(
        "sequential generation exhausted {} attempts: {last}",
        config.max_attempts
    ))
}
/// Arc-length-weighted descriptors, not hard requirements on every candidate.
pub fn characterize(track: &Track, m: &mut GeometryMetrics) {
    let n = track.samples.len();
    let mut ks = vec![];
    let mut ds = vec![];
    for i in 0..n {
        let a = track.samples[(i + n - 1) % n].position;
        let b = track.samples[i].position;
        let c = track.samples[(i + 1) % n].position;
        let ab = b - a;
        let bc = c - b;
        let den = ab.length() * bc.length() * c.distance(a);
        ks.push(2. * ab.perp_dot(bc) / den.max(1e-6));
        ds.push(b.distance(c));
    }
    let mut straight = 0.;
    let mut longest = 0_f32;
    let mut total_straight = 0.;
    let mut radii = vec![];
    let mut groups: Vec<(i32, f32, f32, f32)> = vec![];
    let mut gap = 0.;
    for i in 0..n * 2 {
        let j = i % n;
        let k = ks[j];
        let d = ds[j];
        if k.abs() < 0.0004 {
            straight += d;
            if i < n {
                total_straight += d;
            }
            longest = longest.max(straight.min(track.total_length));
        } else {
            straight = 0.;
        }
        if i >= n {
            continue;
        }
        if k.abs() > 0.001 {
            radii.push(1. / k.abs());
            let sign = if k > 0. { 1 } else { -1 };
            if groups.last().is_none_or(|g| g.0 != sign) || gap > 40. {
                groups.push((sign, 0., 0., gap));
            }
            let g = groups.last_mut().unwrap();
            g.1 += k * d;
            g.2 += d;
            gap = 0.;
        } else {
            gap += d;
        }
    }
    radii.sort_by(f32::total_cmp);
    m.longest_straight = longest;
    m.straight_fraction = total_straight / track.total_length;
    m.turns = groups.len();
    if !radii.is_empty() {
        m.radius_quantiles = [
            radii[radii.len() / 10],
            radii[radii.len() / 2],
            radii[radii.len() * 9 / 10],
        ];
    }
    m.curvature_variation = (0..n).map(|i| (ks[(i + 1) % n] - ks[i]).abs()).sum();
    m.hairpins = groups.iter().filter(|g| g.1.abs() > 2.6).count();
    m.s_bends = groups
        .windows(2)
        .filter(|g| g[0].0 != g[1].0 && g[1].3 < 250. && g[0].1.abs() > 0.4 && g[1].1.abs() > 0.4)
        .count();
    m.chicanes = groups
        .windows(3)
        .filter(|g| {
            g[0].0 == g[2].0
                && g[0].0 != g[1].0
                && g[1].3 < 200.
                && g[2].3 < 200.
                && (g[0].1 + g[1].1 + g[2].1).abs() < 0.4
                && g.iter().all(|x| x.1.abs() > 0.6)
        })
        .count();
    let area = (0..n)
        .map(|i| {
            track.samples[i]
                .position
                .perp_dot(track.samples[(i + 1) % n].position)
        })
        .sum::<f32>()
        .abs()
        * 0.5;
    m.compactness = 4. * std::f32::consts::PI * area / (track.total_length * track.total_length);
    // Eight normalized geometric descriptors plus a 16-point aligned shape signature.
    m.descriptor = vec![
        track.total_length,
        m.longest_straight,
        m.straight_fraction,
        m.turns as f32,
        m.hairpins as f32,
        m.chicanes as f32,
        m.s_bends as f32,
        m.compactness,
    ];
    let bounds = track.bounds();
    let scale = (bounds.max - bounds.min).max_element().max(1.);
    let center = (bounds.max + bounds.min) * 0.5;
    for i in 0..16 {
        let p = (track.point_at_distance(track.total_length * i as f32 / 16.) - center) / scale;
        m.descriptor.extend([p.x, p.y]);
    }
}
pub fn descriptor_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != 40 || b.len() != 40 {
        return 1.;
    }
    let metric = (0..8)
        .map(|i| {
            (a[i] - b[i]).abs()
                / a[i]
                    .abs()
                    .max(b[i].abs())
                    .max(if i == 2 || i == 7 { 0.2 } else { 1. })
        })
        .sum::<f32>()
        / 8.;
    let mut shape = f32::INFINITY;
    for shift in 0..16 {
        for rotation in 0..4 {
            for mirror in [-1., 1.] {
                let mut error = 0.;
                for i in 0..16 {
                    let x = b[8 + 2 * ((i + shift) % 16)];
                    let y = b[9 + 2 * ((i + shift) % 16)] * mirror;
                    let (x, y) = match rotation {
                        0 => (x, y),
                        1 => (-y, x),
                        2 => (-x, -y),
                        _ => (y, -x),
                    };
                    error += (a[8 + 2 * i] - x).powi(2) + (a[9 + 2 * i] - y).powi(2);
                }
                shape = shape.min((error / 16.).sqrt());
            }
        }
    }
    let signature = shape;
    (0.55 * metric + 0.65 * signature).clamp(0., 1.)
}
#[cfg(test)]
#[path = "sequential_validation.rs"]
mod validation;
