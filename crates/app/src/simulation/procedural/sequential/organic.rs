//! Structured exploration and rotation-invariant shape diagnostics.
use super::*;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct OrganicMetrics {
    pub longest_straight_ratio: f32,
    pub straight_lengths: Vec<f32>,
    pub accumulated_turn: f32,
    pub heading_histogram: Vec<f32>,
    pub heading_entropy: f32,
    /// Magnitude of the fourth circular moment: 1 for two orthogonal axes.
    pub axis_concentration: f32,
    pub bounding_box_fill: f32,
    pub bounding_box_aspect: f32,
    /// Box fill times fourth-moment concentration. A circle is not a rectangle.
    pub rectangularity: f32,
    pub orthogonal_turn_fraction: f32,
    pub geometric_events: usize,
    pub element_families: usize,
    pub spatial_cells: usize,
    pub complexity: f32,
}

/// Uses the actual sampled centerline, including the seam only for closed paths.
/// All averages are distance weighted; no sample-count or world-axis bias.
pub fn organic_metrics(points: &[Vec2], closed: bool) -> OrganicMetrics {
    let mut m = OrganicMetrics::default();
    if points.len() < 3 {
        return m;
    }
    let n = points.len();
    let count = if closed { n } else { n - 1 };
    let mut histogram = vec![0_f32; 24];
    let mut moment = Vec2::ZERO;
    let mut length = 0.;
    let mut area = 0_f32;
    let mut straights = vec![0_f32];
    let mut groups: Vec<(i32, f32)> = vec![];
    let mut previous_class = 0;
    let mut event_length = 0.;
    let mut families = std::collections::BTreeSet::new();
    let mut cells = std::collections::BTreeSet::new();
    for i in 0..count {
        let a = points[i];
        let b = points[(i + 1) % n];
        let edge = b - a;
        let d = edge.length();
        if d < 1e-4 {
            continue;
        }
        length += d;
        area += a.perp_dot(b);
        let h = edge.to_angle();
        moment += Vec2::from_angle(4. * h) * d;
        // Linear deposition avoids bin-edge jumps on rotated copies.
        let bin = h.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU * 24.;
        let j = bin.floor() as usize % 24;
        histogram[j] += d * (1. - bin.fract());
        histogram[(j + 1) % 24] += d * bin.fract();
        let k = if !closed && i == 0 {
            0.
        } else {
            let prev = a - points[(i + n - 1) % n];
            2. * prev.perp_dot(edge)
                / (prev.length() * d * (b - points[(i + n - 1) % n]).length()).max(1e-6)
        };
        m.accumulated_turn += k.abs() * d;
        let class = if k.abs() < 0.0004 {
            0
        } else {
            k.signum() as i32
                * if k.abs() > 0.012 {
                    3
                } else if k.abs() > 0.0045 {
                    2
                } else {
                    1
                }
        };
        families.insert(class.abs());
        if class == 0 {
            *straights.last_mut().unwrap() += d;
        } else if *straights.last().unwrap() > 0. {
            straights.push(0.);
        }
        if class != previous_class {
            if event_length >= 30. {
                m.geometric_events += 1;
            }
            previous_class = class;
            event_length = 0.;
        }
        event_length += d;
        if class != 0 {
            let s = class.signum();
            if groups.last().is_none_or(|g| g.0 != s) {
                groups.push((s, 0.));
            }
            groups.last_mut().unwrap().1 += k.abs() * d;
        } else if d > 0. && groups.last().is_some_and(|g| g.0 != 0) {
            groups.push((0, 0.));
        }
    }
    if event_length >= 30. {
        m.geometric_events += 1;
    }
    if closed && straights.len() > 1 {
        let tail = straights.pop().unwrap();
        straights[0] += tail;
    }
    straights.retain(|l| *l > 20.);
    m.longest_straight_ratio = straights.iter().copied().fold(0., f32::max) / length.max(1.);
    m.straight_lengths = straights;
    m.axis_concentration = moment.length() / length.max(1.);
    m.heading_histogram = histogram.iter().map(|x| x / length.max(1.)).collect();
    m.heading_entropy = -m
        .heading_histogram
        .iter()
        .filter(|p| **p > 0.)
        .map(|p| p * p.ln())
        .sum::<f32>()
        / 24_f32.ln();
    let turns: Vec<_> = groups.iter().filter(|g| g.0 != 0 && g.1 > 0.15).collect();
    m.orthogonal_turn_fraction = turns
        .iter()
        .filter(|g| (g.1 - std::f32::consts::FRAC_PI_2).abs() < 0.16)
        .count() as f32
        / turns.len().max(1) as f32;
    let mut box_area = f32::INFINITY;
    let mut extent = Vec2::ONE;
    let origin = points[0];
    for j in 0..24 {
        let axis = Vec2::from_angle(j as f32 * std::f32::consts::FRAC_PI_2 / 24.);
        let mut lo = Vec2::splat(f32::INFINITY);
        let mut hi = Vec2::splat(f32::NEG_INFINITY);
        for p in points {
            let q = Vec2::new((*p - origin).dot(axis), (*p - origin).dot(axis.perp()));
            lo = lo.min(q);
            hi = hi.max(q);
        }
        let e = hi - lo;
        if e.x * e.y < box_area {
            box_area = e.x * e.y;
            extent = e;
        }
    }
    m.bounding_box_fill = if closed {
        (area.abs() * 0.5 / box_area.max(1.)).clamp(0., 1.)
    } else {
        0.
    };
    m.bounding_box_aspect = extent.max_element() / extent.min_element().max(1.);
    m.rectangularity = m.bounding_box_fill * m.axis_concentration;
    // A scale-normalized occupancy diagnostic, not an objective on its own.
    let cell_size = length / 24.;
    for p in points {
        cells.insert((
            ((p.x - origin.x) / cell_size.max(1.)).floor() as i32,
            ((p.y - origin.y) / cell_size.max(1.)).floor() as i32,
        ));
    }
    m.spatial_cells = cells.len();
    m.element_families = families.len();
    m.complexity = (0.30 * (m.geometric_events as f32 / 18.).min(1.)
        + 0.25 * ((m.accumulated_turn / std::f32::consts::TAU - 1.) / 2.).clamp(0., 1.)
        + 0.25 * m.heading_entropy
        + 0.20 * (1. - m.axis_concentration))
        .clamp(0., 1.);
    m
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstructionPhase {
    #[default]
    FreeExploration,
    GuidedExploration,
    Closure,
}
impl ConstructionPhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::FreeExploration => "Exploração livre",
            Self::GuidedExploration => "Exploração orientada",
            Self::Closure => "Fechamento",
        }
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExplorationView {
    pub phase: ConstructionPhase,
    pub closure_eta: f64,
    pub reason: String,
    pub chose_extra_exploration: bool,
    pub exploration_decisions: usize,
    pub length: f32,
    pub events: usize,
    pub families: usize,
    pub remaining_decisions: usize,
}

/// A finite family of asymmetric arc sequences. Two straight lengths are solved
/// from the displacement equations in arbitrary directions, not cardinal axes.
pub(super) fn completion(
    path: &PathState,
    config: &GeneratorConfig,
    remaining: usize,
    check: &mut dyn FnMut() -> GenerationResult<()>,
) -> GenerationResult<Vec<Segment>> {
    if path.closed() {
        return Ok(vec![]);
    }
    let r = config.primitive_radius;
    let mut best: Option<(f32, Vec<Segment>)> = None;
    let mut feasible = 0;
    let mut trials = 0;
    for sign in [1, -1] {
        let need = 24 * sign - path.winding;
        for (a, b) in [
            (5, 7),
            (7, 5),
            (4, 9),
            (8, 5),
            (3, 7),
            (6, 6),
            (4, -3),
            (-3, 5),
            (9, 8),
        ] {
            let q = [a * sign, b * sign, need - (a + b) * sign];
            if q[2].abs() > 12 || q[2] == 0 {
                continue;
            }
            for scale in [3., 5., 1., 2.] {
                let arcs = [
                    Segment::Bend(scale * r, q[0]),
                    Segment::Bend((scale + 1.) * r, q[1]),
                    Segment::Bend(scale * r, q[2]),
                ];
                let mut p = path.position;
                let mut h = path.heading;
                let mut dirs = vec![direction(h)];
                for arc in &arcs {
                    (p, h) = end_pose(p, h, arc);
                    dirs.push(direction(h));
                }
                for i in 0..dirs.len() {
                    for j in i + 1..dirs.len() {
                        let det = dirs[i].perp_dot(dirs[j]);
                        if det.abs() < 0.15 {
                            continue;
                        }
                        for extra in [0., 4. * r, 10. * r] {
                            trials += 1;
                            if trials % 16 == 0 {
                                check()?;
                            }
                            let k = (j + 1) % dirs.len();
                            let target = -p - dirs[k] * extra;
                            let x = target.perp_dot(dirs[j]) / det;
                            let y = dirs[i].perp_dot(target) / det;
                            if x < 0. || y < 0. {
                                continue;
                            }
                            let mut lengths = vec![0.; dirs.len()];
                            lengths[i] += x;
                            lengths[j] += y;
                            lengths[k] += extra;
                            let mut plan = vec![];
                            for (index, l) in lengths.iter().enumerate() {
                                if *l > 0.05 {
                                    plan.push(if *l > 16. * r {
                                        Segment::Link(*l, 2 * sign)
                                    } else {
                                        Segment::Straight(*l)
                                    });
                                }
                                if index < arcs.len() {
                                    plan.push(arcs[index].clone());
                                }
                            }
                            if plan.len() > remaining {
                                continue;
                            }
                            let added = plan.iter().map(Segment::length).sum::<f32>();
                            let total = path.length + added;
                            if total < config.min_length || total > config.max_length {
                                continue;
                            }
                            let mut candidate = path.clone();
                            if plan
                                .iter()
                                .any(|s| candidate.append(s, Some(total)).is_err())
                                || !candidate.closed()
                            {
                                continue;
                            }
                            let longest = lengths.iter().copied().fold(0., f32::max);
                            let score = added
                                + 2. * longest
                                + if q[0].abs() == q[1].abs() { r * 2. } else { 0. };
                            if best.as_ref().is_none_or(|(s, _)| score < *s) {
                                best = Some((score, plan));
                            }
                            feasible += 1;
                            if feasible >= 4 {
                                return Ok(best.unwrap().1);
                            }
                        }
                    }
                }
            }
        }
    }
    best.map(|(_, p)| p).ok_or_else(|| {
        "no collision-free organic closure within remaining segment/length budget".into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostics_distinguish_a_box_from_a_circle_and_ignore_rotation() {
        let mut rectangle = vec![];
        let corners = [
            Vec2::ZERO,
            Vec2::new(1000., 0.),
            Vec2::new(1000., 600.),
            Vec2::new(0., 600.),
        ];
        for i in 0..4 {
            for j in 0..100 {
                rectangle.push(corners[i].lerp(corners[(i + 1) % 4], j as f32 / 100.));
            }
        }
        let circle: Vec<_> = (0..720)
            .map(|i| Vec2::from_angle(i as f32 * std::f32::consts::TAU / 720.) * 500.)
            .collect();
        let box_metrics = organic_metrics(&rectangle, true);
        let circle_metrics = organic_metrics(&circle, true);
        assert!(box_metrics.rectangularity > 0.95);
        assert!(circle_metrics.rectangularity < 0.01);
        assert!(circle_metrics.heading_entropy > 0.99);
        let axis = Vec2::from_angle(0.713);
        let rotated: Vec<_> = rectangle
            .iter()
            .map(|p| axis * p.x + axis.perp() * p.y + Vec2::splat(1250.))
            .collect();
        let rotated_metrics = organic_metrics(&rotated, true);
        assert!((box_metrics.axis_concentration - rotated_metrics.axis_concentration).abs() < 0.01);
        assert!((box_metrics.rectangularity - rotated_metrics.rectangularity).abs() < 0.04);
        assert!(
            (box_metrics.longest_straight_ratio - rotated_metrics.longest_straight_ratio).abs()
                < 0.01
        );
    }
    #[test]
    fn structured_connectors_preserve_endpoint_tangent_and_clearance_checks() {
        let config = GeneratorConfig::default();
        let mut path = PathState::new(&config);
        path.append(&Segment::Link(1500., 2), None).unwrap();
        assert!(path.position.distance(Vec2::new(1500., 0.)) < 0.02);
        assert_eq!(path.heading, 0);
        assert_eq!(path.winding, 0);
        assert!(path.points.iter().any(|p| p.y > 90.));
        assert!(path.length > 1500.);
        assert!(
            PathState::new(&config)
                .append(&Segment::Link(50., 2), None)
                .is_err()
        );
    }
    #[test]
    fn phase_pressure_delays_closure_and_retains_a_verified_escape() {
        let config = GeneratorConfig::default();
        let mut state = Construction::new(&config);
        let free = state.exploration_status(&config);
        assert_eq!(free.phase, ConstructionPhase::FreeExploration);
        assert!(free.closure_eta < 0.001);
        let mut guided = false;
        let mut late = false;
        let mut stroll = false;
        while !state.closed() && state.used < config.sectors {
            let status = state.exploration_status(&config);
            guided |= status.phase == ConstructionPhase::GuidedExploration;
            late |= status.phase == ConstructionPhase::Closure;
            let mut options = state.options(&config, &mut || Ok(())).unwrap();
            let choice =
                if status.phase == ConstructionPhase::Closure && options[CLOSE].next.is_some() {
                    CLOSE
                } else {
                    options
                        .iter()
                        .enumerate()
                        .filter(|(i, a)| *i != CLOSE && a.next.is_some())
                        .max_by(|a, b| a.1.eta.total_cmp(&b.1.eta))
                        .map_or(CLOSE, |(i, _)| i)
                };
            state = options[choice]
                .next
                .take()
                .expect("every chosen expansion has a verified escape");
            stroll |= state
                .exploration
                .as_ref()
                .is_some_and(|e| e.chose_extra_exploration);
        }
        assert!(state.closed() && guided && late && stroll);
        assert!(state.history.iter().position(|c| *c == CLOSE).unwrap() >= 18);
    }
}
