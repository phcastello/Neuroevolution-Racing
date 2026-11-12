//! Deterministic, bounded procedural generation in the existing TrackDefinition format.
//! Decisions are radial offsets in ordered sectors, followed by one road-width decision.
use super::track::{Track, TrackCategory, TrackDefinition, TrackDifficulty, TrackRole};
use super::{CAR_LENGTH, CAR_WIDTH};
use bevy::prelude::Vec2;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};
pub mod sequential;
#[cfg(test)]
mod width_validation;

pub type GenerationResult<T> = Result<T, String>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum GeneratorShape {
    #[default]
    Radial,
    RoundedRectangle,
    Sequential,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default = "GeneratorConfig::legacy", deny_unknown_fields)]
pub struct GeneratorConfig {
    pub shape: GeneratorShape,
    pub sectors: usize,
    pub radius: f32,
    pub radial_levels: Vec<f32>,
    pub widths: Vec<f32>,
    pub samples_per_control_point: usize,
    pub min_length: f32,
    pub max_length: f32,
    /// Inverse world units. Radius must also exceed half-width plus half car length.
    pub max_curvature: f32,
    pub max_attempts: usize,
    pub primitive_radius: f32,
    pub minimum_border_gap: f32,
    /// Opt in explicitly so archived v2 searches retain their original graph and closure.
    pub organic: bool,
    /// Zero preserves validation semantics of existing archived generators.
    pub width_geometry_version: u32,
}
impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            shape: GeneratorShape::Sequential,
            sectors: 26,
            radius: 650.,
            radial_levels: vec![0.82, 0.91, 1., 1.09, 1.18],
            widths: vec![96., 100., 104.],
            samples_per_control_point: 12,
            min_length: 1800.,
            max_length: 14000.,
            max_curvature: 0.02094,
            max_attempts: 32,
            primitive_radius: 75.,
            minimum_border_gap: 12.,
            organic: true,
            width_geometry_version: 1,
        }
    }
}
impl GeneratorConfig {
    pub fn legacy() -> Self {
        Self {
            shape: GeneratorShape::Radial,
            sectors: 12,
            widths: vec![80., 100., 120.],
            min_length: 2800.,
            max_length: 6200.,
            max_curvature: 0.012,
            organic: false,
            primitive_radius: 50.,
            width_geometry_version: 0,
            ..Self::default()
        }
    }
    pub fn is_sequential(&self) -> bool {
        self.shape == GeneratorShape::Sequential
    }
    pub fn version(&self) -> u32 {
        if self.is_sequential() { 2 } else { 1 }
    }
    pub fn decisions_valid(&self, decisions: &[usize]) -> bool {
        if self.is_sequential() {
            (2..=self.sectors + 1).contains(&decisions.len())
                && decisions[..decisions.len() - 1].iter().all(|c| {
                    *c < if self.organic {
                        sequential::CHOICES
                    } else {
                        sequential::LEGACY_CHOICES
                    }
                })
                && decisions.last().is_some_and(|c| *c < self.widths.len())
        } else {
            decisions.len() == self.sectors + 1
                && decisions
                    .iter()
                    .zip(self.choice_counts())
                    .all(|(c, n)| *c < n)
        }
    }
    pub fn validate(&self) -> GenerationResult<()> {
        if self.width_geometry_version > 1 {
            return Err("unsupported width geometry parameter version".into());
        }
        if self.is_sequential() && self.width_geometry_version == 1 {
            let required = (1. / self.max_curvature)
                .max(self.widths.iter().copied().fold(0., f32::max) * 0.5 + CAR_LENGTH * 0.5 + 3.);
            if self.primitive_radius < required {
                return Err(format!(
                    "road width requires primitive radius >= {required:.2}; configured {:.2}; width was not reduced",
                    self.primitive_radius
                ));
            }
        }
        if !self.primitive_radius.is_finite()
            || self.primitive_radius < 50.
            || self.primitive_radius > 1000.
            || !self.minimum_border_gap.is_finite()
            || self.minimum_border_gap < 4.
        {
            return Err("invalid primitive radius / border clearance".into());
        }
        if self.is_sequential() && self.max_length > 100_000. {
            return Err("sequential length budget must not exceed 100000 world units".into());
        }
        if (self.shape == GeneratorShape::RoundedRectangle && self.sectors % 4 != 0)
            || !(6..=32).contains(&self.sectors)
            || !(4..=32).contains(&self.samples_per_control_point)
            || self.radial_levels.is_empty()
            || self.radial_levels.len() > 32
            || self.widths.is_empty()
            || self.widths.len() > 32
            || self.max_attempts == 0
            || self.max_attempts > 1024
        {
            return Err("generator requires 6..32 sectors, 4..32 samples, 1..32 choices and 1..1024 attempts".into());
        }
        if !self.radius.is_finite()
            || self.radius < 100.
            || !self.min_length.is_finite()
            || !self.max_length.is_finite()
            || self.min_length <= 0.
            || self.max_length < self.min_length
            || !self.max_curvature.is_finite()
            || self.max_curvature <= 0.
            || self
                .radial_levels
                .iter()
                .any(|r| !r.is_finite() || !(0.5..=1.5).contains(r))
            || self
                .widths
                .iter()
                .any(|w| !w.is_finite() || *w < 2. * CAR_WIDTH + 10.)
        {
            return Err("invalid generator dimensions, curvature or car clearance".into());
        }
        Ok(())
    }
    pub fn choice_counts(&self) -> Vec<usize> {
        let mut counts = vec![
            if self.is_sequential() {
                if self.organic {
                    sequential::CHOICES
                } else {
                    sequential::LEGACY_CHOICES
                }
            } else {
                self.radial_levels.len()
            };
            self.sectors
        ];
        counts.push(self.widths.len());
        counts
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GenerationProvenance {
    pub version: u32,
    pub seed: u64,
    pub attempt: usize,
    pub config: GeneratorConfig,
    pub decisions: Vec<usize>,
    pub geometry_hash: String,
}
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GeometryMetrics {
    pub length: f32,
    pub width: f32,
    pub max_curvature: f32,
    pub mean_absolute_curvature: f32,
    pub direction_changes: usize,
    pub spawn_clearance: f32,
    #[serde(default)]
    pub longest_straight: f32,
    #[serde(default)]
    pub straight_fraction: f32,
    #[serde(default)]
    pub turns: usize,
    #[serde(default)]
    pub radius_quantiles: [f32; 3],
    #[serde(default)]
    pub curvature_variation: f32,
    #[serde(default)]
    pub hairpins: usize,
    #[serde(default)]
    pub chicanes: usize,
    #[serde(default)]
    pub s_bends: usize,
    #[serde(default)]
    pub compactness: f32,
    #[serde(default)]
    pub descriptor: Vec<f32>,
    #[serde(default)]
    pub organic: sequential::OrganicMetrics,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeneratedTrack {
    pub definition: TrackDefinition,
    pub provenance: GenerationProvenance,
    pub geometry: GeometryMetrics,
}
impl GeneratedTrack {
    pub fn validate(&self) -> GenerationResult<Track> {
        if self.provenance.version != self.provenance.config.version()
            || self.definition.role != TrackRole::Training
        {
            return Err("generated representation version mismatch or track is not training-only; refusing validation/test registration".into());
        }
        let rebuilt = from_decisions(
            &self.provenance.config,
            self.provenance.seed,
            self.provenance.attempt,
            &self.provenance.decisions,
        )?;
        if self.definition != rebuilt.definition
            || self.provenance.geometry_hash != rebuilt.provenance.geometry_hash
            || self.geometry != rebuilt.geometry
        {
            return Err("generated track definition/hash/metrics do not match provenance".into());
        }
        Track::from_definition(&self.definition).map_err(|e| e.to_string())
    }
    /// No overwrite: exported artifacts are immutable and training-only.
    pub fn save(&self, path: impl AsRef<Path>) -> GenerationResult<()> {
        self.validate()?;
        write_new_ron(path.as_ref(), self)
    }
    pub fn load(path: impl AsRef<Path>) -> GenerationResult<Self> {
        let value: Self = read_ron(path.as_ref())?;
        value.validate()?;
        Ok(value)
    }
    pub fn export_definition(&self, path: impl AsRef<Path>) -> GenerationResult<()> {
        self.validate()?;
        write_new_ron(path.as_ref(), &self.definition)
    }
}

pub fn generate(config: &GeneratorConfig, seed: u64) -> GenerationResult<GeneratedTrack> {
    config.validate()?;
    if config.is_sequential() {
        return sequential::generate(config, seed);
    }
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let counts = config.choice_counts();
    let mut last_error = String::new();
    for attempt in 0..config.max_attempts {
        let decisions = counts
            .iter()
            .map(|n| rng.random_range(0..*n))
            .collect::<Vec<_>>();
        match from_decisions(config, seed, attempt, &decisions) {
            Ok(track) => return Ok(track),
            Err(error) => last_error = error,
        }
    }
    Err(format!(
        "generation exhausted {} attempts: {last_error}",
        config.max_attempts
    ))
}

pub fn from_decisions(
    config: &GeneratorConfig,
    seed: u64,
    attempt: usize,
    decisions: &[usize],
) -> GenerationResult<GeneratedTrack> {
    config.validate()?;
    if !config.decisions_valid(decisions) {
        return Err(if config.is_sequential() {
            "construction incomplete: no feasible closure within the decision/length budget".into()
        } else {
            "generator decision shape/choice out of bounds".into()
        });
    }
    let definition = definition_from_decisions(config, decisions)?;
    let hash = geometry_hash(
        &definition.control_points,
        definition.width,
        definition.samples_per_control_point,
    );
    let track = Track::from_definition(&definition).map_err(|e| e.to_string())?;
    let geometry = validate_geometry(&track, config)?;
    Ok(GeneratedTrack {
        definition,
        geometry,
        provenance: GenerationProvenance {
            version: config.version(),
            seed,
            attempt,
            config: config.clone(),
            decisions: decisions.to_vec(),
            geometry_hash: hash,
        },
    })
}

pub fn validate_geometry(
    track: &Track,
    config: &GeneratorConfig,
) -> GenerationResult<GeometryMetrics> {
    if track.definition.role != TrackRole::Training {
        return Err("generation may only use training tracks".into());
    }
    if track.total_length < config.min_length || track.total_length > config.max_length {
        return Err(format!(
            "length {} outside [{},{}]",
            track.total_length, config.min_length, config.max_length
        ));
    }
    if track.width < 2. * CAR_WIDTH + 10. {
        return Err("insufficient vehicle clearance".into());
    }
    let mut max_curvature: f32 = 0.;
    let mut sum = 0.;
    let mut direction_changes = 0;
    let mut last_sign = 0.;
    let n = track.samples.len();
    for i in 0..n {
        let a = track.samples[(i + n - 1) % n].position;
        let b = track.samples[i].position;
        let c = track.samples[(i + 1) % n].position;
        let ab = b - a;
        let bc = c - b;
        let denominator = ab.length() * bc.length() * c.distance(a);
        let curvature = if denominator > 1.0e-6 {
            2. * ab.perp_dot(bc) / denominator
        } else {
            return Err("degenerate curvature".into());
        };
        if !curvature.is_finite() {
            return Err("non-finite curvature".into());
        }
        max_curvature = max_curvature.max(curvature.abs());
        sum += curvature.abs();
        if curvature.abs() > 0.0001 {
            if last_sign != 0. && last_sign != curvature.signum() {
                direction_changes += 1;
            }
            last_sign = curvature.signum();
        }
    }
    let safe_curvature = config.max_curvature.min(
        1. / (track.width * 0.5
            + if config.is_sequential() {
                CAR_LENGTH * 0.5 + 3.
            } else {
                CAR_LENGTH
            }),
    );
    if max_curvature > safe_curvature {
        return Err(format!(
            "curvature {max_curvature:.6} exceeds {safe_curvature:.6}"
        ));
    }
    // Conservative rectangle clearance for the canonical spawn at arc distance 16.
    let (spawn, tangent) = track.pose_at_distance(16.);
    let corners = [-1., 1.].into_iter().flat_map(|longitudinal| {
        [-1., 1.].map(move |lateral| {
            spawn
                + tangent * (longitudinal * CAR_LENGTH * 0.5)
                + tangent.perp() * (lateral * CAR_WIDTH * 0.5)
        })
    });
    let mut spawn_clearance = f32::INFINITY;
    for corner in corners {
        let offset = track.project(corner).point.distance(corner);
        spawn_clearance = spawn_clearance.min(track.width * 0.5 - offset);
    }
    if spawn_clearance < 3. {
        return Err("spawn rectangle too close to track border".into());
    }
    let mut metrics = GeometryMetrics {
        length: track.total_length,
        width: track.width,
        max_curvature,
        mean_absolute_curvature: sum / n as f32,
        direction_changes,
        spawn_clearance,
        ..Default::default()
    };
    if config.is_sequential() {
        sequential::characterize(track, &mut metrics);
        if config.organic {
            metrics.organic = sequential::organic_metrics(
                &track.samples.iter().map(|s| s.position).collect::<Vec<_>>(),
                true,
            );
        }
    }
    Ok(metrics)
}

/// FNV-1a over explicit little-endian geometry bits, stable across Rust versions.
fn geometry_hash(points: &[[f32; 2]], width: f32, samples: usize) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in (points.len() as u64)
        .to_le_bytes()
        .into_iter()
        .chain((samples as u64).to_le_bytes())
        .chain(width.to_bits().to_le_bytes())
        .chain(
            points
                .iter()
                .flatten()
                .flat_map(|x| x.to_bits().to_le_bytes()),
        )
    {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

pub(crate) fn write_new_ron<T: Serialize>(path: &Path, value: &T) -> GenerationResult<()> {
    use std::io::Write;
    let serialized = ron::ser::to_string_pretty(value, ron::ser::PrettyConfig::default())
        .map_err(|e| e.to_string())?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    file.write_all(serialized.as_bytes())
        .map_err(|e| e.to_string())
}
pub(crate) fn read_ron<T: serde::de::DeserializeOwned>(path: &Path) -> GenerationResult<T> {
    let source = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    ron::from_str(&source).map_err(|e| format!("{}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn deterministic_closed_valid_training_tracks() {
        let config = GeneratorConfig::default();
        for seed in [3, 17, 47] {
            let a = generate(&config, seed).unwrap();
            assert_eq!(a, generate(&config, seed).unwrap());
            let track = a.validate().unwrap();
            assert!(track.total_length >= config.min_length);
            assert!(a.geometry.max_curvature <= config.max_curvature);
            assert!(a.geometry.spawn_clearance > 3.);
            assert!(
                track
                    .point_at_distance(0.)
                    .distance(track.point_at_distance(track.total_length))
                    < 0.01
            );
        }
    }
    #[test]
    fn rounded_rectangle_can_mix_actual_straights_and_curves() {
        let config = GeneratorConfig {
            shape: GeneratorShape::RoundedRectangle,
            radius: 700.,
            radial_levels: vec![1.],
            widths: vec![80.],
            ..GeneratorConfig::legacy()
        };
        let generated = generate(&config, 47).unwrap();
        let track = generated.validate().unwrap();
        let straight = track.samples.windows(3).any(|p| {
            let a = p[1].position - p[0].position;
            let b = p[2].position - p[1].position;
            a.perp_dot(b).abs() < 1.0e-5
        });
        assert!(straight);
        assert!(generated.geometry.max_curvature > 0.001);
    }
    #[test]
    fn bounded_rejection_and_no_validation_role() {
        let config = GeneratorConfig {
            min_length: 1.,
            max_length: 2.,
            max_attempts: 2,
            ..Default::default()
        };
        assert!(generate(&config, 1).unwrap_err().contains("exhausted 2"));
        let mut generated = generate(&GeneratorConfig::default(), 4).unwrap();
        generated.definition.role = TrackRole::Validation;
        assert!(generated.validate().is_err());
    }
    #[test]
    fn generated_track_uses_real_avian_walls_and_valid_spawn() {
        use crate::simulation::{
            SimulationConfig,
            rl_environment::{RacingEnvironment, Scene},
        };
        use racing_rl::{config::Config, rollout::Environment};
        use std::rc::Rc;
        let generated = generate(&GeneratorConfig::default(), 47).unwrap();
        let track = generated.validate().unwrap();
        let scene = Rc::new(Scene::new(track));
        let mut environment = RacingEnvironment::new(
            0,
            vec![scene],
            &Config::default(),
            &SimulationConfig::default(),
            ChaCha12Rng::seed_from_u64(47),
        )
        .unwrap();
        let observation = environment.observe().unwrap();
        assert!(observation.iter().all(|x| x.is_finite()));
        for _ in 0..3 {
            let step = environment.step([0., 0.]).unwrap();
            assert!(!step.terminated);
        }
    }
    #[test]
    fn serialization_preserves_hash_and_provenance() {
        let original = generate(&GeneratorConfig::default(), 9).unwrap();
        let encoded = ron::to_string(&original).unwrap();
        let restored: GeneratedTrack = ron::from_str(&encoded).unwrap();
        assert_eq!(original, restored);
        restored.validate().unwrap();
        let mut tampered = restored;
        tampered.definition.width += 1.;
        assert!(tampered.validate().is_err());
    }
}

/// A single generator control point, shared by construction and the live view.
pub fn control_point(config: &GeneratorConfig, sector: usize, level: usize) -> [f32; 2] {
    let base = match config.shape {
        GeneratorShape::Radial => {
            let angle = std::f32::consts::TAU * sector as f32 / config.sectors as f32;
            Vec2::from_angle(angle)
        }
        GeneratorShape::RoundedRectangle => {
            // Four perimeter edges, with enough collinear controls to preserve
            // actual straight spans in the same closed Catmull-Rom representation.
            let along = sector as f32 * 4. / config.sectors as f32;
            let t = along.fract();
            match along as usize {
                0 => Vec2::new(1., -1. + 2. * t),
                1 => Vec2::new(1. - 2. * t, 1.),
                2 => Vec2::new(-1., 1. - 2. * t),
                _ => Vec2::new(-1. + 2. * t, -1.),
            }
        }
        GeneratorShape::Sequential => panic!("sequential points require a construction state"),
    };
    (base * config.radius * config.radial_levels[level]).to_array()
}

/// Complete geometry before validation, also used to display rejected circuits honestly.
pub fn definition_from_decisions(
    config: &GeneratorConfig,
    decisions: &[usize],
) -> GenerationResult<TrackDefinition> {
    config.validate()?;
    if config.is_sequential() {
        return sequential::definition(config, decisions);
    }
    if decisions.len() != config.sectors + 1
        || decisions
            .iter()
            .zip(config.choice_counts())
            .any(|(c, n)| *c >= n)
    {
        return Err("generator decision shape/choice out of bounds".into());
    }
    let control_points = decisions[..config.sectors]
        .iter()
        .enumerate()
        .map(|(sector, level)| control_point(config, sector, *level))
        .collect::<Vec<_>>();
    let width = config.widths[decisions[config.sectors]];
    let hash = geometry_hash(&control_points, width, config.samples_per_control_point);
    let definition = TrackDefinition {
        sampled_centerline: false,
        id: format!("generated_{hash}"),
        name: format!("Procedural {hash}"),
        country: None,
        category: TrackCategory::PermanentCircuit,
        role: TrackRole::Training,
        difficulty: TrackDifficulty::MediumHard,
        width,
        samples_per_control_point: config.samples_per_control_point,
        start_index: 0,
        control_points,
        description: Some(
            "Procedural training circuit; not a held-out validation/test track".into(),
        ),
        approximation_note: Some(
            "Closed Catmull-Rom radial sectors; validated sampled borders and curvature".into(),
        ),
    };
    Ok(definition)
}
