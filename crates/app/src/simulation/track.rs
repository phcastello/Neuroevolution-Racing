use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt, fs,
    path::{Path, PathBuf},
};

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

const MIN_TOTAL_LENGTH: f32 = 1.0;
const GEOMETRY_EPSILON: f32 = 1.0e-4;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackRole {
    Training,
    Validation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackCategory {
    PermanentCircuit,
    StreetCircuit,
}

impl TrackCategory {
    pub fn label(self) -> &'static str {
        match self {
            Self::PermanentCircuit => "Permanent circuit",
            Self::StreetCircuit => "Street circuit",
        }
    }
}

impl TrackRole {
    pub fn label(self) -> &'static str {
        match self {
            Self::Training => "Training",
            Self::Validation => "Validation",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackDifficulty {
    Easy,
    EasyMedium,
    Medium,
    MediumHard,
    Hard,
    VeryHard,
}

impl TrackDifficulty {
    pub fn label(self) -> &'static str {
        match self {
            Self::Easy => "Easy",
            Self::EasyMedium => "Easy / Medium",
            Self::Medium => "Medium",
            Self::MediumHard => "Medium / Hard",
            Self::Hard => "Hard",
            Self::VeryHard => "Very Hard",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TrackDefinition {
    pub id: String,
    pub name: String,
    pub country: Option<String>,
    pub category: TrackCategory,
    pub role: TrackRole,
    pub difficulty: TrackDifficulty,
    pub width: f32,
    pub samples_per_control_point: usize,
    pub start_index: usize,
    pub control_points: Vec<[f32; 2]>,
    pub description: Option<String>,
    pub approximation_note: Option<String>,
}

impl TrackDefinition {
    pub fn from_ron_str(source: &str) -> Result<Self, TrackLoadError> {
        ron::from_str(source)
            .map_err(|error| TrackLoadError::new(format!("could not parse track RON: {error}")))
    }

    fn context(&self) -> String {
        if self.id.trim().is_empty() {
            self.name.clone()
        } else {
            format!("{} ({})", self.name, self.id)
        }
    }

    fn validate_input(&self) -> Result<(), TrackLoadError> {
        let context = self.context();
        if self.id.trim().is_empty() {
            return Err(TrackLoadError::new(format!(
                "track {context}: id must not be empty"
            )));
        }
        if self.name.trim().is_empty() {
            return Err(TrackLoadError::new(format!(
                "track {context}: display name must not be empty"
            )));
        }
        if !self.width.is_finite() || self.width <= 0.0 {
            return Err(TrackLoadError::new(format!(
                "track {context}: width must be finite and greater than zero"
            )));
        }
        if self.samples_per_control_point == 0 {
            return Err(TrackLoadError::new(format!(
                "track {context}: samples_per_control_point must be greater than zero"
            )));
        }
        if self.control_points.len() < 4 {
            return Err(TrackLoadError::new(format!(
                "track {context}: at least 4 control points are required"
            )));
        }
        if self.start_index >= self.control_points.len() {
            return Err(TrackLoadError::new(format!(
                "track {context}: start_index {} is outside {} control points",
                self.start_index,
                self.control_points.len()
            )));
        }
        if self
            .control_points
            .iter()
            .flatten()
            .any(|coordinate| !coordinate.is_finite())
        {
            return Err(TrackLoadError::new(format!(
                "track {context}: control point coordinates must be finite"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct TrackSuiteDefinition {
    tracks: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrackLoadError {
    message: String,
}

impl TrackLoadError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TrackLoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for TrackLoadError {}

#[derive(Resource, Clone, Debug)]
pub struct TrackLibrary {
    definitions: HashMap<String, TrackDefinition>,
    training_ids: Vec<String>,
    validation_ids: Vec<String>,
}

impl TrackLibrary {
    pub fn load_default() -> Result<Self, TrackLoadError> {
        Self::load_from_directory(default_tracks_directory())
    }

    pub fn load_from_directory(directory: impl AsRef<Path>) -> Result<Self, TrackLoadError> {
        let directory = directory.as_ref();
        let entries = fs::read_dir(directory).map_err(|error| {
            TrackLoadError::new(format!(
                "could not read track directory {}: {error}",
                directory.display()
            ))
        })?;
        let mut definition_paths = entries
            .map(|entry| {
                entry.map(|entry| entry.path()).map_err(|error| {
                    TrackLoadError::new(format!(
                        "could not read an entry in {}: {error}",
                        directory.display()
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        definition_paths.retain(|path| {
            path.extension().is_some_and(|extension| extension == "ron")
                && !matches!(
                    path.file_stem().and_then(|stem| stem.to_str()),
                    Some("training" | "validation")
                )
        });
        definition_paths.sort();

        let mut definitions = HashMap::new();
        for path in definition_paths {
            let source = read_to_string(&path)?;
            let definition = TrackDefinition::from_ron_str(&source)
                .map_err(|error| TrackLoadError::new(format!("{}: {error}", path.display())))?;
            definition.validate_input()?;
            Track::from_definition(&definition)?;
            let expected_file_stem = path.file_stem().and_then(|stem| stem.to_str());
            if expected_file_stem != Some(definition.id.as_str()) {
                return Err(TrackLoadError::new(format!(
                    "track {}: id must match file name {}",
                    definition.context(),
                    path.display()
                )));
            }
            if definitions
                .insert(definition.id.clone(), definition.clone())
                .is_some()
            {
                return Err(TrackLoadError::new(format!(
                    "track {}: duplicate id",
                    definition.context()
                )));
            }
        }

        let training_ids = load_suite(directory.join("training.ron"), TrackRole::Training)?;
        let validation_ids = load_suite(directory.join("validation.ron"), TrackRole::Validation)?;
        validate_suites(&definitions, &training_ids, &validation_ids)?;

        Ok(Self {
            definitions,
            training_ids,
            validation_ids,
        })
    }

    pub fn definition(&self, id: &str) -> Option<&TrackDefinition> {
        self.definitions.get(id)
    }

    pub fn training_tracks(&self) -> impl ExactSizeIterator<Item = &TrackDefinition> {
        self.training_ids
            .iter()
            .map(|id| &self.definitions[id.as_str()])
    }

    pub fn validation_tracks(&self) -> impl ExactSizeIterator<Item = &TrackDefinition> {
        self.validation_ids
            .iter()
            .map(|id| &self.definitions[id.as_str()])
    }

    pub fn all_tracks(&self) -> impl Iterator<Item = &TrackDefinition> {
        self.training_tracks().chain(self.validation_tracks())
    }
}

fn default_tracks_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/tracks")
}

fn read_to_string(path: &Path) -> Result<String, TrackLoadError> {
    fs::read_to_string(path)
        .map_err(|error| TrackLoadError::new(format!("could not read {}: {error}", path.display())))
}

fn load_suite(path: PathBuf, expected_role: TrackRole) -> Result<Vec<String>, TrackLoadError> {
    let source = read_to_string(&path)?;
    let suite: TrackSuiteDefinition = ron::from_str(&source).map_err(|error| {
        TrackLoadError::new(format!("could not parse suite {}: {error}", path.display()))
    })?;
    if suite.tracks.is_empty() {
        return Err(TrackLoadError::new(format!(
            "{} suite must contain at least one track",
            expected_role.label()
        )));
    }
    Ok(suite.tracks)
}

fn validate_suites(
    definitions: &HashMap<String, TrackDefinition>,
    training_ids: &[String],
    validation_ids: &[String],
) -> Result<(), TrackLoadError> {
    let mut listed = HashSet::new();
    for (ids, expected_role) in [
        (training_ids, TrackRole::Training),
        (validation_ids, TrackRole::Validation),
    ] {
        for id in ids {
            let definition = definitions.get(id).ok_or_else(|| {
                TrackLoadError::new(format!(
                    "{} suite references missing track {id:?}",
                    expected_role.label()
                ))
            })?;
            if definition.role != expected_role {
                return Err(TrackLoadError::new(format!(
                    "track {}: role {:?} does not match its {} suite",
                    definition.context(),
                    definition.role,
                    expected_role.label()
                )));
            }
            if !listed.insert(id.clone()) {
                return Err(TrackLoadError::new(format!(
                    "track {} is listed more than once across suites",
                    definition.context()
                )));
            }
        }
    }
    for definition in definitions.values() {
        if !listed.contains(&definition.id) {
            return Err(TrackLoadError::new(format!(
                "track {} is not referenced by its suite",
                definition.context()
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct TrackSample {
    pub position: Vec2,
    pub tangent: Vec2,
    pub cumulative_distance: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct TrackProjection {
    pub segment_index: usize,
    pub point: Vec2,
    pub track_distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackBounds {
    pub min: Vec2,
    pub max: Vec2,
}

impl TrackBounds {
    pub fn size(self) -> Vec2 {
        self.max - self.min
    }

    pub fn center(self) -> Vec2 {
        (self.min + self.max) * 0.5
    }
}

#[derive(Resource, Clone, Debug)]
pub struct Track {
    pub definition: TrackDefinition,
    pub control_points: Vec<Vec2>,
    pub samples: Vec<TrackSample>,
    pub left_border: Vec<Vec2>,
    pub right_border: Vec<Vec2>,
    pub total_length: f32,
    pub width: f32,
}

impl Track {
    pub fn from_definition(definition: &TrackDefinition) -> Result<Self, TrackLoadError> {
        definition.validate_input()?;
        let mut control_points = definition
            .control_points
            .iter()
            .copied()
            .map(Vec2::from_array)
            .collect::<Vec<_>>();
        control_points.rotate_left(definition.start_index);

        let samples_per_control_point = definition.samples_per_control_point;
        let mut positions = Vec::with_capacity(control_points.len() * samples_per_control_point);
        for index in 0..control_points.len() {
            let p0 = control_points[(index + control_points.len() - 1) % control_points.len()];
            let p1 = control_points[index];
            let p2 = control_points[(index + 1) % control_points.len()];
            let p3 = control_points[(index + 2) % control_points.len()];
            for step in 0..samples_per_control_point {
                let t = step as f32 / samples_per_control_point as f32;
                positions.push(catmull_rom(p0, p1, p2, p3, t));
            }
        }

        Self::from_centerline(definition.clone(), control_points, positions)
    }

    fn from_centerline(
        definition: TrackDefinition,
        control_points: Vec<Vec2>,
        positions: Vec<Vec2>,
    ) -> Result<Self, TrackLoadError> {
        let context = definition.context();
        let mut cumulative_distance = 0.0;
        let mut samples = Vec::with_capacity(positions.len());
        for index in 0..positions.len() {
            if index > 0 {
                cumulative_distance += positions[index - 1].distance(positions[index]);
            }
            let previous = positions[(index + positions.len() - 1) % positions.len()];
            let next = positions[(index + 1) % positions.len()];
            let tangent = (next - previous).normalize_or(Vec2::X);
            samples.push(TrackSample {
                position: positions[index],
                tangent,
                cumulative_distance,
            });
        }

        let total_length =
            cumulative_distance + positions[positions.len() - 1].distance(positions[0]);
        if !total_length.is_finite() || total_length < MIN_TOTAL_LENGTH {
            return Err(TrackLoadError::new(format!(
                "track {context}: generated total length is zero or near zero"
            )));
        }

        let half_width = definition.width * 0.5;
        let left_border = samples
            .iter()
            .map(|sample| sample.position + sample.tangent.perp() * half_width)
            .collect::<Vec<_>>();
        let right_border = samples
            .iter()
            .map(|sample| sample.position - sample.tangent.perp() * half_width)
            .collect::<Vec<_>>();

        validate_closed_polyline(&positions, "centerline", &context)?;
        validate_closed_polyline(&left_border, "left border", &context)?;
        validate_closed_polyline(&right_border, "right border", &context)?;
        validate_distinct_polylines(&left_border, &right_border, &context)?;

        Ok(Self {
            width: definition.width,
            definition,
            control_points,
            samples,
            left_border,
            right_border,
            total_length,
        })
    }

    pub fn project(&self, position: Vec2) -> TrackProjection {
        self.best_projection(position, 0..self.samples.len())
    }

    pub fn bounds(&self) -> TrackBounds {
        let mut points = self.left_border.iter().chain(&self.right_border).copied();
        let first = points.next().expect("a validated track has border points");
        let (min, max) = points.fold((first, first), |(min, max), point| {
            (min.min(point), max.max(point))
        });
        TrackBounds { min, max }
    }

    pub fn project_near(
        &self,
        position: Vec2,
        hint_segment: usize,
        search_radius: usize,
    ) -> TrackProjection {
        let segment_count = self.samples.len();
        let radius = search_radius.min(segment_count.saturating_sub(1) / 2);
        let indices = (-(radius as isize)..=radius as isize).map(|offset| {
            (hint_segment as isize + offset).rem_euclid(segment_count as isize) as usize
        });
        self.best_projection(position, indices)
    }

    pub fn point_at_distance(&self, distance: f32) -> Vec2 {
        let distance = distance.rem_euclid(self.total_length);
        let next_index = self
            .samples
            .partition_point(|sample| sample.cumulative_distance <= distance);
        let segment_index = next_index.saturating_sub(1).min(self.samples.len() - 1);
        let (start, end, segment_length) = self.segment(segment_index);
        let t = if segment_length > f32::EPSILON {
            (distance - start.cumulative_distance) / segment_length
        } else {
            0.0
        };
        start.position.lerp(end.position, t.clamp(0.0, 1.0))
    }

    fn best_projection(
        &self,
        position: Vec2,
        segment_indices: impl IntoIterator<Item = usize>,
    ) -> TrackProjection {
        segment_indices
            .into_iter()
            .map(|segment_index| self.project_onto_segment(position, segment_index))
            .min_by(|left, right| {
                left.point
                    .distance_squared(position)
                    .total_cmp(&right.point.distance_squared(position))
            })
            .expect("a track projection needs at least one segment")
    }

    fn project_onto_segment(&self, position: Vec2, segment_index: usize) -> TrackProjection {
        let (start, end, segment_length) = self.segment(segment_index);
        let delta = end.position - start.position;
        let t = if segment_length > f32::EPSILON {
            ((position - start.position).dot(delta) / delta.length_squared()).clamp(0.0, 1.0)
        } else {
            0.0
        };

        TrackProjection {
            segment_index,
            point: start.position + delta * t,
            track_distance: start.cumulative_distance + segment_length * t,
        }
    }

    fn segment(&self, index: usize) -> (&TrackSample, &TrackSample, f32) {
        let start = &self.samples[index];
        let end = &self.samples[(index + 1) % self.samples.len()];
        let length = start.position.distance(end.position);
        (start, end, length)
    }
}

pub fn normalized_progress(track_distance: f32, total_length: f32) -> f32 {
    if total_length <= f32::EPSILON {
        0.0
    } else {
        (track_distance / total_length).clamp(0.0, 1.0)
    }
}

pub fn wrapped_distance_delta(previous: f32, current: f32, total_length: f32) -> f32 {
    if total_length <= f32::EPSILON {
        return 0.0;
    }
    (current - previous + total_length * 0.5).rem_euclid(total_length) - total_length * 0.5
}

pub fn closed_segments(points: &[Vec2]) -> impl Iterator<Item = (Vec2, Vec2)> + '_ {
    points
        .iter()
        .copied()
        .zip(points.iter().copied().cycle().skip(1))
        .take(points.len())
}

fn catmull_rom(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

fn validate_closed_polyline(
    points: &[Vec2],
    geometry_name: &str,
    context: &str,
) -> Result<(), TrackLoadError> {
    for (index, (start, end)) in closed_segments(points).enumerate() {
        if start.distance_squared(end) <= GEOMETRY_EPSILON * GEOMETRY_EPSILON {
            return Err(TrackLoadError::new(format!(
                "track {context}: {geometry_name} segment {index} has zero or near-zero length"
            )));
        }
    }

    for first in 0..points.len() {
        let first_next = (first + 1) % points.len();
        for second in first + 1..points.len() {
            let second_next = (second + 1) % points.len();
            let adjacent = first_next == second || second_next == first;
            if adjacent {
                continue;
            }
            if segments_intersect(
                points[first],
                points[first_next],
                points[second],
                points[second_next],
            ) {
                return Err(TrackLoadError::new(format!(
                    "track {context}: {geometry_name} self-intersects at segments {first} and {second}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_distinct_polylines(
    left: &[Vec2],
    right: &[Vec2],
    context: &str,
) -> Result<(), TrackLoadError> {
    for (left_index, (left_start, left_end)) in closed_segments(left).enumerate() {
        for (right_index, (right_start, right_end)) in closed_segments(right).enumerate() {
            if segments_intersect(left_start, left_end, right_start, right_end) {
                return Err(TrackLoadError::new(format!(
                    "track {context}: left border segment {left_index} intersects right border segment {right_index}"
                )));
            }
        }
    }
    Ok(())
}

fn segments_intersect(a: Vec2, b: Vec2, c: Vec2, d: Vec2) -> bool {
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let cd = d - c;
    let ca = a - c;
    let cb = b - c;
    let first_sides = ab.perp_dot(ac) * ab.perp_dot(ad);
    let second_sides = cd.perp_dot(ca) * cd.perp_dot(cb);
    if first_sides < -GEOMETRY_EPSILON && second_sides < -GEOMETRY_EPSILON {
        return true;
    }

    let collinear = |cross: f32, point: Vec2, start: Vec2, end: Vec2| {
        cross.abs() <= GEOMETRY_EPSILON
            && point.x >= start.x.min(end.x) - GEOMETRY_EPSILON
            && point.x <= start.x.max(end.x) + GEOMETRY_EPSILON
            && point.y >= start.y.min(end.y) - GEOMETRY_EPSILON
            && point.y <= start.y.max(end.y) + GEOMETRY_EPSILON
    };
    collinear(ab.perp_dot(ac), c, a, b)
        || collinear(ab.perp_dot(ad), d, a, b)
        || collinear(cd.perp_dot(ca), a, c, d)
        || collinear(cd.perp_dot(cb), b, c, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition(control_points: Vec<[f32; 2]>) -> TrackDefinition {
        TrackDefinition {
            id: "test".into(),
            name: "Test Track".into(),
            country: None,
            category: TrackCategory::PermanentCircuit,
            role: TrackRole::Training,
            difficulty: TrackDifficulty::Easy,
            width: 2.0,
            samples_per_control_point: 8,
            start_index: 0,
            control_points,
            description: None,
            approximation_note: None,
        }
    }

    fn square_track() -> Track {
        Track::from_definition(&definition(vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ]))
        .unwrap()
    }

    #[test]
    fn loads_a_valid_track_definition_from_ron() {
        let parsed = TrackDefinition::from_ron_str(
            r#"(
                id: "test",
                name: "Test",
                country: None,
                category: PermanentCircuit,
                role: Training,
                difficulty: Easy,
                width: 10.0,
                samples_per_control_point: 4,
                start_index: 0,
                control_points: [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
                description: None,
                approximation_note: None,
            )"#,
        )
        .unwrap();

        assert_eq!(parsed.id, "test");
        assert_eq!(parsed.role, TrackRole::Training);
    }

    #[test]
    fn rejects_malformed_definitions_with_track_context() {
        let invalid = definition(vec![[0.0, 0.0]; 3]);
        let error = Track::from_definition(&invalid).unwrap_err().to_string();

        assert!(error.contains("Test Track (test)"));
        assert!(error.contains("at least 4 control points"));
    }

    #[test]
    fn rejects_invalid_width_sampling_and_self_intersection() {
        let mut invalid_width =
            definition(vec![[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]);
        invalid_width.width = 0.0;
        assert!(
            Track::from_definition(&invalid_width)
                .unwrap_err()
                .to_string()
                .contains("width")
        );

        let mut invalid_sampling = invalid_width.clone();
        invalid_sampling.width = 2.0;
        invalid_sampling.samples_per_control_point = 0;
        assert!(
            Track::from_definition(&invalid_sampling)
                .unwrap_err()
                .to_string()
                .contains("samples_per_control_point")
        );

        let crossing = definition(vec![
            [-100.0, -100.0],
            [100.0, 100.0],
            [-100.0, 100.0],
            [100.0, -100.0],
        ]);
        assert!(
            Track::from_definition(&crossing)
                .unwrap_err()
                .to_string()
                .contains("centerline self-intersects")
        );
    }

    #[test]
    fn computes_cumulative_and_closed_arc_length() {
        let track = square_track();
        assert!(track.total_length > 40.0);
        assert_eq!(track.samples[0].cumulative_distance, 0.0);
    }

    #[test]
    fn projects_with_interpolated_distance() {
        let track = square_track();
        let point = track.point_at_distance(track.total_length * 0.2);
        let projection = track.project(point);
        assert!(projection.point.distance(point) < 1.0e-3);
        assert!(projection.track_distance > 0.0);
    }

    #[test]
    fn normalizes_progress_and_handles_zero_length() {
        assert!((normalized_progress(15.0, 40.0) - 0.375).abs() < 1e-6);
        assert_eq!(normalized_progress(50.0, 40.0), 1.0);
        assert_eq!(normalized_progress(1.0, 0.0), 0.0);
    }

    #[test]
    fn wrap_delta_is_continuous_in_both_directions() {
        assert!((wrapped_distance_delta(39.0, 1.0, 40.0) - 2.0).abs() < 1e-6);
        assert!((wrapped_distance_delta(1.0, 39.0, 40.0) + 2.0).abs() < 1e-6);
    }

    #[test]
    fn every_bundled_track_parses_generates_and_projects_progress() {
        let library = TrackLibrary::load_default().unwrap();
        assert_eq!(library.all_tracks().count(), 8);

        for definition in library.all_tracks() {
            let track = Track::from_definition(definition).unwrap_or_else(|error| {
                panic!("bundled track {} failed validation: {error}", definition.id)
            });
            assert!(track.total_length > MIN_TOTAL_LENGTH);
            let expected_distance = track.total_length * 0.375;
            let point = track.point_at_distance(expected_distance);
            let projection = track.project(point);
            let progress = normalized_progress(projection.track_distance, track.total_length);
            assert!(
                (progress - 0.375).abs() < 0.01,
                "{} projected progress was {progress}",
                definition.id
            );
        }
    }

    #[test]
    fn library_keeps_training_and_validation_disjoint() {
        let library = TrackLibrary::load_default().unwrap();
        let training = library.training_tracks().collect::<Vec<_>>();
        let validation = library.validation_tracks().collect::<Vec<_>>();

        assert_eq!(training.len(), 5);
        assert_eq!(validation.len(), 3);
        assert!(
            training
                .iter()
                .all(|track| track.role == TrackRole::Training)
        );
        assert!(
            validation
                .iter()
                .all(|track| track.role == TrackRole::Validation)
        );
        assert!(training.iter().all(|training_track| {
            validation
                .iter()
                .all(|validation_track| training_track.id != validation_track.id)
        }));
    }
}
