use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use bevy::prelude::*;

use super::{
    CarProgress, EvaluationState, FinishReason, KinematicCar, SavedNetwork, SelectedCar,
    SimulationMode, Track, checkpoint::CheckpointEntry,
};

pub const MAX_RACE_PARTICIPANTS: usize = 32;
pub const MAX_RACE_TRAIL_POINTS: usize = 4096;
pub const RACE_TRAIL_SAMPLE_TICKS: u8 = 6;
pub const RACE_TRAIL_MIN_DISTANCE: f32 = 4.0;
pub const DEFAULT_RACE_SAFETY_TIMEOUT: f32 = 300.0;

const RACE_PALETTE: [[u8; 3]; 16] = [
    [65, 200, 255],
    [255, 105, 120],
    [105, 225, 135],
    [255, 205, 70],
    [185, 120, 255],
    [255, 145, 65],
    [80, 230, 215],
    [245, 120, 215],
    [150, 220, 75],
    [115, 145, 255],
    [255, 175, 200],
    [205, 190, 110],
    [80, 175, 120],
    [220, 105, 65],
    [155, 155, 235],
    [235, 235, 235],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaceColor(pub [u8; 3]);

impl RaceColor {
    pub fn as_srgba(self, alpha: f32) -> Color {
        Color::srgba_u8(self.0[0], self.0[1], self.0[2], (alpha * 255.0) as u8)
    }
}

#[derive(Clone, Debug)]
pub struct RaceEntry {
    pub source_path: PathBuf,
    pub source_display: String,
    pub display_name: String,
    pub architecture: String,
    pub generation: usize,
    pub saved: SavedNetwork,
    pub color: RaceColor,
    pub sprite_index: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaceRosterAdd {
    Added,
    Duplicate,
}

#[derive(Resource, Debug, Default)]
pub struct RaceRoster {
    entries: Vec<RaceEntry>,
    revision: u64,
    next_visual_index: usize,
}

impl RaceRoster {
    pub fn entries(&self) -> &[RaceEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn add(
        &mut self,
        source_path: PathBuf,
        saved: SavedNetwork,
    ) -> Result<RaceRosterAdd, String> {
        saved.validate().map_err(|error| error.to_string())?;
        let identity = normalized_checkpoint_identity(&source_path);
        if self
            .entries
            .iter()
            .any(|entry| normalized_checkpoint_identity(&entry.source_path) == identity)
        {
            return Ok(RaceRosterAdd::Duplicate);
        }
        if self.entries.len() >= MAX_RACE_PARTICIPANTS {
            return Err(format!(
                "race roster is limited to {MAX_RACE_PARTICIPANTS} participants"
            ));
        }

        let visual_index = self.next_visual_index;
        self.next_visual_index = self.next_visual_index.wrapping_add(1);
        let source_display = source_path.display().to_string();
        let filename = source_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(&source_display)
            .to_string();
        self.entries.push(RaceEntry {
            source_path,
            source_display,
            display_name: filename,
            architecture: saved.architecture.display(),
            generation: saved.generation,
            saved,
            color: RaceColor(RACE_PALETTE[visual_index % RACE_PALETTE.len()]),
            sprite_index: visual_index,
        });
        self.revision = self.revision.wrapping_add(1);
        Ok(RaceRosterAdd::Added)
    }

    pub fn remove(&mut self, index: usize) -> bool {
        if index >= self.entries.len() {
            return false;
        }
        self.entries.remove(index);
        self.revision = self.revision.wrapping_add(1);
        true
    }

    pub fn clear(&mut self) -> bool {
        if self.entries.is_empty() {
            return false;
        }
        self.entries.clear();
        self.revision = self.revision.wrapping_add(1);
        true
    }
}

fn normalized_checkpoint_identity(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

pub fn latest_checkpoint_indices(entries: &[CheckpointEntry]) -> Vec<usize> {
    let mut latest = BTreeMap::<String, (usize, String, usize)>::new();
    for (index, entry) in entries.iter().enumerate() {
        let Some(summary) = &entry.summary else {
            continue;
        };
        let path = entry.path.display().to_string();
        let candidate = (summary.generation, path, index);
        match latest.get_mut(&summary.architecture) {
            Some(current)
                if candidate.0 > current.0
                    || (candidate.0 == current.0 && candidate.1 < current.1) =>
            {
                *current = candidate;
            }
            None => {
                latest.insert(summary.architecture.clone(), candidate);
            }
            _ => {}
        }
    }
    latest.into_values().map(|(_, _, index)| index).collect()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RacePhase {
    #[default]
    Ready,
    Running,
    Finished,
}

#[derive(Resource, Clone, Debug)]
pub struct RaceSession {
    pub phase: RacePhase,
    pub elapsed: f32,
    pub safety_timeout: f32,
    pub show_trails: bool,
    pub auto_select_leader: bool,
    pub inspected_roster_index: Option<usize>,
    pub status: String,
    pub export_status: String,
    run_revision: u64,
    awaiting_rebuild: bool,
}

impl Default for RaceSession {
    fn default() -> Self {
        Self {
            phase: RacePhase::Ready,
            elapsed: 0.0,
            safety_timeout: DEFAULT_RACE_SAFETY_TIMEOUT,
            show_trails: true,
            auto_select_leader: true,
            inspected_roster_index: None,
            status: "Add checkpoints to the race roster".into(),
            export_status: String::new(),
            run_revision: 0,
            awaiting_rebuild: false,
        }
    }
}

impl RaceSession {
    pub fn run_revision(&self) -> u64 {
        self.run_revision
    }

    pub fn is_simulating(&self) -> bool {
        self.phase == RacePhase::Running && !self.awaiting_rebuild
    }

    pub(crate) fn mark_rebuilt(&mut self) {
        self.awaiting_rebuild = false;
    }

    pub fn prepare(&mut self, participant_count: usize) {
        self.phase = RacePhase::Ready;
        self.awaiting_rebuild = false;
        self.elapsed = 0.0;
        self.inspected_roster_index = self
            .inspected_roster_index
            .filter(|&index| index < participant_count);
        self.status = if participant_count == 0 {
            "Add checkpoints to the race roster".into()
        } else {
            format!("{participant_count} racer(s) ready")
        };
        self.export_status.clear();
    }

    pub fn start_or_restart(&mut self, participant_count: usize) -> bool {
        if participant_count == 0 {
            self.prepare(0);
            return false;
        }
        self.phase = RacePhase::Running;
        self.elapsed = 0.0;
        self.status = format!("Race running with {participant_count} racer(s)");
        self.export_status.clear();
        self.run_revision = self.run_revision.wrapping_add(1);
        self.awaiting_rebuild = true;
        true
    }
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaceParticipant {
    pub roster_index: usize,
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct RaceVisual {
    pub color: RaceColor,
    pub sprite_index: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RaceStatus {
    #[default]
    Running,
    Completed,
    Collision,
    Timeout,
}

impl RaceStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Running => "RUNNING",
            Self::Completed => "COMPLETED",
            Self::Collision => "COLLISION",
            Self::Timeout => "TIMEOUT",
        }
    }

    pub fn is_finished(self) -> bool {
        self != Self::Running
    }
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct RaceMetrics {
    pub current_progress: f32,
    pub best_progress: f32,
    pub normalized_progress: f32,
    pub current_speed: f32,
    pub useful_progress_speed: f32,
    pub elapsed: f32,
    pub status: RaceStatus,
    pub finish_time: Option<f32>,
}

impl RaceMetrics {
    pub fn new(initial_progress: f32) -> Self {
        Self {
            current_progress: initial_progress,
            best_progress: initial_progress,
            normalized_progress: 0.0,
            current_speed: 0.0,
            useful_progress_speed: 0.0,
            elapsed: 0.0,
            status: RaceStatus::Running,
            finish_time: None,
        }
    }
}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct RaceTrail {
    pub points: Vec<Vec2>,
    pub ticks_since_sample: u8,
}

impl RaceTrail {
    pub fn new(start: Vec2) -> Self {
        Self {
            points: vec![start],
            ticks_since_sample: 0,
        }
    }

    pub fn sample(&mut self, position: Vec2) {
        if self.points.len() >= MAX_RACE_TRAIL_POINTS {
            return;
        }
        self.ticks_since_sample = self.ticks_since_sample.saturating_add(1);
        let moved = self
            .points
            .last()
            .is_none_or(|last| last.distance_squared(position) >= RACE_TRAIL_MIN_DISTANCE.powi(2));
        if self.ticks_since_sample >= RACE_TRAIL_SAMPLE_TICKS && moved {
            self.points.push(position);
            self.ticks_since_sample = 0;
        }
    }

    pub fn finish_at(&mut self, position: Vec2) {
        if self.points.len() < MAX_RACE_TRAIL_POINTS
            && self.points.last().is_none_or(|last| *last != position)
        {
            self.points.push(position);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaceStanding {
    pub roster_index: usize,
    pub metrics: RaceMetricsSummary,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RaceMetricsSummary {
    pub current_progress: f32,
    pub best_progress: f32,
    pub normalized_progress: f32,
    pub current_speed: f32,
    pub useful_progress_speed: f32,
    pub elapsed: f32,
    pub status: RaceStatus,
    pub finish_time: Option<f32>,
}

impl From<&RaceMetrics> for RaceMetricsSummary {
    fn from(value: &RaceMetrics) -> Self {
        Self {
            current_progress: value.current_progress,
            best_progress: value.best_progress,
            normalized_progress: value.normalized_progress,
            current_speed: value.current_speed,
            useful_progress_speed: value.useful_progress_speed,
            elapsed: value.elapsed,
            status: value.status,
            finish_time: value.finish_time,
        }
    }
}

pub fn sort_race_standings(standings: &mut [RaceStanding]) {
    standings.sort_by(compare_standings);
}

fn compare_standings(left: &RaceStanding, right: &RaceStanding) -> Ordering {
    let group = |status| match status {
        RaceStatus::Completed => 0,
        RaceStatus::Running => 1,
        RaceStatus::Collision | RaceStatus::Timeout => 2,
    };
    group(left.metrics.status)
        .cmp(&group(right.metrics.status))
        .then_with(|| match left.metrics.status {
            RaceStatus::Completed => left
                .metrics
                .finish_time
                .unwrap_or(f32::INFINITY)
                .total_cmp(&right.metrics.finish_time.unwrap_or(f32::INFINITY))
                .then_with(|| {
                    right
                        .metrics
                        .useful_progress_speed
                        .total_cmp(&left.metrics.useful_progress_speed)
                }),
            RaceStatus::Running => right
                .metrics
                .current_progress
                .total_cmp(&left.metrics.current_progress)
                .then_with(|| {
                    right
                        .metrics
                        .best_progress
                        .total_cmp(&left.metrics.best_progress)
                }),
            RaceStatus::Collision | RaceStatus::Timeout => right
                .metrics
                .best_progress
                .total_cmp(&left.metrics.best_progress)
                .then_with(|| {
                    right
                        .metrics
                        .useful_progress_speed
                        .total_cmp(&left.metrics.useful_progress_speed)
                }),
        })
        .then_with(|| left.roster_index.cmp(&right.roster_index))
}

pub fn export_race_results(
    directory: &Path,
    track_id: &str,
    roster: &RaceRoster,
    standings: &[RaceStanding],
) -> Result<PathBuf, String> {
    fs::create_dir_all(directory)
        .map_err(|error| format!("failed to create {}: {error}", directory.display()))?;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock error: {error}"))?
        .as_secs();
    let safe_track = track_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect::<String>();
    let mut suffix = 0usize;
    let (path, mut file) = loop {
        let filename = if suffix == 0 {
            format!("{timestamp}_{safe_track}.csv")
        } else {
            format!("{timestamp}_{safe_track}_{suffix}.csv")
        };
        let path = directory.join(filename);
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => break (path, file),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => suffix += 1,
            Err(error) => return Err(format!("failed to create {}: {error}", path.display())),
        }
    };
    writeln!(
        file,
        "track,source_checkpoint,architecture,checkpoint_generation,stored_champion_training_fitness,stored_champion_validation_score,race_status,finish_time,final_best_normalized_progress,useful_progress_speed,final_elapsed_time"
    )
    .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    let by_index = standings
        .iter()
        .map(|standing| (standing.roster_index, standing.metrics))
        .collect::<BTreeMap<_, _>>();
    for (index, entry) in roster.entries().iter().enumerate() {
        let metrics = by_index.get(&index).copied().unwrap_or(RaceMetricsSummary {
            current_progress: 0.0,
            best_progress: 0.0,
            normalized_progress: 0.0,
            current_speed: 0.0,
            useful_progress_speed: 0.0,
            elapsed: 0.0,
            status: RaceStatus::Timeout,
            finish_time: None,
        });
        writeln!(
            file,
            "{},{},{},{},{:.8},{:.8},{},{},{:.8},{:.8},{:.8}",
            csv_field(track_id),
            csv_field(&entry.source_display),
            csv_field(&entry.architecture),
            entry.generation,
            entry.saved.training_metadata.champion_training_fitness,
            entry.saved.validation_metadata.score,
            metrics.status.label(),
            metrics
                .finish_time
                .map(|time| format!("{time:.8}"))
                .unwrap_or_default(),
            metrics.normalized_progress,
            metrics.useful_progress_speed,
            metrics.elapsed,
        )
        .map_err(|error| format!("failed to write {}: {error}", path.display()))?;
    }
    Ok(path)
}

fn csv_field(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

#[allow(clippy::type_complexity)]
pub fn update_race_session(
    mut commands: Commands,
    time: Res<Time<Fixed>>,
    mode: Res<SimulationMode>,
    track: Res<Track>,
    mut session: ResMut<RaceSession>,
    mut racers: Query<(
        Entity,
        &RaceParticipant,
        &Transform,
        &mut KinematicCar,
        &CarProgress,
        &mut EvaluationState,
        &mut RaceMetrics,
        &mut RaceTrail,
        Has<SelectedCar>,
    )>,
) {
    if *mode != SimulationMode::Race {
        return;
    }

    if session.is_simulating() {
        let delta = time.delta_secs().max(0.0);
        session.elapsed += delta;
        for (_, _, transform, mut state, progress, mut evaluation, mut metrics, mut trail, _) in
            &mut racers
        {
            if !evaluation.is_finished() {
                evaluation.elapsed += delta;
                if progress.best_track_distance >= track.total_length {
                    evaluation.finish(FinishReason::Completed);
                } else if evaluation.elapsed >= session.safety_timeout {
                    evaluation.finish(FinishReason::Timeout);
                }
            }

            let useful_distance =
                (progress.best_track_distance - evaluation.initial_progress).max(0.0);
            let remaining_distance =
                (track.total_length - evaluation.initial_progress).max(f32::EPSILON);
            metrics.current_progress = progress.track_distance;
            metrics.best_progress = progress.best_track_distance;
            metrics.normalized_progress = (useful_distance / remaining_distance).clamp(0.0, 1.0);
            metrics.current_speed = state.speed;
            metrics.elapsed = evaluation.elapsed;
            metrics.useful_progress_speed = if evaluation.elapsed > f32::EPSILON {
                useful_distance / evaluation.elapsed
            } else {
                0.0
            };
            let previous_status = metrics.status;
            metrics.status = match evaluation.finish_reason {
                None => RaceStatus::Running,
                Some(FinishReason::Completed) => RaceStatus::Completed,
                Some(FinishReason::Collision) => RaceStatus::Collision,
                Some(FinishReason::Timeout) => RaceStatus::Timeout,
                Some(FinishReason::EliminatedByLaser) => {
                    // Race never advances or applies the training laser. Keeping this
                    // impossible branch explicit prevents it from silently becoming a
                    // valid race outcome if training evaluation changes later.
                    RaceStatus::Timeout
                }
            };
            if metrics.status == RaceStatus::Completed && metrics.finish_time.is_none() {
                metrics.finish_time = Some(evaluation.elapsed);
            }
            if metrics.status.is_finished() && !previous_status.is_finished() {
                trail.finish_at(transform.translation.truncate());
            } else if !metrics.status.is_finished() {
                trail.sample(transform.translation.truncate());
            }
            if metrics.status.is_finished() {
                state.speed = 0.0;
            }
        }

        if !racers.is_empty()
            && racers
                .iter()
                .all(|(_, _, _, _, _, _, metrics, _, _)| metrics.status.is_finished())
        {
            session.phase = RacePhase::Finished;
            session.status = format!("Race finished in {:.2}s", session.elapsed);
        }
    }

    let mut standings = racers
        .iter()
        .map(|(_, participant, _, _, _, _, metrics, _, _)| RaceStanding {
            roster_index: participant.roster_index,
            metrics: RaceMetricsSummary::from(&*metrics),
        })
        .collect::<Vec<_>>();
    sort_race_standings(&mut standings);
    let target_index = if session.auto_select_leader {
        standings.first().map(|standing| standing.roster_index)
    } else {
        session.inspected_roster_index
    };
    if session.auto_select_leader {
        session.inspected_roster_index = target_index;
    }
    for (entity, participant, _, _, _, _, _, _, selected) in &racers {
        let should_select = Some(participant.roster_index) == target_index;
        if should_select && !selected {
            commands.entity(entity).insert(SelectedCar);
        } else if !should_select && selected {
            commands.entity(entity).remove::<SelectedCar>();
        }
    }
}

pub fn apply_race_selection(
    mut commands: Commands,
    mode: Res<SimulationMode>,
    session: Res<RaceSession>,
    racers: Query<(Entity, &RaceParticipant, &RaceMetrics, Has<SelectedCar>)>,
) {
    if *mode != SimulationMode::Race {
        return;
    }
    let mut standings = racers
        .iter()
        .map(|(_, participant, metrics, _)| RaceStanding {
            roster_index: participant.roster_index,
            metrics: RaceMetricsSummary::from(metrics),
        })
        .collect::<Vec<_>>();
    sort_race_standings(&mut standings);
    let target = if session.auto_select_leader {
        standings.first().map(|standing| standing.roster_index)
    } else {
        session.inspected_roster_index
    };
    for (entity, participant, _, selected) in &racers {
        let should_select = target == Some(participant.roster_index);
        if should_select && !selected {
            commands.entity(entity).insert(SelectedCar);
        } else if !should_select && selected {
            commands.entity(entity).remove::<SelectedCar>();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::simulation::{
        CarProgress, EvaluationState, FinishReason, KinematicCar, LaserState, TrackLibrary,
        checkpoint::{
            CHECKPOINT_FORMAT_VERSION, CheckpointSummary, SavedActivation, SavedArchitecture,
            SavedEvaluationParameters, SavedFinishReason, SavedFinishReasonCounts,
            SavedLaserConfig, SavedTrainingMetadata, SavedTrainingTrackSelection,
            SavedUsefulSpeedNormalization, SavedValidationMetadata,
        },
    };

    fn saved_network(generation: usize, layers: &[usize]) -> SavedNetwork {
        let parameter_count = layers
            .windows(2)
            .map(|pair| pair[0] * pair[1] + pair[1])
            .sum();
        SavedNetwork {
            format_version: CHECKPOINT_FORMAT_VERSION,
            saved_at_unix_seconds: 1,
            generation,
            architecture: SavedArchitecture {
                layer_sizes: layers.to_vec(),
                activations: vec![SavedActivation::Tanh; layers.len() - 1],
            },
            genome: vec![0.0; parameter_count],
            training_metadata: SavedTrainingMetadata {
                champion_training_fitness: 1.25,
                population_average_fitness: 0.75,
                average_useful_progress_speed: 80.0,
                completion_rate: 0.5,
                finish_counts: SavedFinishReasonCounts::default(),
                training_tracks: vec!["interlagos".into()],
            },
            validation_metadata: SavedValidationMetadata {
                track_id: "spa".into(),
                score: 1.0,
                normalized_progress: 0.8,
                useful_progress_speed: 75.0,
                elapsed: 10.0,
                finish_reason: SavedFinishReason::Collision,
            },
            evaluation_parameters: SavedEvaluationParameters {
                maximum_episode_duration: 180.0,
                stall_timeout: 0.0,
                significant_progress_epsilon: 0.0,
                laser: Some(SavedLaserConfig {
                    grace_period: 3.0,
                    acceleration: 30.0,
                    maximum_speed: 130.0,
                }),
                sensor_max_distance: Some(1000.0),
                progress_weight: 1.0,
                speed_weight: 0.4,
                collision_penalty: 0.08,
                completion_bonus: 0.45,
                useful_speed_normalization: Some(
                    SavedUsefulSpeedNormalization::AsymptoticHalfSaturation,
                ),
                progress_speed_half_saturation: 120.0,
                training_track_selection: SavedTrainingTrackSelection::All,
            },
        }
    }

    fn entry(path: &str, generation: usize, layers: &[usize]) -> CheckpointEntry {
        let saved = saved_network(generation, layers);
        CheckpointEntry {
            path: path.into(),
            folder: "results".into(),
            filename: path.into(),
            summary: Some(CheckpointSummary::from(&saved)),
            error: None,
        }
    }

    fn metrics(
        status: RaceStatus,
        progress: f32,
        speed: f32,
        finish: Option<f32>,
    ) -> RaceMetricsSummary {
        RaceMetricsSummary {
            current_progress: progress,
            best_progress: progress,
            normalized_progress: progress,
            current_speed: speed,
            useful_progress_speed: speed,
            elapsed: 10.0,
            status,
            finish_time: finish,
        }
    }

    #[test]
    fn roster_add_duplicate_remove_clear_and_revision_are_deterministic() {
        let mut roster = RaceRoster::default();
        assert_eq!(
            roster.add("a.ron".into(), saved_network(100, &[6, 16, 2])),
            Ok(RaceRosterAdd::Added)
        );
        let revision = roster.revision();
        assert_eq!(
            roster.add("a.ron".into(), saved_network(100, &[6, 16, 2])),
            Ok(RaceRosterAdd::Duplicate)
        );
        assert_eq!(roster.revision(), revision);
        assert_eq!(
            roster.add("b.ron".into(), saved_network(500, &[6, 16, 2])),
            Ok(RaceRosterAdd::Added)
        );
        assert_eq!(
            roster.len(),
            2,
            "same architecture at another source is allowed"
        );
        let second_color = roster.entries()[1].color;
        assert!(roster.remove(0));
        assert_eq!(roster.entries()[0].color, second_color);
        assert!(roster.clear());
        assert!(roster.is_empty());
        assert!(roster.revision() > revision);
    }

    #[test]
    fn latest_per_architecture_selects_highest_generation_without_fixed_architecture_list() {
        let entries = vec![
            entry("a100.ron", 100, &[6, 8, 2]),
            entry("a500.ron", 500, &[6, 8, 2]),
            entry("b300.ron", 300, &[6, 16, 16, 2]),
        ];
        let indices = latest_checkpoint_indices(&entries);
        assert_eq!(indices.len(), 2);
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));
    }

    #[test]
    fn dynamic_checkpoint_architectures_reconstruct_valid_mlps() {
        for layers in [
            vec![6, 8, 2],
            vec![6, 32, 2],
            vec![6, 8, 8, 2],
            vec![6, 16, 32, 2],
        ] {
            let (_, mlp) = saved_network(1, &layers).reconstruct().unwrap();
            assert_eq!(mlp.layer_sizes(), layers);
        }
    }

    #[test]
    fn ranking_orders_completed_running_and_failed_deterministically() {
        let mut standings = vec![
            RaceStanding {
                roster_index: 4,
                metrics: metrics(RaceStatus::Collision, 0.9, 90.0, None),
            },
            RaceStanding {
                roster_index: 3,
                metrics: metrics(RaceStatus::Running, 0.8, 80.0, None),
            },
            RaceStanding {
                roster_index: 2,
                metrics: metrics(RaceStatus::Completed, 1.0, 70.0, Some(42.0)),
            },
            RaceStanding {
                roster_index: 1,
                metrics: metrics(RaceStatus::Completed, 1.0, 60.0, Some(39.0)),
            },
            RaceStanding {
                roster_index: 0,
                metrics: metrics(RaceStatus::Running, 0.95, 50.0, None),
            },
        ];
        sort_race_standings(&mut standings);
        assert_eq!(
            standings.iter().map(|s| s.roster_index).collect::<Vec<_>>(),
            vec![1, 2, 0, 3, 4]
        );
    }

    fn race_app(progress: f32, finish_reason: Option<FinishReason>, timeout: f32) -> App {
        let library = TrackLibrary::load_default().unwrap();
        let track = Track::from_definition(library.definition("interlagos").unwrap()).unwrap();
        let mut fixed = Time::<Fixed>::from_hz(60.0);
        fixed.advance_by(Duration::from_secs_f32(1.0 / 60.0));
        let mut session = RaceSession {
            safety_timeout: timeout,
            ..default()
        };
        session.start_or_restart(1);
        session.mark_rebuilt();
        let mut evaluation = EvaluationState::new(0.0);
        if let Some(reason) = finish_reason {
            evaluation.finish(reason);
        }
        let mut car_progress = CarProgress::new(progress, track.total_length, 0, Vec2::ZERO);
        car_progress.best_track_distance = progress;
        let mut app = App::new();
        app.insert_resource(fixed)
            .insert_resource(SimulationMode::Race)
            .insert_resource(track)
            .insert_resource(session)
            .insert_resource(LaserState {
                elapsed: 1000.0,
                progress: 1000.0,
                ..default()
            })
            .add_systems(FixedUpdate, update_race_session);
        app.world_mut().spawn((
            RaceParticipant { roster_index: 0 },
            Transform::default(),
            KinematicCar {
                heading: 0.0,
                speed: 100.0,
            },
            car_progress,
            evaluation,
            RaceMetrics::new(0.0),
            RaceTrail::new(Vec2::ZERO),
        ));
        app
    }

    #[test]
    fn race_clock_uses_fixed_delta_and_training_laser_is_ignored() {
        let mut app = race_app(25.0, None, 300.0);
        app.world_mut().run_schedule(FixedUpdate);
        let session = app.world().resource::<RaceSession>();
        assert!((session.elapsed - 1.0 / 60.0).abs() < 1.0e-6);
        let evaluation = app
            .world_mut()
            .query::<&EvaluationState>()
            .single(app.world())
            .unwrap();
        assert_eq!(evaluation.finish_reason, None);
    }

    #[test]
    fn collision_completion_and_safety_timeout_map_to_race_status() {
        for (progress, reason, timeout, expected) in [
            (
                10.0,
                Some(FinishReason::Collision),
                300.0,
                RaceStatus::Collision,
            ),
            (f32::MAX, None, 300.0, RaceStatus::Completed),
            (10.0, None, 0.001, RaceStatus::Timeout),
        ] {
            let mut app = race_app(progress, reason, timeout);
            app.world_mut().run_schedule(FixedUpdate);
            let status = app
                .world_mut()
                .query::<&RaceMetrics>()
                .single(app.world())
                .unwrap()
                .status;
            assert_eq!(status, expected);
            assert_eq!(
                app.world().resource::<RaceSession>().phase,
                RacePhase::Finished
            );
        }
    }

    #[test]
    fn restart_advances_revision_and_resets_session_results() {
        let mut session = RaceSession::default();
        assert!(session.start_or_restart(2));
        let first = session.run_revision();
        session.elapsed = 12.0;
        session.phase = RacePhase::Finished;
        assert!(session.start_or_restart(2));
        assert!(session.run_revision() > first);
        assert_eq!(session.elapsed, 0.0);
        assert_eq!(session.phase, RacePhase::Running);
    }

    #[test]
    fn csv_export_contains_exactly_one_data_line_per_racer() {
        let mut roster = RaceRoster::default();
        roster
            .add("a.ron".into(), saved_network(1, &[6, 8, 2]))
            .unwrap();
        roster
            .add("b.ron".into(), saved_network(2, &[6, 16, 2]))
            .unwrap();
        let standings = vec![
            RaceStanding {
                roster_index: 0,
                metrics: metrics(RaceStatus::Completed, 1.0, 100.0, Some(40.0)),
            },
            RaceStanding {
                roster_index: 1,
                metrics: metrics(RaceStatus::Collision, 0.7, 80.0, None),
            },
        ];
        let directory = std::env::temp_dir().join(format!(
            "neuroevolution_race_export_{}_{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let path = export_race_results(&directory, "test/track", &roster, &standings).unwrap();
        let csv = fs::read_to_string(&path).unwrap();
        assert_eq!(csv.lines().count(), 3);
        assert!(csv.contains("stored_champion_training_fitness"));
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
