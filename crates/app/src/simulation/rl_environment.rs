//! RL episodes reuse the exact kinematic integrator, collision resolver, track geometry
//! and Avian wall queries. Static scenes contain WALLS ONLY: cars are private state.
mod reward;
use super::{
    CAR_LENGTH, CAR_WIDTH, SimulationConfig, Track, TrackLibrary,
    components::SimulationLayer,
    controller::controls_from_network_outputs,
    systems::{
        VehicleState, integrate_vehicle, normalize_speed, resolve_vehicle_motion,
        spawn_track_colliders,
    },
    track::wrapped_distance_delta,
};
use avian2d::prelude::*;
use bevy::{ecs::system::SystemState, prelude::*, time::TimeUpdateStrategy};
use racing_rl::{
    Result,
    config::{Config, ObservationKind, SpeedNormalization},
    ensure,
    rollout::{EndReason, Environment, RewardParts, Step},
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha12Rng;
use std::{cell::RefCell, collections::BTreeMap, rc::Rc, time::Duration};

/// Reuse GA's overflow-safe magnitude without changing GA's unsigned behavior.
/// NaN maps to zero; infinities map to their signed limit. Config rejects bad scales.
fn normalize_signed_speed(speed: f32, scale: f32) -> f32 {
    if !scale.is_finite() || scale <= 0. || speed.is_nan() {
        return 0.;
    }
    normalize_speed(speed, scale).copysign(speed)
}

pub struct Scene {
    pub track: Track,
    app: App,
    queries: RefCell<SystemState<SpatialQuery<'static, 'static>>>,
}
impl Scene {
    pub fn new(track: Track) -> Self {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .add_plugins(TransformPlugin)
            .add_plugins(PhysicsPlugins::default().with_length_unit(20.))
            .insert_resource(Gravity::ZERO)
            .insert_resource(Time::<Fixed>::from_hz(60.))
            .insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(
                1. / 60.,
            )));
        {
            let mut commands = app.world_mut().commands();
            spawn_track_colliders(&mut commands, &track);
        }
        app.world_mut().flush();
        // App::run normally performs this lifecycle. These static query-only apps
        // are advanced manually, so finish plugins (including Avian diagnostics) first.
        app.finish();
        app.cleanup();
        app.update();
        app.update();
        app.update();
        let queries = RefCell::new(SystemState::new(app.world_mut()));
        Self {
            track,
            app,
            queries,
        }
    }
    fn intersects(&self, position: Vec2, heading: f32) -> bool {
        let mut state = self.queries.borrow_mut();
        let query = state
            .get(self.app.world())
            .expect("static Avian scene initialized and never mutated");
        !query
            .shape_intersections(
                &Collider::rectangle(CAR_LENGTH, CAR_WIDTH),
                position,
                heading,
                &SpatialQueryFilter::from_mask([SimulationLayer::TrackWall]),
            )
            .is_empty()
    }
    fn ray(&self, origin: Vec2, angle: f32, range: f32) -> f32 {
        let mut state = self.queries.borrow_mut();
        let query = state
            .get(self.app.world())
            .expect("static Avian scene initialized and never mutated");
        query
            .cast_ray(
                origin,
                Dir2::new(Vec2::from_angle(angle)).unwrap(),
                range,
                true,
                &SpatialQueryFilter::from_mask([SimulationLayer::TrackWall]),
            )
            .map_or(range, |hit| hit.distance)
    }
}
#[derive(Clone, Debug, Default)]
struct Progress {
    previous: f32,
    cumulative: f32,
    best: f32,
}
impl Progress {
    #[cfg(test)]
    fn advance(&mut self, projected: f32, length: f32, movement: f32) -> f32 {
        self.advance_connected(projected, length, movement, false)
    }
    fn advance_connected(
        &mut self,
        projected: f32,
        length: f32,
        movement: f32,
        adjacent: bool,
    ) -> f32 {
        let delta = wrapped_distance_delta(self.previous, projected, length);
        // Like GA's centerline_distance, the reference follows every projection.
        self.previous = projected;
        // Reject discontinuities rather than crediting teleportation or a wrong branch.
        if delta.abs() > 4. * movement + 2. && !adjacent {
            // Do not join disconnected forward stretches into a fictitious lap.
            // Keep the high-water mark and any existing reverse-motion debt.
            self.cumulative = self.cumulative.min(0.);
            return 0.;
        }
        self.cumulative += delta;
        let best = self.cumulative.clamp(0., length).max(self.best);
        let gain = best - self.best;
        self.best = best;
        gain
    }
}

/// Physical query events, not episode endings. Rotation can be rejected safely
/// while translation continues. Episode resets deliberately retain these counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct CollisionEvents {
    pub physical_ticks: u64,
    pub translation: u64,
    pub rotation: u64,
    pub rotation_only: u64,
}

/// Lightweight accumulators for decisions since the caller last drained them.
/// Resets retain these values, so episodes ending inside a rollout are included.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct TrackMetrics {
    pub decisions: u64,
    pub physical_ticks: u64,
    pub reward_sum: f64,
    pub progress_reward_sum: f64,
    pub progress_gain: f64,
    pub progress_fraction_sum: f64,
    pub progress_fraction_max: f32,
    pub episodes_completed: u64,
    pub episodes_collision: u64,
    pub episodes_timeout: u64,
    /// Final progress of all ended episodes, including collisions and timeouts.
    pub ended_episode_progress_sum: f64,
    pub collision_events: CollisionEvents,
}
impl TrackMetrics {
    pub fn merge(&mut self, other: &Self) {
        self.decisions += other.decisions;
        self.physical_ticks += other.physical_ticks;
        self.reward_sum += other.reward_sum;
        self.progress_reward_sum += other.progress_reward_sum;
        self.progress_gain += other.progress_gain;
        self.progress_fraction_sum += other.progress_fraction_sum;
        self.progress_fraction_max = self.progress_fraction_max.max(other.progress_fraction_max);
        self.episodes_completed += other.episodes_completed;
        self.episodes_collision += other.episodes_collision;
        self.episodes_timeout += other.episodes_timeout;
        self.ended_episode_progress_sum += other.ended_episode_progress_sum;
        self.collision_events.physical_ticks += other.collision_events.physical_ticks;
        self.collision_events.translation += other.collision_events.translation;
        self.collision_events.rotation += other.collision_events.rotation;
        self.collision_events.rotation_only += other.collision_events.rotation_only;
    }
}

pub struct RacingEnvironment {
    pub env_id: usize,
    pub scenes: Vec<Rc<Scene>>,
    pub scene_index: usize,
    pub state: VehicleState,
    pub rng: ChaCha12Rng,
    config: Config,
    pub physics: SimulationConfig,
    pub collision_events: CollisionEvents,
    pub track_metrics: BTreeMap<String, TrackMetrics>,
    episode: u64,
    ticks: u64,
    previous_action: [f32; 2],
    yaw_rate: f32,
    progress: Progress,
    progress_position: Vec2,
    nearest: usize,
    finished: bool,
}
fn movement_is_connected(stored: Vec2, before: Vec2, old: usize, new: usize, count: usize) -> bool {
    stored == before && old != new && ((old + 1) % count == new || (new + 1) % count == old)
}
impl RacingEnvironment {
    pub fn new(
        env_id: usize,
        scenes: Vec<Rc<Scene>>,
        config: &Config,
        physics: &SimulationConfig,
        rng: ChaCha12Rng,
    ) -> Result<Self> {
        ensure(!scenes.is_empty(), "empty track suite")?;
        let mut env = Self {
            env_id,
            scenes,
            scene_index: 0,
            state: VehicleState {
                position: Vec2::ZERO,
                heading: 0.,
                speed: 0.,
            },
            rng,
            config: config.clone(),
            physics: physics.clone(),
            collision_events: CollisionEvents::default(),
            track_metrics: BTreeMap::new(),
            episode: 0,
            ticks: 0,
            previous_action: [0.; 2],
            yaw_rate: 0.,
            progress: Progress::default(),
            progress_position: Vec2::ZERO,
            nearest: 0,
            finished: false,
        };
        env.reset_state()?;
        Ok(env)
    }
    fn reset_state(&mut self) -> Result<()> {
        self.scene_index = self.rng.random_range(0..self.scenes.len());
        let scene = &self.scenes[self.scene_index];
        // Same canonical starting distance/speed as GA, with no perturbation of GA's RNG.
        let position = scene.track.point_at_distance(16.);
        let projection = scene.track.project(position);
        let tangent = scene.track.samples[projection.segment_index].tangent;
        self.state = VehicleState {
            position,
            heading: tangent.to_angle(),
            speed: 45.,
        };
        self.progress = Progress {
            previous: projection.track_distance,
            ..Default::default()
        };
        self.nearest = projection.segment_index;
        self.progress_position = position;
        self.previous_action = [0.; 2];
        self.yaw_rate = 0.;
        self.ticks = 0;
        self.finished = false;
        ensure(
            !scene.intersects(position, self.state.heading),
            "RL spawn overlaps a wall",
        )
    }
    pub fn progress_fraction(&self) -> f32 {
        self.progress.best / self.scenes[self.scene_index].track.total_length
    }
    #[cfg(test)]
    pub(crate) fn diagnostic_progress(&self) -> [f32; 4] {
        [
            self.progress.previous,
            self.progress.cumulative,
            self.progress.best,
            self.scenes[self.scene_index].track.total_length,
        ]
    }
    pub fn take_track_metrics(&mut self) -> BTreeMap<String, TrackMetrics> {
        std::mem::take(&mut self.track_metrics)
    }
}
impl Environment for RacingEnvironment {
    fn env_id(&self) -> usize {
        self.env_id
    }
    fn observe(&mut self) -> Result<Vec<f32>> {
        let scene = &self.scenes[self.scene_index];
        let spec = &self.config.observation;
        let mut obs: Vec<_> = spec
            .angles()
            .iter()
            .map(|angle| {
                (scene.ray(
                    self.state.position,
                    self.state.heading + angle.to_radians(),
                    spec.sensor_range,
                ) / spec.sensor_range)
                    .clamp(0., 1.)
            })
            .collect();
        match spec.kind {
            ObservationKind::Legacy6 => {
                obs.push(normalize_speed(self.state.speed, spec.speed_scale))
            }
            ObservationKind::Extended { .. } => {
                obs.push(match spec.speed_normalization {
                    SpeedNormalization::LinearClamp => {
                        (self.state.speed / spec.speed_scale).clamp(-1., 1.)
                    }
                    SpeedNormalization::SignedAsymptotic => {
                        normalize_signed_speed(self.state.speed, spec.speed_scale)
                    }
                });
                obs.push((self.yaw_rate / spec.angular_speed_scale).clamp(-1., 1.));
                obs.extend(self.previous_action);
            }
        }
        racing_rl::finite(&obs, "environment observation")?;
        Ok(obs)
    }
    fn reset(&mut self) -> Result<Vec<f32>> {
        self.episode += 1;
        self.reset_state()?;
        self.observe()
    }
    fn episode_id(&self) -> u64 {
        self.episode
    }
    fn step(&mut self, action: [f32; 2]) -> Result<Step> {
        ensure(!self.finished, "step after done: reset this environment")?;
        ensure(
            action.iter().all(|x| x.is_finite() && x.abs() <= 1.),
            "invalid continuous action [steering,acceleration]",
        )?;
        let controls = controls_from_network_outputs(action);
        let mut reward = RewardParts::default();
        let mut reason = None;
        let mut ticks = 0;
        let mut progress_gain = 0.;
        let previous_events = self.collision_events;
        for _ in 0..self.config.action_repeat {
            let scene = &self.scenes[self.scene_index];
            let before = self.state;
            let integrated =
                integrate_vehicle(before, controls, &self.physics, self.config.physical_dt);
            let resolved = resolve_vehicle_motion(before, integrated, |position, heading| {
                scene.intersects(position, heading)
            });
            self.collision_events.physical_ticks += 1;
            self.collision_events.translation += u64::from(resolved.translation_collided);
            self.collision_events.rotation += u64::from(resolved.rotation_collided);
            self.collision_events.rotation_only +=
                u64::from(resolved.rotation_collided && !resolved.translation_collided);
            self.state = resolved.state;
            self.yaw_rate = (self.state.heading - before.heading) / self.config.physical_dt;
            self.previous_action = action;
            self.ticks += 1;
            ticks += 1;
            let projected = scene.track.project_near(
                self.state.position,
                self.nearest,
                self.physics.progress_search_radius,
            );
            // Nearest-point projection on a polyline is discontinuous outside a
            // corner: switching to its adjacent segment can move the projected
            // arclength farther than the physical displacement. This is connected
            // motion, not a branch jump. Require continuous stored vehicle state;
            // only immediate topological neighbors qualify (including the seam).
            let segment_count = scene.track.samples.len();
            let adjacent = movement_is_connected(
                self.progress_position,
                before.position,
                self.nearest,
                projected.segment_index,
                segment_count,
            );
            let relocated = self.progress_position != before.position;
            self.progress_position = self.state.position;
            self.nearest = projected.segment_index;
            let gain = if relocated {
                self.progress.previous = projected.track_distance;
                self.progress.cumulative = self.progress.cumulative.min(0.);
                0.
            } else {
                self.progress.advance_connected(
                    projected.track_distance,
                    scene.track.total_length,
                    before.position.distance(self.state.position),
                    adjacent,
                )
            };
            progress_gain += f64::from(gain);
            reward.progress += reward::progress_reward(
                &self.config.reward,
                gain,
                scene.track.total_length,
                self.config.physical_dt,
            );
            if resolved.translation_collided {
                // GA evaluation stops the car on terminal translation collisions.
                self.state.speed = 0.;
                reason = Some(EndReason::Collision);
                reward.collision = -self.config.reward.collision_penalty;
            } else if self.progress.best >= scene.track.total_length - 0.01 {
                reason = Some(EndReason::Completed);
                reward.completion = self.config.reward.completion_bonus;
            } else if self.ticks >= self.config.timeout_ticks {
                reason = Some(EndReason::Timeout);
            }
            if reason.is_some() {
                self.finished = true;
                break;
            }
        }
        racing_rl::finite(
            &[
                self.state.position.x,
                self.state.position.y,
                self.state.heading,
                self.state.speed,
                reward.total(),
            ],
            "physical step",
        )?;
        let observation = self.observe()?;
        let progress = self.progress_fraction();
        let track_id = &self.scenes[self.scene_index].track.definition.id;
        if !self.track_metrics.contains_key(track_id) {
            self.track_metrics
                .insert(track_id.clone(), TrackMetrics::default());
        }
        let metrics = self.track_metrics.get_mut(track_id).unwrap();
        metrics.decisions += 1;
        metrics.physical_ticks += ticks;
        metrics.reward_sum += f64::from(reward.total());
        metrics.progress_reward_sum += f64::from(reward.progress);
        metrics.progress_gain += progress_gain;
        metrics.progress_fraction_sum += f64::from(progress);
        metrics.progress_fraction_max = metrics.progress_fraction_max.max(progress);
        metrics.episodes_completed += u64::from(reason == Some(EndReason::Completed));
        metrics.episodes_collision += u64::from(reason == Some(EndReason::Collision));
        metrics.episodes_timeout += u64::from(reason == Some(EndReason::Timeout));
        if reason.is_some() {
            metrics.ended_episode_progress_sum += f64::from(progress);
        }
        metrics.collision_events.physical_ticks +=
            self.collision_events.physical_ticks - previous_events.physical_ticks;
        metrics.collision_events.translation +=
            self.collision_events.translation - previous_events.translation;
        metrics.collision_events.rotation +=
            self.collision_events.rotation - previous_events.rotation;
        metrics.collision_events.rotation_only +=
            self.collision_events.rotation_only - previous_events.rotation_only;
        Ok(Step {
            observation,
            reward,
            terminated: matches!(reason, Some(EndReason::Collision | EndReason::Completed)),
            truncated: reason == Some(EndReason::Timeout),
            reason,
            physical_ticks: ticks,
            progress,
        })
    }
}
pub fn scenes(library: &TrackLibrary, ids: &[String], evaluation: bool) -> Result<Vec<Rc<Scene>>> {
    let definitions: Vec<_> = if ids.is_empty() {
        if evaluation {
            library.validation_tracks().collect()
        } else {
            library.training_tracks().collect()
        }
    } else {
        ids.iter()
            .map(|id| {
                library
                    .definition(id)
                    .ok_or_else(|| racing_rl::Error::Invalid(format!("unknown track {id}")))
            })
            .collect::<Result<_>>()?
    };
    definitions
        .into_iter()
        .map(|d| {
            ensure(
                (d.role == super::track::TrackRole::Validation) == evaluation,
                "training/evaluation track leakage",
            )?;
            let track =
                Track::from_definition(d).map_err(|e| racing_rl::Error::Invalid(e.to_string()))?;
            Ok(Rc::new(Scene::new(track)))
        })
        .collect()
}
pub fn environments(
    scenes: Vec<Rc<Scene>>,
    c: &Config,
    physics: &SimulationConfig,
    rngs: &[ChaCha12Rng],
) -> Result<Vec<RacingEnvironment>> {
    ensure(rngs.len() == c.num_envs, "env RNGs")?;
    (0..c.num_envs)
        .map(|i| RacingEnvironment::new(i, scenes.clone(), c, physics, rngs[i].clone()))
        .collect()
}
pub fn seeded_environments(
    scenes: Vec<Rc<Scene>>,
    c: &Config,
    physics: &SimulationConfig,
    seed: u64,
) -> Result<Vec<RacingEnvironment>> {
    let rngs: Vec<_> = (0..c.num_envs)
        .map(|i| ChaCha12Rng::seed_from_u64(seed.wrapping_add(i as u64)))
        .collect();
    environments(scenes, c, physics, &rngs)
}

#[cfg(test)]
mod parity_tests;
#[cfg(test)]
mod progress_regression_tests;
#[cfg(test)]
mod reward_tests;
#[cfg(test)]
mod track_metrics_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use racing_rl::backend::Cpu;
    use racing_rl::network::Network;
    fn fixture() -> (Config, Vec<Rc<Scene>>) {
        let c = Config {
            num_envs: 2,
            rollout_length: 4,
            minibatch_size: 4,
            hidden: vec![8],
            training_tracks: vec!["interlagos".into()],
            ..Default::default()
        };
        let library = TrackLibrary::load_default().unwrap();
        let scenes = scenes(&library, &c.training_tracks, false).unwrap();
        (c, scenes)
    }
    #[test]
    fn progress_standing_reverse_oscillation_seam_and_invalid_jump() {
        let mut p = Progress {
            previous: 10.,
            ..Default::default()
        };
        assert_eq!(p.advance(10., 100., 0.), 0.);
        assert_eq!(p.advance(9., 100., 1.), 0.);
        assert_eq!(p.advance(10., 100., 1.), 0.);
        assert_eq!(p.advance(12., 100., 2.), 2.);
        assert_eq!(p.advance(10., 100., 2.), 0.);
        assert_eq!(p.advance(12., 100., 2.), 0.);
        assert_eq!(p.advance(50., 100., 0.), 0.);
        let mut seam = Progress {
            previous: 99.,
            ..Default::default()
        };
        assert_eq!(seam.advance(1., 100., 2.), 2.);
        assert_eq!(seam.advance(99., 100., 2.), 0.);
        assert_eq!(seam.advance(1., 100., 2.), 0.);
    }
    #[test]
    fn cars_and_other_track_walls_cannot_interact_and_reset_is_partial() {
        let (c, scenes) = fixture();
        let physics = SimulationConfig::default();
        let mut one = seeded_environments(scenes.clone(), &c, &physics, 7).unwrap();
        let mut two = seeded_environments(scenes, &c, &physics, 7).unwrap();
        let before = one[1].observe().unwrap();
        let id = one[1].episode_id();
        one[0].state.position += Vec2::splat(10000.);
        one[0].reset().unwrap();
        assert_eq!(one[1].observe().unwrap(), before);
        assert_eq!(one[1].episode_id(), id);
        // A different scene can contain a wall at this coordinate, but isn't in this scene's queries.
        let source = &two[0].scenes[0].track;
        let shift = one[1].state.position - source.left_border[0];
        let mut definition = source.definition.clone();
        for point in &mut definition.control_points {
            point[0] += shift.x;
            point[1] += shift.y;
        }
        let other = Scene::new(Track::from_definition(&definition).unwrap());
        assert!(other.intersects(one[1].state.position, one[1].state.heading));
        assert_eq!(one[1].observe().unwrap(), before);
        for _ in 0..15 {
            let a = one[1].step([0.2, 0.7]).unwrap();
            let b = two[1].step([0.2, 0.7]).unwrap();
            assert_eq!(a.observation, b.observation);
            assert_eq!(a.reward.total(), b.reward.total());
        }
        assert_eq!(one[1].state, two[1].state);
        assert_eq!(one[1].env_id, 1);
    }
    #[test]
    fn post_step_sensors_actions_yaw_and_timeout_final_before_reset() {
        let (mut c, scenes) = fixture();
        c.action_repeat = 5;
        c.timeout_ticks = 2;
        let mut envs = seeded_environments(scenes, &c, &SimulationConfig::default(), 3).unwrap();
        let e = &mut envs[0];
        let old = e.observe().unwrap();
        let step = e.step([0.6, 1.]).unwrap();
        assert_eq!(step.physical_ticks, 2);
        assert!(step.truncated && !step.terminated);
        assert_eq!(step.reason, Some(EndReason::Timeout));
        assert_eq!(step.observation, e.observe().unwrap());
        assert_ne!(&step.observation[..21], &old[..21]);
        assert_eq!(&step.observation[23..], [0.6, 1.]);
        assert_ne!(step.observation[22], 0.);
        let final_obs = step.observation;
        let reset = e.reset().unwrap();
        assert_ne!(final_obs, reset);
        assert_eq!(&reset[22..], [0., 0., 0.]);
        assert_eq!(e.progress.best, 0.);
    }
    #[test]
    fn collision_completion_bonus_once_and_no_reward_for_reset() {
        let (c, scenes) = fixture();
        let mut envs = seeded_environments(scenes, &c, &SimulationConfig::default(), 4).unwrap();
        let e = &mut envs[0];
        e.state.position = e.scenes[0].track.left_border[0];
        let hit = e.step([0., 1.]).unwrap();
        assert_eq!(hit.reason, Some(EndReason::Collision));
        assert_eq!(hit.reward.collision, -c.reward.collision_penalty);
        assert!(e.step([0., 1.]).is_err());
        e.reset().unwrap();
        assert_eq!(e.progress.best, 0.);
        let length = e.scenes[0].track.total_length;
        e.progress.best = length - 0.1;
        e.progress.cumulative = length - 0.1;
        let completed = e.step([0., 1.]).unwrap();
        assert_eq!(completed.reason, Some(EndReason::Completed));
        assert_eq!(completed.reward.completion, c.reward.completion_bonus);
        assert!(e.step([0., 1.]).is_err());
    }
    #[test]
    fn real_rollout_determinism_resets_at_different_times_and_manual_train_gate() {
        let (mut c, scenes) = fixture();
        c.timeout_ticks = 3;
        let mut a =
            seeded_environments(scenes.clone(), &c, &SimulationConfig::default(), 3).unwrap();
        let mut b = seeded_environments(scenes, &c, &SimulationConfig::default(), 3).unwrap();
        // Different per-env episode ages, without changing the identity mapping.
        a[1].step([0., 0.]).unwrap();
        b[1].step([0., 0.]).unwrap();
        let network = Network::<Cpu>::new(&c, &Default::default()).unwrap();
        let mut rng = ChaCha12Rng::seed_from_u64(c.action_seed);
        let mut rng2 = rng.clone();
        let r = racing_rl::rollout::collect(&network, &mut a, &c, &mut rng, &Default::default())
            .unwrap();
        let r2 = racing_rl::rollout::collect(&network, &mut b, &c, &mut rng2, &Default::default())
            .unwrap();
        r.validate().unwrap();
        for (x, y) in r.transitions.iter().zip(&r2.transitions) {
            assert_eq!(x.observation, y.observation);
            assert_eq!(x.action, y.action);
            assert_eq!(x.next_observation, y.next_observation);
        }
        assert_eq!(r.transitions[3].reason, Some(EndReason::Timeout));
        assert_eq!(r.transitions[4].reason, Some(EndReason::Timeout));
        assert_eq!(r.transitions[5].episode_id, 1);
        match racing_rl::manual::preflight() {
            Err(racing_rl::Error::MissingManualImplementation { function }) => {
                eprintln!("PENDENTE PPO train smoke: {function}")
            }
            Ok(()) => {
                let mut n = network;
                let mut optimizer =
                    racing_rl::optimizer::Optimizer::new(c.optimizer.clone(), &n.parameters());
                let metrics = racing_rl::trainer::update(
                    &mut n,
                    &mut optimizer,
                    &r,
                    &c,
                    &mut rng,
                    &Default::default(),
                )
                .unwrap();
                assert!(!metrics.is_empty());
                assert!(optimizer.step > 0);
            }
            Err(e) => panic!("{e}"),
        }
    }
    #[test]
    fn signed_speed_formula_extremes_and_legacy_observation_are_preserved() {
        for (speed, expected) in [
            (0., 0.),
            (150., 0.375),
            (250., 0.5),
            (500., 2. / 3.),
            (-250., -0.5),
        ] {
            assert!((normalize_signed_speed(speed, 250.) - expected).abs() < 1e-6);
        }
        let speeds = [-1000., -500., -250., -150., 0., 150., 250., 500., 1000.];
        let values = speeds.map(|v| normalize_signed_speed(v, 250.));
        assert!(values.windows(2).all(|p| p[0] < p[1]));
        assert_eq!(normalize_signed_speed(f32::NAN, 250.), 0.);
        assert_eq!(normalize_signed_speed(f32::INFINITY, 250.), 1.);
        assert_eq!(normalize_signed_speed(f32::NEG_INFINITY, 250.), -1.);
        assert_eq!(normalize_signed_speed(f32::MAX, f32::MAX), 0.5);
        assert_eq!(normalize_signed_speed(-f32::MAX, f32::MAX), -0.5);
        for scale in [0., -1., f32::NAN, f32::INFINITY] {
            assert_eq!(normalize_signed_speed(150., scale), 0.);
            let mut spec = racing_rl::config::ObservationSpec::default();
            spec.speed_scale = scale;
            assert!(spec.validate().is_err());
        }
        let (c, scenes) = fixture();
        let mut old =
            seeded_environments(scenes.clone(), &c, &SimulationConfig::default(), 9).unwrap();
        let mut continuous = c.clone();
        continuous.observation.speed_normalization = SpeedNormalization::SignedAsymptotic;
        let mut new =
            seeded_environments(scenes, &continuous, &SimulationConfig::default(), 9).unwrap();
        for speed in [-500., -250., 0., 150., 250., 500., 1000.] {
            old[0].state.speed = speed;
            new[0].state.speed = speed;
            let a = old[0].observe().unwrap();
            let b = new[0].observe().unwrap();
            assert_eq!(a.len(), 25);
            assert_eq!(b.len(), 25);
            assert_eq!(a[21], (speed / 250.).clamp(-1., 1.));
            assert_eq!(b[21], normalize_signed_speed(speed, 250.));
            assert_eq!(&a[..21], &b[..21]);
            assert_eq!(&a[22..], &b[22..]);
        }
    }

    #[test]
    fn held_out_tracks_rejected_from_collection() {
        let library = TrackLibrary::load_default().unwrap();
        let held = library.validation_tracks().next().unwrap().id.clone();
        assert!(scenes(&library, &[held], false).is_err());
    }
    #[test]
    fn real_stationary_and_reverse_motion_do_not_earn_progress() {
        let (c, scenes) = fixture();
        let mut envs = seeded_environments(scenes, &c, &SimulationConfig::default(), 9).unwrap();
        let e = &mut envs[0];
        e.state.speed = 0.;
        let initial = e.state;
        let standing = e.step([0., 0.]).unwrap();
        assert_eq!(standing.reward.total(), 0.);
        assert_eq!(initial.position, e.state.position);
        e.state.speed = -45.;
        for _ in 0..3 {
            let reverse = e.step([0., -0.5]).unwrap();
            assert_eq!(reverse.reward.progress, 0.);
            assert!(reverse.observation[21] < 0.);
            assert_eq!(&reverse.observation[23..], [0., -0.5]);
        }
    }
}
