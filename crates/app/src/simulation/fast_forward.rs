use std::time::{Duration, Instant};

use bevy::prelude::Resource;

/// Maximum wall-clock time spent running exact 1/60 s ticks before yielding to
/// window events and the minimal turbo UI. This naturally limits rendering to
/// roughly five updates per second while keeping the CPU simulation saturated.
pub const FAST_FORWARD_BATCH_BUDGET: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlaybackRestore {
    pub speed: f32,
    pub max_delta: Duration,
}

#[derive(Resource, Clone, Debug)]
pub struct TrainingFastForward {
    pub target_generation: Option<usize>,
    pub target_input: String,
    pub status: String,
    previous_playback: Option<PlaybackRestore>,
    started_at: Option<Instant>,
    started_generation: usize,
    logical_ticks: u64,
}

impl Default for TrainingFastForward {
    fn default() -> Self {
        Self {
            target_generation: None,
            target_input: String::new(),
            status: String::new(),
            previous_playback: None,
            started_at: None,
            started_generation: 0,
            logical_ticks: 0,
        }
    }
}

impl TrainingFastForward {
    pub fn is_active(&self) -> bool {
        self.target_generation.is_some()
    }

    pub fn start(
        &mut self,
        target_generation: usize,
        current_generation: usize,
        current_speed: f32,
        current_max_delta: Duration,
    ) -> bool {
        if target_generation <= current_generation {
            self.status =
                format!("Target must be greater than current generation ({current_generation})");
            return false;
        }
        self.target_generation = Some(target_generation);
        self.previous_playback = Some(PlaybackRestore {
            speed: current_speed,
            max_delta: current_max_delta,
        });
        self.started_at = Some(Instant::now());
        self.started_generation = current_generation;
        self.logical_ticks = 0;
        self.status = format!("Fast-forwarding to generation {target_generation}");
        true
    }

    pub fn elapsed(&self) -> Duration {
        self.started_at
            .map_or(Duration::ZERO, |start| start.elapsed())
    }

    pub fn generations_per_second(&self, current_generation: usize) -> f64 {
        let elapsed = self.elapsed().as_secs_f64();
        if elapsed <= f64::EPSILON {
            0.0
        } else {
            current_generation.saturating_sub(self.started_generation) as f64 / elapsed
        }
    }

    pub fn fixed_ticks_per_second(&self) -> f64 {
        let elapsed = self.elapsed().as_secs_f64();
        if elapsed <= f64::EPSILON {
            0.0
        } else {
            self.logical_ticks as f64 / elapsed
        }
    }

    pub(crate) fn record_fixed_tick(&mut self) {
        self.logical_ticks = self.logical_ticks.saturating_add(1);
    }

    pub fn finish_if_reached(&mut self, current_generation: usize) -> Option<PlaybackRestore> {
        let target = self.target_generation?;
        if current_generation < target {
            return None;
        }
        let rate = self.generations_per_second(current_generation);
        let tick_rate = self.fixed_ticks_per_second();
        self.status = format!(
            "Reached generation {target} at {rate:.2} gen/s ({tick_rate:.0} fixed ticks/s); training continues at previous playback speed"
        );
        self.target_generation = None;
        self.started_at = None;
        self.previous_playback.take()
    }

    pub fn cancel(&mut self) -> Option<PlaybackRestore> {
        let target = self.target_generation.take()?;
        self.status = format!("Fast-forward to generation {target} cancelled");
        self.started_at = None;
        self.previous_playback.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NORMAL_MAX_DELTA: Duration = Duration::from_millis(250);

    #[test]
    fn only_a_future_target_activates_fast_forward() {
        for target in [49, 50] {
            let mut state = TrainingFastForward::default();
            assert!(!state.start(target, 50, 2.0, NORMAL_MAX_DELTA));
            assert!(!state.is_active());
        }
        let mut state = TrainingFastForward::default();
        assert!(state.start(51, 50, 2.0, NORMAL_MAX_DELTA));
        assert!(state.is_active());
    }

    #[test]
    fn fast_forward_stays_active_before_target_and_finishes_at_or_after_it() {
        let mut state = TrainingFastForward::default();
        state.start(200, 50, 2.0, NORMAL_MAX_DELTA);
        assert_eq!(state.finish_if_reached(199), None);
        assert!(state.is_active());
        assert_eq!(
            state.finish_if_reached(200),
            Some(PlaybackRestore {
                speed: 2.0,
                max_delta: NORMAL_MAX_DELTA,
            })
        );
        assert!(!state.is_active());
    }

    #[test]
    fn cancel_and_completion_restore_the_previous_user_speed() {
        let mut cancelled = TrainingFastForward::default();
        cancelled.start(100, 10, 10.0, NORMAL_MAX_DELTA);
        assert_eq!(cancelled.cancel().unwrap().speed, 10.0);

        let mut completed = TrainingFastForward::default();
        completed.start(100, 10, 25.0, NORMAL_MAX_DELTA);
        assert_eq!(completed.finish_if_reached(100).unwrap().speed, 25.0);
    }
}
