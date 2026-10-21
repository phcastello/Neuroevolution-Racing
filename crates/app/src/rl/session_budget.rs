//! Session control only. Bounds never change PPO updates or schedules.
use std::time::Instant;

pub(super) const ENVIRONMENT_SEMANTICS: &str = "rl-progress-connected-adjacent-v2";
pub(super) trait Clock {
    fn seconds(&self) -> f64;
}
pub(super) struct Monotonic(pub Instant);
impl Clock for Monotonic {
    fn seconds(&self) -> f64 {
        self.0.elapsed().as_secs_f64()
    }
}
pub(super) struct Budget<C: Clock> {
    pub clock: C,
    pub limit: Option<f64>,
    pub rollout_bound: f64,
    pub evaluation_bound: f64,
    pub persistence_bound: f64,
}
impl<C: Clock> Budget<C> {
    pub fn new(clock: C, limit: Option<f64>) -> Self {
        Self {
            clock,
            limit,
            rollout_bound: 60.,
            evaluation_bound: 300.,
            persistence_bound: 60.,
        }
    }
    pub fn remaining(&self) -> f64 {
        self.limit.map_or(f64::INFINITY, |limit| {
            (limit - self.clock.seconds()).max(0.)
        })
    }
    pub fn can_start(&self, seconds: f64) -> bool {
        self.remaining() > seconds + self.persistence_bound
    }
    pub fn can_rollout(&self, evaluate: bool) -> bool {
        self.can_start(self.rollout_bound + if evaluate { self.evaluation_bound } else { 0. })
    }
    pub fn observe_rollout(&mut self, seconds: f64) {
        self.rollout_bound = self.rollout_bound.max(3. * seconds + 30.);
    }
    pub fn observe_evaluation(&mut self, seconds: f64) {
        self.evaluation_bound = self.evaluation_bound.max(3. * seconds + 30.);
    }
    pub fn observe_persistence(&mut self, seconds: f64) {
        self.persistence_bound = self.persistence_bound.max(3. * seconds + 30.);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    struct Fake(Cell<f64>);
    impl Clock for Fake {
        fn seconds(&self) -> f64 {
            self.0.get()
        }
    }
    #[test]
    fn initialization_and_all_work_consume_one_monotonic_budget() {
        let mut b = Budget::new(Fake(Cell::new(0.)), Some(1000.));
        assert!(b.can_rollout(true));
        b.clock.0.set(700.); // compilation/initialization already spent by session
        assert!(!b.can_rollout(true));
        assert!(b.can_rollout(false));
        b.observe_rollout(50.);
        assert_eq!(b.rollout_bound, 180.);
        b.clock.0.set(950.);
        assert!(!b.can_rollout(false));
        assert!(!b.can_start(b.evaluation_bound));
        b.clock.0.set(1001.);
        assert_eq!(b.remaining(), 0.);
        assert!(!b.can_rollout(false));
        assert!(!Budget::new(Fake(Cell::new(0.)), Some(0.)).can_rollout(false));
    }
}
