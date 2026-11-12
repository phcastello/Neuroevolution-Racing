//! Cooperative execution controls for the same engine used by the CLI/curriculum.
use super::*;
use crate::simulation::TrackDefinition;
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::time::Duration;

#[derive(Clone, Debug)]
pub enum SearchEvent {
    AntStarted {
        iteration: usize,
        ant: usize,
        pheromones: Vec<Vec<f64>>,
    },
    Decision(DecisionSnapshot),
    Geometry {
        definition: TrackDefinition,
        error: Option<String>,
    },
    Candidate(AntSnapshot),
    Iteration {
        archive: Box<SearchArchive>,
    },
    Finished {
        archive: Box<SearchArchive>,
    },
    Failed(String),
}
pub trait SearchObserver {
    fn is_observing(&self) -> bool {
        true
    }
    fn checkpoint(&mut self) -> GenerationResult<()> {
        Ok(())
    }
    fn event(&mut self, event: SearchEvent) -> GenerationResult<()>;
}
pub struct Unobserved;
impl SearchObserver for Unobserved {
    fn is_observing(&self) -> bool {
        false
    }
    fn event(&mut self, _: SearchEvent) -> GenerationResult<()> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunMode {
    Running,
    Paused,
    Decision,
    Iteration,
}
struct ControlState {
    mode: RunMode,
    cancelled: bool,
    waiting: bool,
    delay: Duration,
}
#[derive(Clone)]
pub struct SearchControl(Arc<(Mutex<ControlState>, Condvar)>);
impl SearchControl {
    pub fn new(mode: RunMode, decisions_per_second: f32) -> Self {
        Self(Arc::new((
            Mutex::new(ControlState {
                mode,
                cancelled: false,
                waiting: false,
                delay: Duration::from_secs_f32(1. / decisions_per_second.clamp(0.5, 120.)),
            }),
            Condvar::new(),
        )))
    }
    pub fn set_mode(&self, mode: RunMode) {
        let mut state = self.0.0.lock().unwrap();
        state.mode = mode;
        self.0.1.notify_all();
    }
    pub fn set_speed(&self, speed: f32) {
        self.0.0.lock().unwrap().delay = Duration::from_secs_f32(1. / speed.clamp(0.5, 120.));
        self.0.1.notify_all();
    }
    pub fn cancel(&self) {
        self.0.0.lock().unwrap().cancelled = true;
        self.0.1.notify_all();
    }
    pub fn is_paused(&self) -> bool {
        let state = self.0.0.lock().unwrap();
        state.mode == RunMode::Paused && state.waiting
    }
    pub fn mode(&self) -> RunMode {
        self.0.0.lock().unwrap().mode
    }
    /// Also called between every physical/reference/PPO decision. No animation
    /// clock controls simulation time and cancellation wakes every wait.
    pub fn checkpoint(&self) -> GenerationResult<()> {
        let mut state = self.0.0.lock().unwrap();
        while state.mode == RunMode::Paused && !state.cancelled {
            state.waiting = true;
            state = self.0.1.wait(state).unwrap();
        }
        state.waiting = false;
        if state.cancelled {
            Err("search cancelled".into())
        } else {
            Ok(())
        }
    }
    fn after_event(&self, event: &SearchEvent) -> GenerationResult<()> {
        if matches!(event, SearchEvent::Iteration { archive } if archive.is_finished()) {
            return Ok(());
        }
        let mut state = self.0.0.lock().unwrap();
        if (state.mode == RunMode::Decision && matches!(event, SearchEvent::Decision(_)))
            || (state.mode == RunMode::Iteration && matches!(event, SearchEvent::Iteration { .. }))
        {
            state.mode = RunMode::Paused;
        }
        if state.mode == RunMode::Running
            && matches!(
                event,
                SearchEvent::Decision(_)
                    | SearchEvent::Geometry { .. }
                    | SearchEvent::Iteration { .. }
            )
        {
            let delay = state.delay;
            // UI changes wake this wait immediately (pause, speed, stop).
            state = self.0.1.wait_timeout(state, delay).unwrap().0;
        }
        drop(state);
        self.checkpoint()
    }
}

struct Publisher {
    control: SearchControl,
    sender: mpsc::SyncSender<SearchEvent>,
}
impl SearchObserver for Publisher {
    fn checkpoint(&mut self) -> GenerationResult<()> {
        self.control.checkpoint()
    }
    fn event(&mut self, event: SearchEvent) -> GenerationResult<()> {
        // Bounded mailbox: a hidden/slow UI cannot accumulate unbounded snapshots.
        let mut pending = event.clone();
        loop {
            match self.sender.try_send(pending) {
                Ok(()) => break,
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    return Err("viewer disconnected".into());
                }
                Err(mpsc::TrySendError::Full(event)) => {
                    pending = event;
                    self.control.checkpoint()?;
                    std::thread::sleep(Duration::from_millis(2));
                }
            }
        }
        self.control.after_event(&event)
    }
}

/// Only serialized data crosses the thread boundary. PPO networks and their Rc
/// environments are loaded and owned entirely inside the worker.
pub struct LiveSearch {
    pub control: SearchControl,
    receiver: Mutex<mpsc::Receiver<SearchEvent>>,
    worker: Option<std::thread::JoinHandle<()>>,
}
impl LiveSearch {
    pub fn start(
        mut config: AcoConfig,
        seed: u64,
        checkpoint: Option<std::path::PathBuf>,
        mode: RunMode,
        speed: f32,
    ) -> GenerationResult<Self> {
        config.validate()?;
        let control = SearchControl::new(mode, speed);
        let worker_control = control.clone();
        let (sender, receiver) = mpsc::sync_channel(8);
        let worker = std::thread::Builder::new()
            .name("aco-search".into())
            .spawn(move || {
                let result = (|| {
                    worker_control.checkpoint()?;
                    config.policy_label = checkpoint
                        .as_ref()
                        .map(|p| super::cli::checkpoint_identity(p))
                        .transpose()?;
                    let mut evaluator = checkpoint
                        .as_ref()
                        .map(|p| {
                            crate::rl::curriculum::checkpoint_evaluator_controlled(
                                p,
                                worker_control.clone(),
                            )
                        })
                        .transpose()?;
                    let mut archive = SearchArchive::new(config, seed)?;
                    let mut observer = Publisher {
                        control: worker_control.clone(),
                        sender: sender.clone(),
                    };
                    while !archive.is_finished() {
                        worker_control.checkpoint()?;
                        archive.run_iteration_observed(
                            evaluator.as_mut().map(|e| e as &mut dyn CandidateEvaluator),
                            &mut observer,
                        )?;
                    }
                    Ok::<_, String>(archive)
                })();
                // Finish/error is terminal and must not wait for a paused observer.
                let event = match result {
                    Ok(archive) => SearchEvent::Finished {
                        archive: Box::new(archive),
                    },
                    Err(error) => SearchEvent::Failed(error),
                };
                let _ = sender.send(event);
            })
            .map_err(|e| e.to_string())?;
        Ok(Self {
            control,
            receiver: Mutex::new(receiver),
            worker: Some(worker),
        })
    }
    pub fn poll(&self) -> Vec<SearchEvent> {
        self.receiver.lock().unwrap().try_iter().collect()
    }
    pub fn is_finished(&self) -> bool {
        self.worker.as_ref().is_none_or(|w| w.is_finished())
    }
}
impl Drop for LiveSearch {
    fn drop(&mut self) {
        self.control.cancel();
        // Never join on Bevy's render thread. Dropping the receiver releases any
        // terminal send; the cooperative worker then exits without touching UI.
    }
}
#[cfg(test)]
#[path = "live_tests.rs"]
mod tests;
