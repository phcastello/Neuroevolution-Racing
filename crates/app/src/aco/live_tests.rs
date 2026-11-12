use super::*;
use std::time::{Duration, Instant};
fn config() -> AcoConfig {
    AcoConfig {
        generator: GeneratorConfig::legacy(),
        ants: 2,
        iterations: 2,
        max_candidates: 4,
        ..Default::default()
    }
}
fn wait(
    live: &LiveSearch,
    events: &mut Vec<SearchEvent>,
    condition: impl Fn(&[SearchEvent]) -> bool,
) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        events.extend(live.poll());
        if condition(events) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "worker timeout after {} events",
            events.len()
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}
fn signatures(a: &SearchArchive) -> Vec<Vec<usize>> {
    a.iterations
        .iter()
        .flat_map(|i| {
            i.ants
                .iter()
                .map(|a| a.decisions.iter().map(|d| d.choice).collect())
        })
        .collect()
}
#[test]
fn decisions_pause_step_geometry_pheromones_restart_and_headless_parity() {
    let live = LiveSearch::start(config(), 47, None, RunMode::Paused, 120.).unwrap();
    let mut events = vec![];
    wait(&live, &mut events, |_| live.control.is_paused());
    assert!(events.is_empty());
    for expected in 1..=3 {
        live.control.set_mode(RunMode::Decision);
        wait(&live, &mut events, |e| {
            live.control.is_paused()
                && e.iter()
                    .filter(|e| matches!(e, SearchEvent::Decision(_)))
                    .count()
                    == expected
        });
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, SearchEvent::Geometry { .. } | SearchEvent::Candidate(_)))
        );
        let count = events.len();
        std::thread::sleep(Duration::from_millis(25));
        events.extend(live.poll());
        assert_eq!(events.len(), count, "pause must stop actual computation");
    }
    live.control.set_mode(RunMode::Iteration);
    wait(&live, &mut events, |e| {
        live.control.is_paused() && e.iter().any(|e| matches!(e, SearchEvent::Iteration { .. }))
    });
    let first = events
        .iter()
        .find_map(|e| {
            if let SearchEvent::Iteration { archive } = e {
                Some(archive)
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(first.state.iteration, 1);
    assert_ne!(
        first.iterations[0].pheromones_before,
        first.state.pheromones
    );
    assert!(
        first.iterations[0]
            .pheromones_after
            .iter()
            .flatten()
            .zip(first.iterations[0].pheromones_evaporated.iter().flatten())
            .any(|(a, b)| a > b)
    );
    let mut chosen = vec![];
    let mut geometries = 0;
    for e in &events {
        match e {
            SearchEvent::AntStarted { .. } => chosen.clear(),
            SearchEvent::Decision(d) => chosen.push(d.choice),
            SearchEvent::Geometry { definition, .. } => {
                assert!(config().generator.decisions_valid(&chosen));
                let rebuilt =
                    procedural::definition_from_decisions(&config().generator, &chosen).unwrap();
                assert_eq!(definition, &rebuilt);
                geometries += 1;
            }
            _ => (),
        }
    }
    assert_eq!(geometries, 2);
    live.control.set_mode(RunMode::Running);
    wait(&live, &mut events, |e| {
        e.iter().any(|e| matches!(e, SearchEvent::Finished { .. }))
    });
    let actual = events
        .iter()
        .find_map(|e| {
            if let SearchEvent::Finished { archive } = e {
                Some(archive)
            } else {
                None
            }
        })
        .unwrap();
    let mut headless = SearchArchive::new(config(), 47).unwrap();
    while !headless.is_finished() {
        headless.run_iteration(None).unwrap();
    }
    assert_eq!(signatures(actual), signatures(&headless));
    assert_eq!(actual.state.pheromones, headless.state.pheromones);
    assert_eq!(
        actual.state.selected[0].track,
        headless.state.selected[0].track
    );
    let restarted = LiveSearch::start(config(), 47, None, RunMode::Iteration, 120.).unwrap();
    let mut repeated = vec![];
    wait(&restarted, &mut repeated, |_| restarted.control.is_paused());
    restarted.control.set_mode(RunMode::Iteration);
    wait(&restarted, &mut repeated, |e| {
        e.iter().any(|e| matches!(e, SearchEvent::Finished { .. }))
    });
    let repeat = repeated
        .iter()
        .find_map(|e| {
            if let SearchEvent::Finished { archive } = e {
                Some(archive)
            } else {
                None
            }
        })
        .unwrap();
    assert_eq!(signatures(actual), signatures(repeat));
    assert_eq!(actual.state.pheromones, repeat.state.pheromones);
    println!(
        "real worker: 52 decisions, 4 candidates, 2 iterations; pause/step/resume/restart == headless"
    );
}
#[test]
fn cancel_wakes_paused_worker_and_drop_never_joins() {
    let live = LiveSearch::start(config(), 47, None, RunMode::Paused, 1.).unwrap();
    let mut events = vec![];
    wait(&live, &mut events, |_| live.control.is_paused());
    live.control.cancel();
    wait(&live, &mut events, |e| {
        e.iter()
            .any(|e| matches!(e,SearchEvent::Failed(s) if s.contains("cancelled")))
    });
    assert!(!events.iter().any(|e| matches!(e, SearchEvent::Decision(_))));
    drop(live);
}
#[test]
fn invalid_checkpoint_reports_error_without_fabricating_candidates() {
    let live = LiveSearch::start(
        config(),
        47,
        Some("missing-aco-checkpoint.ron".into()),
        RunMode::Running,
        120.,
    )
    .unwrap();
    let mut events = vec![];
    wait(&live, &mut events, |e| {
        e.iter().any(|e| matches!(e, SearchEvent::Failed(_)))
    });
    assert_eq!(events.len(), 1);
}
