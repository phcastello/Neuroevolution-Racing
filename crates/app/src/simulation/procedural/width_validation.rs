use super::*;
use crate::{
    aco::{AcoConfig, PilotEvaluation, SearchArchive},
    simulation::{
        CAR_WIDTH, SimulationConfig, TrackLibrary, WALL_THICKNESS,
        rl_environment::{RacingEnvironment, Scene},
    },
};
use racing_rl::{
    config::Config,
    rollout::{EndReason, Environment},
};
use std::rc::Rc;

fn drive(t: &GeneratedTrack, limit: u32) -> (bool, bool, f32, u64) {
    let mut env = RacingEnvironment::new(
        0,
        vec![Rc::new(Scene::new(t.validate().unwrap()))],
        &Config {
            num_envs: 1,
            timeout_ticks: limit as u64,
            ..Default::default()
        },
        &SimulationConfig::default(),
        ChaCha12Rng::seed_from_u64(55),
    )
    .unwrap();
    let mut traveled = 0.;
    for tick in 1..=limit {
        let before = env.state.position;
        let step = env.step(env.reference_action()).unwrap();
        traveled += before.distance(env.state.position);
        if step.terminated || step.truncated {
            let ok = step.reason == Some(EndReason::Completed);
            if ok {
                assert!(traveled > t.geometry.length * 0.9);
            }
            return (
                ok,
                step.reason == Some(EndReason::Collision),
                env.progress_fraction(),
                tick as u64,
            );
        }
    }
    (false, false, env.progress_fraction(), limit as u64)
}
fn wide_hairpin() -> GeneratedTrack {
    let c = GeneratorConfig {
        widths: vec![100.],
        ..Default::default()
    };
    let mut state = sequential::Construction::new(&c);
    let mut choices = vec![];
    for choice in [2, 7] {
        let mut options = state.options(&c, &mut || Ok(())).unwrap();
        state = options[choice].next.take().unwrap();
        choices.push(choice);
    }
    while !state.closed() {
        let mut options = state.options(&c, &mut || Ok(())).unwrap();
        state = options[sequential::CLOSE].next.take().unwrap();
        choices.push(sequential::CLOSE);
    }
    choices.push(0);
    from_decisions(&c, 55, 0, &choices).unwrap()
}
#[test]
fn wide_hairpin_and_interlagos_share_physical_width_and_wall_geometry() {
    let generated = wide_hairpin();
    let t = generated.validate().unwrap();
    assert!((1. / generated.geometry.max_curvature - 75.).abs() < 0.2);
    let scene = Scene::new(t);
    let (p, f) = scene.track.pose_at_distance(300.);
    let h = f.to_angle();
    let normal = f.perp();
    let left = scene.ray(p, h + std::f32::consts::FRAC_PI_2, 200.);
    let right = scene.ray(p, h - std::f32::consts::FRAC_PI_2, 200.);
    assert!((left + right - (100. - WALL_THICKNESS)).abs() < 0.2);
    for i in -2..=2 {
        assert!(
            !scene.intersects(p + normal * i as f32 * 16., h),
            "five cars must fit between physical walls"
        );
    }
    assert!(scene.intersects(p + normal * 40., h));
    for (i, sample) in scene.track.samples.iter().enumerate().step_by(20) {
        assert!(
            (scene.track.left_border[i].distance(scene.track.right_border[i]) - 100.).abs() < 0.001
        );
        let projection = scene.track.project_near(sample.position, i, 24);
        assert!(projection.point.distance(sample.position) < 0.01);
    }
    let interlagos = Track::from_definition(
        TrackLibrary::load_default()
            .unwrap()
            .definition("interlagos")
            .unwrap(),
    )
    .unwrap();
    let other = Scene::new(interlagos);
    let (p, f) = other.track.pose_at_distance(300.);
    let h = f.to_angle();
    let measured = other.ray(p, h + std::f32::consts::FRAC_PI_2, 200.)
        + other.ray(p, h - std::f32::consts::FRAC_PI_2, 200.);
    assert!((measured - 86.).abs() < 1.);
    assert_eq!(other.track.width, scene.track.width);
    assert!(drive(&generated, 10800).0);
    let incompatible = GeneratorConfig {
        primitive_radius: 50.,
        widths: vec![120.],
        ..Default::default()
    };
    assert!(
        incompatible
            .validate()
            .unwrap_err()
            .contains("width was not reduced")
    );
}

#[test]
#[ignore = "fixed-seed physical width and rejection artifacts; run explicitly"]
fn width_and_rejection_evidence_battery() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/validation/aco_width_inspection");
    assert!(!root.exists());
    fs::create_dir_all(&root).unwrap();
    let library = TrackLibrary::load_default().unwrap();
    let interlagos = Track::from_definition(library.definition("interlagos").unwrap()).unwrap();
    let dump = |track: &Track| serde_json::json!({"definition":track.definition,"centerline":track.samples.iter().map(|s|s.position.to_array()).collect::<Vec<_>>(),"left":track.left_border.iter().map(|p|p.to_array()).collect::<Vec<_>>(),"right":track.right_border.iter().map(|p|p.to_array()).collect::<Vec<_>>()});
    fs::write(
        root.join("interlagos.json"),
        serde_json::to_vec(&dump(&interlagos)).unwrap(),
    )
    .unwrap();
    let mut rows = vec![];
    for wide in [false, true] {
        for seed in [17, 47] {
            let variant = if wide { "wide" } else { "previous" };
            let dir = root.join(format!("{variant}-{seed}"));
            fs::create_dir_all(&dir).unwrap();
            let mut search = SearchArchive::new(
                AcoConfig {
                    generator: GeneratorConfig {
                        widths: if wide {
                            vec![96., 100., 104.]
                        } else {
                            vec![44., 52., 60.]
                        },
                        primitive_radius: if wide { 75. } else { 50. },
                        ..Default::default()
                    },
                    ants: 3,
                    iterations: 2,
                    max_candidates: 6,
                    max_selected: 6,
                    ..Default::default()
                },
                seed,
            )
            .unwrap();
            while !search.is_finished() {
                search.run_iteration(None).unwrap();
            }
            for it in &search.iterations {
                for ant in &it.ants {
                    if let Some(t) = &ant.candidate {
                        let track = t.validate().unwrap();
                        let (ok, collision, progress, ticks) = drive(t, 10800);
                        let scene = Scene::new(track.clone());
                        let mut clear = vec![];
                        for i in (0..scene.track.samples.len()).step_by(30) {
                            let a = scene.track.samples[i].position;
                            let b =
                                scene.track.samples[(i + 1) % scene.track.samples.len()].position;
                            let position = (a + b) * 0.5;
                            let h = (b - a).to_angle();
                            clear.push(
                                scene.ray(position, h + std::f32::consts::FRAC_PI_2, 200.)
                                    + scene.ray(position, h - std::f32::consts::FRAC_PI_2, 200.),
                            );
                        }
                        let row = serde_json::json!({"variant":variant,"seed":seed,"iteration":it.iteration,"ant":ant.ant,"id":t.definition.id,"width":track.width,"nominal_cars":track.width/CAR_WIDTH,"useful_width_mean":clear.iter().sum::<f32>()/clear.len() as f32,"useful_width_min":clear.iter().copied().fold(f32::INFINITY,f32::min),"physical_cars":(track.width-WALL_THICKNESS)/CAR_WIDTH,"metrics":t.geometry,"accepted":ant.rejection.is_none(),"reason":ant.rejection_reason,"reference_completed":ok,"reference_collision":collision,"reference_progress":progress,"reference_ticks":ticks});
                        fs::write(
                            dir.join(format!("{}.json", t.definition.id)),
                            serde_json::to_vec(
                                &serde_json::json!({"track":dump(&track),"result":row}),
                            )
                            .unwrap(),
                        )
                        .unwrap();
                        println!(
                            "WIDTH {variant} seed={seed} {} width={} physical={} Rmin={:.1} reference={ok}",
                            t.definition.id,
                            track.width,
                            track.width - WALL_THICKNESS,
                            1. / t.geometry.max_curvature
                        );
                        rows.push(row);
                    } else {
                        rows.push(serde_json::json!({"variant":variant,"seed":seed,"id":null,"reason":ant.rejection_reason}));
                    }
                }
            }
            search.save(dir.join("search.ron")).unwrap();
            let loaded = SearchArchive::load(dir.join("search.ron")).unwrap();
            assert_eq!(loaded.state.pheromones, search.state.pheromones);
        }
    }
    let mut rejected = SearchArchive::new(
        AcoConfig {
            ants: 1,
            iterations: 1,
            max_candidates: 1,
            policy_label: Some("reference-only real 120-decision diagnostic".into()),
            ..Default::default()
        },
        17,
    )
    .unwrap();
    let mut evaluator = |t: &GeneratedTrack, _seed: u64, _budget: usize| {
        let (ok, collision, progress, ticks) = drive(t, 120);
        Ok(PilotEvaluation {
            reference_completed: ok,
            reference_collision: collision,
            reference_progress: progress as f64,
            reference_episodes: 1,
            reference_steps: ticks,
            simulation_steps: ticks,
            ..Default::default()
        })
    };
    rejected.run_iteration(Some(&mut evaluator)).unwrap();
    assert!(rejected.iterations[0].ants[0].geometrically_valid());
    assert!(rejected.iterations[0].ants[0].rejection.is_some());
    assert!(
        drive(
            rejected.iterations[0].ants[0].candidate.as_ref().unwrap(),
            10800
        )
        .0
    );
    rejected.save(root.join("reference-rejected.ron")).unwrap();
    rejected
        .export_candidate(0, 0, root.join("reference-rejected-valid"))
        .unwrap();
    let mut invalid = SearchArchive::new(
        AcoConfig {
            ants: 1,
            iterations: 1,
            max_candidates: 1,
            generator: GeneratorConfig {
                min_length: 1.,
                max_length: 2.,
                ..Default::default()
            },
            ..Default::default()
        },
        17,
    )
    .unwrap();
    invalid.run_iteration(None).unwrap();
    invalid.save(root.join("geometry-invalid.ron")).unwrap();
    invalid
        .export_diagnostic(0, 0, root.join("geometry-invalid"))
        .unwrap();
    assert!(
        invalid
            .export_candidate(0, 0, root.join("blocked-training-export"))
            .is_err()
    );
    let hairpin = wide_hairpin();
    fs::write(
        root.join("hairpin.json"),
        serde_json::to_vec(&dump(&hairpin.validate().unwrap())).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("results.json"),
        serde_json::to_vec_pretty(&rows).unwrap(),
    )
    .unwrap();
    for variant in ["wide", "previous"] {
        let valid: Vec<_> = rows
            .iter()
            .filter(|r| r["variant"] == variant && !r["id"].is_null())
            .collect();
        assert!(
            valid.len() >= 11,
            "valid generation rate fell: {variant} {}",
            valid.len()
        );
        assert!(valid.iter().all(|r| r["reference_completed"] == true));
    }
}

#[test]
#[ignore = "adds real corridor-rejection and reference-retry diagnostics after the width battery"]
fn inspection_rejection_cases_evidence() {
    use crate::aco::{
        AntSnapshot, DecisionSnapshot, RejectionReason, inspection::GeometryDiagnostic,
    };
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/validation/aco_width_inspection");
    let rejected = SearchArchive::load(root.join("reference-rejected.ron")).unwrap();
    let retry = drive(
        rejected.iterations[0].ants[0].candidate.as_ref().unwrap(),
        10800,
    );
    assert!(retry.0);
    fs::write(root.join("reference-retry.json"),serde_json::to_vec_pretty(&serde_json::json!({"short_evaluation":rejected.iterations[0].ants[0].evaluation,"long_evaluation":{"completed":retry.0,"collision":retry.1,"progress":retry.2,"ticks":retry.3}})).unwrap()).unwrap();
    let c = GeneratorConfig::default();
    let mut state = sequential::Construction::new(&c);
    let mut decisions = vec![];
    for (position, choice) in [2, 7, 2].into_iter().enumerate() {
        let mut options = state.options(&c, &mut || Ok(())).unwrap();
        let views = options.iter().map(|a| a.view.clone()).collect();
        let start = state.points().len();
        state = options[choice].next.take().unwrap();
        let mut probabilities = vec![0.; sequential::CHOICES];
        probabilities[choice] = 1.;
        decisions.push(DecisionSnapshot {
            position,
            choice,
            probabilities,
            construction: Some(state.view(start, views)),
        });
    }
    let options = state.options(&c, &mut || Ok(())).unwrap();
    let failed = options
        .iter()
        .find(|a| {
            a.view
                .issue
                .as_ref()
                .is_some_and(|i| i.message.contains("Corredores"))
        })
        .expect("fixture must produce a real corridor rejection");
    let reason = failed.view.issue.as_ref().unwrap().message.clone();
    let terminal = state.view(
        state.points().len(),
        options.iter().map(|a| a.view.clone()).collect(),
    );
    let record = AntSnapshot {
        ant: 0,
        diagnostic: Some(GeometryDiagnostic::capture(
            None,
            &decisions,
            Some(terminal),
            &reason,
        )),
        decisions,
        candidate: None,
        rejection: Some(format!(
            "Diagnóstico controlado de alternativa rejeitada: {reason}"
        )),
        fitness: None,
        rejection_reason: Some(RejectionReason::CorridorClearance),
        evaluation: None,
        geometry_valid: Some(false),
        retention_evicted: false,
        candidate_id: None,
    };
    let mut archive = SearchArchive::load(root.join("geometry-invalid.ron")).unwrap();
    archive.config.generator = c;
    archive.iterations[0].ants = vec![record];
    archive.save(root.join("corridor-rejected.ron")).unwrap();
    archive
        .export_diagnostic(0, 0, root.join("corridor-rejected"))
        .unwrap();
}

#[test]
#[ignore = "refresh physical midpoint ray measurements from the already saved immutable track definitions"]
fn remeasure_width_evidence_midsegments() {
    let root =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/validation/aco_width_inspection");
    let mut rows: Vec<serde_json::Value> =
        serde_json::from_slice(&fs::read(root.join("results.json")).unwrap()).unwrap();
    for r in &mut rows {
        if r["id"].is_null() {
            continue;
        }
        let file = root.join(format!(
            "{}-{}/{}.json",
            r["variant"].as_str().unwrap(),
            r["seed"],
            r["id"].as_str().unwrap()
        ));
        let mut source: serde_json::Value =
            serde_json::from_slice(&fs::read(&file).unwrap()).unwrap();
        let d: TrackDefinition =
            serde_json::from_value(source["track"]["definition"].clone()).unwrap();
        let scene = Scene::new(Track::from_definition(&d).unwrap());
        let mut clear = vec![];
        for i in (0..scene.track.samples.len()).step_by(30) {
            let a = scene.track.samples[i].position;
            let b = scene.track.samples[(i + 1) % scene.track.samples.len()].position;
            let p = (a + b) * 0.5;
            let h = (b - a).to_angle();
            clear.push(
                scene.ray(p, h + std::f32::consts::FRAC_PI_2, 200.)
                    + scene.ray(p, h - std::f32::consts::FRAC_PI_2, 200.),
            );
        }
        r["useful_width_mean"] = serde_json::json!(clear.iter().sum::<f32>() / clear.len() as f32);
        r["useful_width_min"] =
            serde_json::json!(clear.iter().copied().fold(f32::INFINITY, f32::min));
        assert!(
            (r["useful_width_mean"].as_f64().unwrap() - (d.width - WALL_THICKNESS) as f64).abs()
                < 0.6
        );
        source["result"] = r.clone();
        fs::write(file, serde_json::to_vec(&source).unwrap()).unwrap();
    }
    fs::write(
        root.join("results.json"),
        serde_json::to_vec_pretty(&rows).unwrap(),
    )
    .unwrap();
}
