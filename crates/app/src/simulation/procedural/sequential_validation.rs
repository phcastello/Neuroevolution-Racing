use super::*;
use crate::{
    aco::{AcoConfig, SearchArchive},
    simulation::{
        SimulationConfig, TrackLibrary,
        rl_environment::{RacingEnvironment, Scene},
    },
};
use racing_rl::{
    config::Config,
    rollout::{EndReason, Environment},
};
use std::rc::Rc;
fn drive(t: &GeneratedTrack) -> (bool, f32, u64) {
    let scene = Rc::new(Scene::new(t.validate().unwrap()));
    let c = Config {
        num_envs: 1,
        timeout_ticks: 10800,
        ..Default::default()
    };
    let mut env = RacingEnvironment::new(
        0,
        vec![scene],
        &c,
        &SimulationConfig::default(),
        ChaCha12Rng::seed_from_u64(55),
    )
    .unwrap();
    let mut traveled = 0.;
    for tick in 1..=10800 {
        let previous = env.state.position;
        let step = env.step(env.reference_action()).unwrap();
        traveled += previous.distance(env.state.position);
        if step.terminated || step.truncated {
            if step.reason == Some(EndReason::Completed) {
                assert!(
                    traveled > t.geometry.length * 0.9,
                    "a nearby branch must not shortcut lap progress"
                );
            }
            return (
                step.reason == Some(EndReason::Completed),
                env.progress_fraction(),
                tick,
            );
        }
    }
    (false, env.progress_fraction(), 10800)
}
#[test]
fn spa_la_source_calibration_uses_existing_sampled_geometry() {
    let library = TrackLibrary::load_default().unwrap();
    let spa = Track::from_definition(library.definition("spa").unwrap()).unwrap();
    let mut k = 0_f32;
    let mut index = 0;
    for i in 3 * 12..11 * 12 {
        let a = spa.samples[i - 1].position;
        let b = spa.samples[i].position;
        let c = spa.samples[i + 1].position;
        let value = (2. * (b - a).perp_dot(c - b)
            / ((b - a).length() * (c - b).length() * c.distance(a)))
        .abs();
        if value > k {
            k = value;
            index = i;
        }
    }
    println!(
        "LA_SOURCE sample={index} curvature={k} min_radius={} width={} generated_radius={}",
        1. / k,
        spa.width,
        GeneratorConfig::default().primitive_radius
    );
    assert!((k - 0.02094).abs() < 0.00005);
    assert!((1. / k - 47.75).abs() < 0.2);
    assert!(1. / GeneratorConfig::default().primitive_radius <= k * 1.01);
}
#[test]
fn small_real_aco_geometry_and_reference_battery() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/sequential-aco-evidence")
        .join(format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    fs::create_dir_all(&root).unwrap();
    let mut search = SearchArchive::new(
        AcoConfig {
            generator: GeneratorConfig {
                widths: vec![44., 52., 60.],
                primitive_radius: 50.,
                ..Default::default()
            },
            ants: 6,
            iterations: 2,
            max_candidates: 12,
            max_selected: 8,
            ..Default::default()
        },
        17,
    )
    .unwrap();
    while !search.is_finished() {
        search.run_iteration(None).unwrap();
    }
    search.save(root.join("search.ron")).unwrap();
    let mut csv = String::from(
        "id,length,longest_straight,straight_fraction,turns,r10,r50,r90,curvature_variation,hairpins,chicanes,s_bends,compactness,max_curvature,min_radius,completed,progress,ticks,accepted,fitness,diversity,reason,choices\n",
    );
    let mut valid = 0;
    let mut completed = 0;
    let mut long = false;
    let mut hairpin = false;
    let mut chicane = false;
    let mut s_bend = false;
    let mut combo = false;
    let mut descriptors = vec![];
    for it in &search.iterations {
        for ant in &it.ants {
            if let Some(t) = &ant.candidate {
                let m = &t.geometry;
                let (ok, progress, ticks) = drive(t);
                fs::write(root.join(format!("{}.json",t.definition.id)),serde_json::to_string(&serde_json::json!({"definition":t.definition,"metrics":m,"decisions":t.provenance.decisions,"accepted":ant.rejection.is_none(),"fitness":ant.fitness,"reference":{"completed":ok,"progress":progress,"ticks":ticks}})).unwrap()).unwrap();
                valid += 1;
                completed += usize::from(ok);
                println!(
                    "TRACK {} L={:.0} straight={:.0} turns={} hairpins={} chicanes={} S={} compact={:.3} complete={ok} progress={progress:.4} ticks={ticks} choices={:?}",
                    t.definition.id,
                    m.length,
                    m.longest_straight,
                    m.turns,
                    m.hairpins,
                    m.chicanes,
                    m.s_bends,
                    m.compactness,
                    t.provenance.decisions
                );
                long |= m.longest_straight > 700.;
                hairpin |= m.hairpins > 0 && m.max_curvature > 0.019;
                chicane |= m.chicanes > 0;
                s_bend |= m.s_bends > 0;
                combo |=
                    m.longest_straight > 700. && m.hairpins > 0 && m.chicanes > 0 && m.s_bends > 0;
                descriptors.push(m.descriptor.clone());
                t.save(root.join(format!("{}.generated.ron", t.definition.id)))
                    .unwrap();
                t.export_definition(root.join(format!("{}.ron", t.definition.id)))
                    .unwrap();
                csv += &format!(
                    "{},{:.2},{:.2},{:.4},{},{:.2},{:.2},{:.2},{:.5},{},{},{},{:.4},{:.6},{:.2},{},{:.5},{},{},{:.4},{:.4},\"{}\",\"{:?}\"\n",
                    t.definition.id,
                    m.length,
                    m.longest_straight,
                    m.straight_fraction,
                    m.turns,
                    m.radius_quantiles[0],
                    m.radius_quantiles[1],
                    m.radius_quantiles[2],
                    m.curvature_variation,
                    m.hairpins,
                    m.chicanes,
                    m.s_bends,
                    m.compactness,
                    m.max_curvature,
                    1. / m.max_curvature,
                    ok,
                    progress,
                    ticks,
                    ant.rejection.is_none(),
                    ant.fitness.as_ref().map_or(0., |f| f.score),
                    ant.fitness.as_ref().map_or(0., |f| f.diversity),
                    ant.rejection.as_deref().unwrap_or(""),
                    t.provenance.decisions
                );
            } else {
                println!("REJECTED {} {} {:?}", it.iteration, ant.ant, ant.rejection);
            }
        }
    }
    fs::write(root.join("metrics.csv"), csv).unwrap();
    println!(
        "EVIDENCE={} valid={valid} completed={completed} long={long} hairpin={hairpin} chicane={chicane} S={s_bend} combo={combo}",
        root.display()
    );
    assert!(valid >= 5);
    assert!(
        search.state.selected.len() >= 5,
        "too few distinct accepted solutions"
    );
    assert!(long && hairpin && chicane && s_bend && combo);
    assert_eq!(
        completed, valid,
        "reference failures remain unproven drivability; investigate rather than relax geometry"
    );
    let max = descriptors
        .iter()
        .flat_map(|a| descriptors.iter().map(move |b| descriptor_distance(a, b)))
        .fold(0_f32, f32::max);
    assert!(max > 0.35, "insufficient geometric diversity");
}

/// Paired population, no retries and no discarded invalid ants. Explicit opt-in
/// because this also drives every valid candidate in the real physics engine.
#[test]
#[ignore = "writes paired visual evidence and drives 48 circuits; run explicitly"]
fn organic_geometry_comparison_battery() {
    let root = std::env::var_os("ACO_ORGANIC_EVIDENCE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/aco-organic-comparison")
        });
    let root = if root.is_absolute() {
        root
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(root)
    };
    assert!(!root.exists(), "evidence output must be new");
    fs::create_dir_all(&root).unwrap();
    let mut records = vec![];
    for organic in [false, true] {
        for seed in [17, 47] {
            let variant = if organic { "after" } else { "before" };
            let output = root.join(format!("{variant}-{seed}"));
            fs::create_dir_all(&output).unwrap();
            let mut search = SearchArchive::new(
                AcoConfig {
                    generator: GeneratorConfig {
                        organic,
                        widths: vec![44., 52., 60.],
                        primitive_radius: 50.,
                        sectors: if organic { 26 } else { 18 },
                        ..Default::default()
                    },
                    ants: 6,
                    iterations: 2,
                    max_candidates: 12,
                    max_selected: 12,
                    ..Default::default()
                },
                seed,
            )
            .unwrap();
            while !search.is_finished() {
                search.run_iteration(None).unwrap();
            }
            search.save(output.join("search.ron")).unwrap();
            for it in &search.iterations {
                for ant in &it.ants {
                    if let Some(t) = &ant.candidate {
                        let (completed, progress, ticks) = drive(t);
                        let points: Vec<_> = t
                            .definition
                            .control_points
                            .iter()
                            .map(|p| Vec2::from_array(*p))
                            .collect();
                        let metrics = organic_metrics(&points, true);
                        let close = ant.decisions.iter().find(|d| d.choice == CLOSE).unwrap();
                        let preclose = if close.position > 0 {
                            ant.decisions[close.position - 1]
                                .construction
                                .as_ref()
                                .unwrap()
                                .points
                                .len()
                        } else {
                            1
                        };
                        let explored = t.definition.control_points[..preclose.min(points.len())]
                            .windows(2)
                            .map(|w| Vec2::from_array(w[0]).distance(Vec2::from_array(w[1])))
                            .sum::<f32>();
                        let strolls = ant
                            .decisions
                            .iter()
                            .filter(|d| {
                                d.construction
                                    .as_ref()
                                    .and_then(|c| c.exploration.as_ref())
                                    .is_some_and(|e| e.chose_extra_exploration)
                            })
                            .count();
                        let record = serde_json::json!({"variant":variant,"seed":seed,"iteration":it.iteration,"ant":ant.ant,"id":t.definition.id,"length":t.geometry.length,"longest_straight":t.geometry.longest_straight,"straight_fraction":t.geometry.straight_fraction,"compactness":t.geometry.compactness,"hairpins":t.geometry.hairpins,"chicanes":t.geometry.chicanes,"s_bends":t.geometry.s_bends,"organic":metrics,"close_decision":close.position,"explored_length_fraction":explored/t.geometry.length,"extra_strolls":strolls,"accepted":ant.rejection.is_none(),"rejection":ant.rejection,"reference_completed":completed,"reference_progress":progress,"reference_ticks":ticks,"decisions":t.provenance.decisions});
                        fs::write(
                            output.join(format!("{}.json", t.definition.id)),
                            serde_json::to_vec_pretty(
                                &serde_json::json!({"definition":t.definition,"metrics":record}),
                            )
                            .unwrap(),
                        )
                        .unwrap();
                        t.save(output.join(format!("{}.generated.ron", t.definition.id)))
                            .unwrap();
                        t.export_definition(output.join(format!("{}.ron", t.definition.id)))
                            .unwrap();
                        println!(
                            "{variant} seed={seed} {} rect={:.3} axis={:.3} entropy={:.3} events={} close={} explored={:.2} complete={completed}",
                            t.definition.id,
                            metrics.rectangularity,
                            metrics.axis_concentration,
                            metrics.heading_entropy,
                            metrics.geometric_events,
                            close.position,
                            explored / t.geometry.length
                        );
                        records.push(record);
                    } else {
                        records.push(serde_json::json!({"variant":variant,"seed":seed,"iteration":it.iteration,"ant":ant.ant,"rejection":ant.rejection,"id":null}));
                    }
                }
            }
            let replay = SearchArchive::load(output.join("search.ron")).unwrap();
            assert_eq!(search.state.pheromones, replay.state.pheromones);
            replay.export_selected(output.join("selected")).unwrap();
        }
    }
    fs::write(
        root.join("comparison.json"),
        serde_json::to_vec_pretty(&records).unwrap(),
    )
    .unwrap();
    println!("ORGANIC_EVIDENCE={}", root.display());
    for variant in ["before", "after"] {
        let tracks: Vec<_> = records
            .iter()
            .filter(|r| r["variant"] == variant && !r["id"].is_null())
            .collect();
        assert!(
            tracks.len() >= 20,
            "too few valid {variant} circuits: {}",
            tracks.len()
        );
        assert!(
            tracks.iter().all(|r| r["reference_completed"] == true),
            "unproven drivability in {variant}"
        );
    }
}
#[test]
fn nearby_parallel_corridors_keep_the_current_progress_branch() {
    let c = GeneratorConfig::default();
    // Construct this fixture through actual admissible graph expansions.
    let mut state = Construction::new(&c);
    let mut choices = vec![];
    for choice in [2, 7, 2] {
        let mut options = state.options(&c, &mut || Ok(())).unwrap();
        state = options[choice]
            .next
            .take()
            .expect("parallel corridor expansion must remain closable");
        choices.push(choice);
    }
    while !state.closed() {
        let mut options = state.options(&c, &mut || Ok(())).unwrap();
        state = options[CLOSE].next.take().unwrap();
        choices.push(CLOSE);
    }
    choices.push(1);
    let t = super::super::from_decisions(&c, 55, 0, &choices).unwrap();
    let track = t.validate().unwrap();
    let n = track.samples.len();
    let mut checks = 0;
    for i in (0..n).step_by(5) {
        for j in (i + 1..n).step_by(5) {
            let a = &track.samples[i];
            let b = &track.samples[j];
            let arc = (b.cumulative_distance - a.cumulative_distance)
                .abs()
                .min(track.total_length - (b.cumulative_distance - a.cumulative_distance).abs());
            if arc > 200.
                && a.position.distance(b.position) < 160.
                && a.tangent.dot(b.tangent) < -0.8
            {
                for side in [-1., 0., 1.] {
                    let p = a.position + a.tangent.perp() * (side * track.width * 0.44);
                    let projection = track.project_near(p, i, 24);
                    let delta = crate::simulation::track::wrapped_distance_delta(
                        a.cumulative_distance,
                        projection.track_distance,
                        track.total_length,
                    );
                    assert!(
                        delta.abs() < 35.,
                        "projection jumped to a nearby future branch: {delta}"
                    );
                    checks += 1;
                }
            }
        }
    }
    assert!(
        checks > 0,
        "fixture must actually contain nearby opposite-direction corridors"
    );
    println!("nearby corridor projection: {checks} checks");
}
#[test]
fn sequential_archive_rng_versions_and_mirror_similarity_are_stable() {
    let config = AcoConfig {
        generator: GeneratorConfig {
            sectors: 8,
            ..Default::default()
        },
        ants: 1,
        iterations: 2,
        max_candidates: 2,
        ..Default::default()
    };
    let mut a = SearchArchive::new(config, 42).unwrap();
    a.run_iteration(None).unwrap();
    assert_eq!(a.version, 2);
    let mut b: SearchArchive = ron::from_str(&ron::to_string(&a).unwrap()).unwrap();
    a.run_iteration(None).unwrap();
    b.run_iteration(None).unwrap();
    assert_eq!(a.state.pheromones, b.state.pheromones);
    assert_eq!(
        a.iterations[1].ants[0].decisions,
        b.iterations[1].ants[0].decisions
    );
    b.version = 1;
    assert!(b.validate().is_err());
    let first = a
        .iterations
        .iter()
        .flat_map(|i| &i.ants)
        .find_map(|a| a.candidate.as_ref())
        .unwrap();
    let d = &first.geometry.descriptor;
    let mut mirror = d.clone();
    for i in 0..16 {
        mirror[9 + 2 * i] *= -1.;
    }
    assert!(
        descriptor_distance(d, &mirror) < 1e-6,
        "mirrors must not fabricate diversity"
    );
    let mut bad = first.clone();
    bad.provenance.version = 1;
    assert!(bad.validate().is_err());
}
#[test]
fn impossible_closure_is_rejected_with_a_reason_and_bounded_work() {
    let c = GeneratorConfig {
        max_length: 2.,
        min_length: 1.,
        max_attempts: 1,
        ..Default::default()
    };
    let mut search = SearchArchive::new(
        AcoConfig {
            generator: c,
            ants: 1,
            iterations: 1,
            max_candidates: 1,
            ..Default::default()
        },
        7,
    )
    .unwrap();
    let result = search.run_iteration(None).unwrap();
    assert_eq!(result.ants.len(), 1);
    assert_eq!(
        result.ants[0].rejection_reason,
        Some(crate::aco::RejectionReason::ClosureInfeasible)
    );
    assert!(result.ants[0].decisions.is_empty());
}
#[test]
fn la_source_equivalent_hairpin_brakes_in_advance_and_completes_in_avian() {
    let c = GeneratorConfig {
        widths: vec![44., 52., 60.],
        primitive_radius: 50.,
        ..Default::default()
    };
    let mut state = Construction::new(&c);
    let mut decisions = vec![];
    for choice in [2, 7] {
        let mut options = state.options(&c, &mut || Ok(())).unwrap();
        state = options[choice]
            .next
            .take()
            .expect("straight/hairpin must have a viable closure");
        decisions.push(choice);
    }
    while !state.closed() {
        let mut options = state.options(&c, &mut || Ok(())).unwrap();
        state = options[CLOSE].next.take().unwrap();
        decisions.push(CLOSE);
    }
    decisions.push(1);
    let generated = super::super::from_decisions(&c, 55, 0, &decisions).unwrap();
    assert!((1. / generated.geometry.max_curvature - 50.).abs() < 0.2);
    let scene = Rc::new(Scene::new(generated.validate().unwrap()));
    let cfg = Config {
        num_envs: 1,
        timeout_ticks: 10800,
        ..Default::default()
    };
    let mut env = RacingEnvironment::new(
        0,
        vec![scene],
        &cfg,
        &SimulationConfig::default(),
        ChaCha12Rng::seed_from_u64(55),
    )
    .unwrap();
    let mut peak = 0_f32;
    let mut hairpin_peak = 0_f32;
    let mut completed = false;
    for _ in 0..10800 {
        let step = env.step(env.reference_action()).unwrap();
        let arc = env.diagnostic_progress()[0];
        if arc < 650. {
            peak = peak.max(env.state.speed);
        }
        if (820.0..940.).contains(&arc) {
            hairpin_peak = hairpin_peak.max(env.state.speed);
        }
        if step.terminated || step.truncated {
            completed = step.reason == Some(EndReason::Completed);
            break;
        }
    }
    println!(
        "R50 hairpin: straight peak={peak}, hairpin peak={hairpin_peak}, completed={completed}"
    );
    assert!(completed);
    assert!(peak > 130.);
    assert!(hairpin_peak > 0. && hairpin_peak < 90.);
}
