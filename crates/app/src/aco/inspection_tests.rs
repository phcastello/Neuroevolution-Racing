use super::inspection::*;
use super::*;
use crate::simulation::{Track, TrackDefinition};

fn output(name: &str) -> std::path::PathBuf {
    std::fs::create_dir_all(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/aco-inspection-tests"),
    )
    .unwrap();
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/aco-inspection-tests")
        .join(format!(
            "{name}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
}
#[test]
fn retention_bounds_details_without_losing_classification_or_rng() {
    let config = AcoConfig {
        ants: 2,
        iterations: 3,
        max_candidates: 6,
        max_selected: 1,
        retention: RetentionPolicy {
            detailed_candidates: 2,
            pheromone_iterations: 1,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut a = SearchArchive::new(config, 17).unwrap();
    a.run_iteration(None).unwrap();
    let mut b: SearchArchive = ron::from_str(&ron::to_string(&a).unwrap()).unwrap();
    while !a.is_finished() {
        a.run_iteration(None).unwrap();
        b.run_iteration(None).unwrap();
    }
    assert_eq!(a.state.pheromones, b.state.pheromones);
    let ants: Vec<_> = a.iterations.iter().flat_map(|i| &i.ants).collect();
    assert_eq!(ants.len(), 6);
    assert_eq!(ants.iter().filter(|a| !a.retention_evicted).count(), 2);
    assert!(ants[..4].iter().all(|a| a.geometrically_valid()
        && a.candidate.is_none()
        && !a.decisions.is_empty()
        && a.decisions.iter().all(|d| d.construction.is_none())));
    assert!(a.iterations[0].pheromones_before.is_empty());
    let root = output("retention");
    a.save(&root).unwrap();
    let loaded = SearchArchive::load(root).unwrap();
    assert_eq!(loaded.state.pheromones, a.state.pheromones);
    assert!(loaded.iterations[0].ants[0].retention_evicted);
}
#[test]
fn valid_reference_rejection_exports_and_invalid_geometry_cannot_export_as_training() {
    let mut search = SearchArchive::new(
        AcoConfig {
            ants: 1,
            iterations: 1,
            max_candidates: 1,
            policy_label: Some("test-evaluator".into()),
            ..Default::default()
        },
        47,
    )
    .unwrap();
    let mut evaluator = |_: &GeneratedTrack, _: u64, _: usize| {
        Ok(PilotEvaluation {
            reference_collision: true,
            reference_episodes: 1,
            simulation_steps: 1,
            reference_steps: 1,
            ..Default::default()
        })
    };
    search.run_iteration(Some(&mut evaluator)).unwrap();
    let record = &search.iterations[0].ants[0];
    assert!(record.geometrically_valid());
    assert_eq!(
        record.rejection_reason,
        Some(RejectionReason::ReferenceCollision)
    );
    let exported = output("valid-rejected");
    search.export_candidate(0, 0, &exported).unwrap();
    let t = GeneratedTrack::load(exported.join("track.generated.ron")).unwrap();
    t.validate().unwrap();
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
    let record = &invalid.iterations[0].ants[0];
    assert!(!record.geometrically_valid());
    assert!(record.diagnostic.as_ref().unwrap().terminal.is_some());
    assert!(
        record
            .diagnostic
            .as_ref()
            .unwrap()
            .terminal
            .as_ref()
            .unwrap()
            .alternatives
            .iter()
            .all(|a| a.rejection.is_some())
    );
    let destination = output("invalid-blocked");
    assert!(invalid.export_candidate(0, 0, &destination).is_err());
    assert!(!destination.exists());
    invalid.export_diagnostic(0, 0, &destination).unwrap();
    assert!(destination.join("diagnostic.json").exists());
    assert!(!destination.join("track.ron").exists());
    let saved = output("invalid-save");
    invalid.save(&saved).unwrap();
    let loaded = SearchArchive::load(saved).unwrap();
    assert!(
        !loaded.iterations[0].ants[0]
            .diagnostic
            .as_ref()
            .unwrap()
            .issues
            .is_empty()
    );
}
#[test]
fn diagnostics_highlight_real_sampled_intersections() {
    let d = TrackDefinition {
        control_points: vec![[0., 0.], [500., 500.], [0., 500.], [500., 0.]],
        sampled_centerline: true,
        ..procedural::definition_from_decisions(&GeneratorConfig::legacy(), &vec![2; 13]).unwrap()
    };
    assert!(Track::from_definition(&d).is_err());
    let diagnostic = GeometryDiagnostic::capture(Some(d), &[], None, "self intersection");
    assert!(
        diagnostic
            .issues
            .iter()
            .any(|i| i.message.contains("Interseção") && i.points.len() == 4)
    );
}

#[test]
fn legacy_width_configuration_does_not_gain_new_parameter_restrictions() {
    let mut value = serde_json::to_value(GeneratorConfig {
        widths: vec![120.],
        primitive_radius: 50.,
        ..Default::default()
    })
    .unwrap();
    assert!(
        serde_json::from_value::<GeneratorConfig>(value.clone())
            .unwrap()
            .validate()
            .is_err()
    );
    value
        .as_object_mut()
        .unwrap()
        .remove("width_geometry_version");
    let legacy: GeneratorConfig = serde_json::from_value(value).unwrap();
    assert_eq!(legacy.width_geometry_version, 0);
    legacy.validate().unwrap();
    assert_eq!(legacy.widths, vec![120.]);
}
