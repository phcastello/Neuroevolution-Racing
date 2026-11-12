use super::*;
use racing_rl::{backend::Cpu, config::Config, network::Network, optimizer::Optimizer};

pub(crate) fn saved_policy(rollout: u64) -> Checkpoint {
    let config = Config {
        hidden: vec![3, 2],
        ..Default::default()
    };
    let network = Network::<Cpu>::new(&config, &Default::default()).unwrap();
    let optimizer = Optimizer::new(config.optimizer.clone(), &network.parameters());
    Checkpoint {
        format: "racing-ppo".into(),
        version: 1,
        features: config.observation.features(),
        parameter_shapes: network
            .parameters()
            .iter()
            .map(|parameter| parameter.dims())
            .collect(),
        parameters: network.snapshot().unwrap(),
        optimizer: optimizer.snapshot().unwrap(),
        rngs: persistence::Rngs::new(&config),
        config,
        policy_version: rollout,
        completed_rollouts: rollout,
        decisions: 0,
        physical_ticks: 0,
        rendered_frames: 0,
        backend: "history test CPU".into(),
        trained: false,
        physics_ron: String::new(),
        resume_policy: "restart environments".into(),
        curriculum_ron: None,
    }
}

#[test]
fn policy_history_uses_checkpoint_rollout_and_keeps_navigation_in_its_run() {
    let directory =
        std::env::temp_dir().join(format!("racing-ppo-policy-history-{}", std::process::id()));
    let early = directory.join("run/checkpoint-000050.ron");
    let late = directory.join("run/checkpoint-000001.ron");
    let other = directory.join("other/checkpoint-000025.ron");
    let saved = saved_policy(1);
    persistence::save(&early, &saved).unwrap();
    persistence::save(&late, &saved_policy(50)).unwrap();
    persistence::save(&other, &saved_policy(25)).unwrap();
    let mut browser = PpoCheckpointBrowser {
        roots: vec![directory.clone()],
        ..Default::default()
    };
    browser.refresh();
    finish(&mut browser);
    let summary = load_summary(&early).unwrap();
    assert_eq!(summary.rollout, 1);
    assert_eq!(summary.parameters, saved.config.parameter_count());
    assert_eq!(summary.architecture, "25 -> 3 -> 2 -> Actor 2 / Critic 1");
    assert_eq!(
        browser.neighbor(&early, 1),
        Some(fs::canonicalize(&late).unwrap())
    );
    assert_eq!(
        browser.neighbor(&late, -1),
        Some(fs::canonicalize(&early).unwrap())
    );
    assert!(browser.neighbor(&early, -1).is_none());
    assert!(browser.neighbor(&late, 1).is_none());
    assert!(browser.neighbor(&other, 1).is_none());
    for path in [&early, &late, &other] {
        fs::remove_file(path).unwrap();
    }
    fs::remove_dir(directory.join("run")).unwrap();
    fs::remove_dir(directory.join("other")).unwrap();
    fs::remove_dir(directory).unwrap();
}

#[test]
fn discovery_is_recursive_deduplicated_and_excludes_ga_and_other_files() {
    let directory = std::env::temp_dir().join(format!("racing-ppo-history-{}", std::process::id()));
    let nested = directory.join("run/subrun");
    fs::create_dir_all(&nested).unwrap();
    fs::write(nested.join("checkpoint-000025.ron"), "invalid PPO").unwrap();
    fs::write(nested.join("generation-25.ron"), "GA checkpoint").unwrap();
    fs::write(nested.join("checkpoint-000025.tmp"), "partial").unwrap();
    let mut browser = PpoCheckpointBrowser {
        roots: normalize_roots(vec![directory.clone(), nested]),
        ..Default::default()
    };
    browser.refresh();
    finish(&mut browser);
    assert_eq!(browser.entries.len(), 1);
    assert!(browser.entries[0].summary.as_ref().unwrap().is_err());
    assert!(browser.status.contains("1 com erro"));
    // Delete only the exact test-owned files, never recursively remove a computed directory.
    fs::remove_file(directory.join("run/subrun/checkpoint-000025.ron")).unwrap();
    fs::remove_file(directory.join("run/subrun/generation-25.ron")).unwrap();
    fs::remove_file(directory.join("run/subrun/checkpoint-000025.tmp")).unwrap();
    fs::remove_dir(directory.join("run/subrun")).unwrap();
    fs::remove_dir(directory.join("run")).unwrap();
    fs::remove_dir(directory).unwrap();
}

fn finish(browser: &mut PpoCheckpointBrowser) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while browser.scan.is_some() {
        browser.poll();
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}
#[test]
fn large_checkpoint_metadata_matches_validated_policy_without_loading_weights() {
    let root = std::env::temp_dir().join(format!("ppo-fast-catalog-{}", std::process::id()));
    let path = root.join("checkpoint-000007.ron");
    let mut saved = saved_policy(42);
    saved.config.hidden = vec![64, 64];
    let network = Network::<Cpu>::new(&saved.config, &Default::default()).unwrap();
    let optimizer = Optimizer::new(saved.config.optimizer.clone(), &network.parameters());
    saved.parameters = network.snapshot().unwrap();
    saved.parameter_shapes = network.parameters().iter().map(|p| p.dims()).collect();
    saved.optimizer = optimizer.snapshot().unwrap();
    persistence::save(&path, &saved).unwrap();
    assert!(fs::metadata(&path).unwrap().len() > 131072);
    let preview = fast_summary(&path).unwrap();
    assert_eq!(preview.rollout, 42);
    assert_eq!(preview.parameters, saved.config.parameter_count());
    assert!(preview.metadata_only);
    assert_eq!(preview.architecture, "25 -> 64 -> 64 -> Actor 2 / Critic 1");
    fs::remove_file(path).unwrap();
    fs::remove_dir(root).unwrap();
}
fn picker_frame(
    browser: &mut PpoCheckpointBrowser,
    ctx: &egui::Context,
    selected: &mut String,
    events: Vec<egui::Event>,
) -> bool {
    let mut clicked = false;
    let mut output = ctx.run_ui(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(900., 700.),
            )),
            events,
            ..Default::default()
        },
        |ui| {
            clicked = draw_policy_picker(ui, browser, selected);
        },
    );
    output.textures_delta.clear();
    clicked
}
#[test]
fn initial_catalog_discovers_nested_checkpoints_and_selects_with_pointer_only() {
    let root = std::env::temp_dir().join(format!("ppo-catalog-click-{}", std::process::id()));
    let path = root.join("continuation/runs/trained/checkpoint-004000.ron");
    persistence::save(&path, &saved_policy(4000)).unwrap();
    let mut browser = PpoCheckpointBrowser {
        roots: vec![root.clone()],
        ..Default::default()
    };
    let ctx = egui::Context::default();
    let mut selected = String::new();
    picker_frame(&mut browser, &ctx, &mut selected, vec![]);
    finish(&mut browser);
    for _ in 0..2 {
        picker_frame(&mut browser, &ctx, &mut selected, vec![]);
    }
    assert!(selected.is_empty());
    let path = fs::canonicalize(path).unwrap();
    let pos = browser.row_rects[&path].center();
    picker_frame(
        &mut browser,
        &ctx,
        &mut selected,
        vec![egui::Event::PointerMoved(pos)],
    );
    for pressed in [true, false] {
        picker_frame(
            &mut browser,
            &ctx,
            &mut selected,
            vec![egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed,
                modifiers: Default::default(),
            }],
        );
    }
    assert_eq!(PathBuf::from(selected), path);
    assert!(browser.status.contains("1 checkpoints"));
    fs::remove_file(&path).unwrap();
    fs::remove_dir(path.parent().unwrap()).unwrap();
    fs::remove_dir(root.join("continuation/runs")).unwrap();
    fs::remove_dir(root.join("continuation")).unwrap();
    fs::remove_dir(root).unwrap();
}
