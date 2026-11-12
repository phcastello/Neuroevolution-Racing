//! Opt-in native Bevy/WGPU smoke test. All input is injected into the real egui
//! input schedule and screenshots come from the renderer, without desktop capture.
use super::*;
use bevy::{
    app::AppExit,
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
};
use bevy_egui::{EguiInput, EguiPreUpdateSet};
#[derive(Resource)]
struct Smoke {
    phase: usize,
    frame: usize,
    since: usize,
    paused: usize,
    started: Instant,
    root: PathBuf,
}
fn inject(world: &mut World) {
    world.resource_scope(|world,mut smoke: Mut<Smoke>| {
        smoke.frame += 1;
        assert!(smoke.started.elapsed().as_secs() < 90,"native smoke timed out at phase {}",smoke.phase);
        let d = world.resource::<SearchDashboard>();
        let mut action = None;
        match smoke.phase {
            0 if smoke.frame > 25 && d.buttons.contains_key("Iniciar") => { action=Some(("Iniciar",true)); smoke.phase=1; }
            1 => { action=Some(("Iniciar",false)); smoke.phase=2; }
            2 if d.decision >= 3 => { action=Some(("Pausar",true)); smoke.phase=3; }
            3 => { action=Some(("Pausar",false)); smoke.phase=4; }
            4 if d.live.as_ref().is_some_and(|l| l.control.is_paused()) => {
                assert!(d.definition.is_none()); smoke.paused=d.decision; smoke.since=smoke.frame; smoke.phase=5;
                world.spawn(Screenshot::primary_window()).observe(save_to_disk(smoke.root.join("01-partial-paused.png")));
            }
            5 => {
                assert_eq!(d.decision,smoke.paused,"rendered pause must freeze computation");
                if smoke.frame > smoke.since+15 { action=Some(("Continuar",true)); smoke.phase=6; }
            }
            6 => { action=Some(("Continuar",false)); smoke.phase=7; }
            7 if d.current.as_ref().and_then(|a|a.decisions.last()).and_then(|d|d.construction.as_ref()).is_some_and(|v|v.closing && !v.closed) => {
                world.spawn(Screenshot::primary_window()).observe(save_to_disk(smoke.root.join("02-closure-progress.png")));
                action=Some(("Pausar",true));smoke.phase=14;
            }
            14 => {action=Some(("Pausar",false));smoke.phase=15;}
            15 if d.live.as_ref().is_some_and(|l|l.control.is_paused()) => {action=Some(("Uma iteração",true));smoke.phase=16;}
            16 => {action=Some(("Uma iteração",false));smoke.phase=17;}
            17 if d.live.as_ref().is_some_and(|l| l.control.is_paused()) && d.archive.as_ref().is_some_and(|a| a.state.iteration==1) => {
                assert!(d.definition.is_some());
                let i=&d.archive.as_ref().unwrap().iterations[0]; assert_ne!(i.pheromones_before,i.pheromones_after);
                smoke.since=smoke.frame; smoke.phase=8;
                world.spawn(Screenshot::primary_window()).observe(save_to_disk(smoke.root.join("02-complete-validated.png")));
            }
            8 if smoke.frame > smoke.since+10 => { action=Some(("Continuar",true)); smoke.phase=9; }
            9 => { action=Some(("Continuar",false)); smoke.phase=10; }
            10 if d.archive.as_ref().is_some_and(SearchArchive::is_finished) && d.live.as_ref().is_some_and(LiveSearch::is_finished) => {
                let a=d.archive.as_ref().unwrap(); a.save(smoke.root.join("search.ron")).unwrap(); a.export_selected(smoke.root.join("tracks")).unwrap();
                world.resource_mut::<SearchDashboard>().tab=1; smoke.since=smoke.frame; smoke.phase=11;
            }
            11 if smoke.frame > smoke.since+5 => {
                world.spawn(Screenshot::primary_window()).observe(save_to_disk(smoke.root.join("03-pheromones.png"))); smoke.phase=12;
            }
            12 if smoke.root.join("03-pheromones.png").is_file() => {
                println!("NATIVE ACO SMOKE PASSED: pointer Start -> progressive construction -> computational pause -> iteration -> resume -> pheromones; artifacts={}",smoke.root.display());
                world.write_message(AppExit::Success); smoke.phase=13;
            }
            _ => ()
        }
        if let Some((label,pressed))=action {
            let pos=world.resource::<SearchDashboard>().buttons[label].center();
            for mut input in world.query::<&mut EguiInput>().iter_mut(world) {
                input.0.events.push(egui::Event::PointerMoved(pos));
                input.0.events.push(egui::Event::PointerButton { pos,button:egui::PointerButton::Primary,pressed,modifiers:Default::default() });
            }
        }
    });
}
#[test]
#[ignore = "opens a native window and captures GPU-rendered evidence; run explicitly"]
fn native_bevy_start_progress_pause_resume_and_capture() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/aco-rendered-smoke")
        .join(format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir_all(&root).unwrap();
    let mut app = crate::visual::base_app();
    app.world_mut().resource_mut::<Shell>().mode = crate::visual::Mode::Aco;
    {
        let mut d = app.world_mut().resource_mut::<SearchDashboard>();
        d.config.ants = 1;
        d.config.iterations = 2;
        d.speed = 5.;
    }
    app.insert_resource(Smoke {
        phase: 0,
        frame: 0,
        since: 0,
        paused: 0,
        started: Instant::now(),
        root: root.clone(),
    });
    app.add_systems(
        PreUpdate,
        inject
            .after(EguiPreUpdateSet::ProcessInput)
            .before(EguiPreUpdateSet::BeginPass),
    );
    assert!(app.run().is_success());
    for file in [
        "01-partial-paused.png",
        "02-complete-validated.png",
        "03-pheromones.png",
        "search.ron",
    ] {
        assert!(root.join(file).is_file(), "missing {file}");
    }
}

#[derive(Resource)]
struct InspectionCapture {
    frame: usize,
    root: PathBuf,
    source: PathBuf,
}
fn capture_inspection(world: &mut World) {
    world.resource_scope(|world, mut capture: Mut<InspectionCapture>| {
        capture.frame += 1;
        if capture.frame == 25 {
            world
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(
                    capture.root.join("01-valid-rejected-list.png"),
                ));
        }
        if capture.frame == 40 {
            world.resource_mut::<SearchDashboard>().tab = 0;
        }
        if capture.frame == 50 {
            world
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(
                    capture.root.join("02-valid-rejected-track.png"),
                ));
        }
        if capture.frame == 65 {
            let a = SearchArchive::load(capture.source.join("corridor-rejected.ron")).unwrap();
            let c = a.iterations[0].ants[0].clone();
            let mut d = world.resource_mut::<SearchDashboard>();
            d.seed = a.state.seed;
            d.config = a.config.clone();
            d.archive = Some(a);
            d.tab = 0;
            d.select_inspection((0, 0), c);
        }
        if capture.frame == 75 {
            world
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(capture.root.join("03-invalid-corridor.png")));
        }
        if capture.frame == 90 {
            let mut d = world.resource_mut::<SearchDashboard>();
            d.tab = 2;
            d.candidate_filter = 3;
            d.reason_filter = "Corridor".into();
        }
        if capture.frame == 100 {
            world
                .spawn(Screenshot::primary_window())
                .observe(save_to_disk(capture.root.join("04-invalid-filter.png")));
        }
        if capture.frame > 110 && capture.root.join("04-invalid-filter.png").exists() {
            println!("NATIVE INSPECTION PASSED: {}", capture.root.display());
            world.write_message(AppExit::Success);
        }
        assert!(capture.frame < 500, "inspection capture timed out");
    });
}
#[test]
#[ignore = "native GPU inspection captures; requires width/rejection evidence battery first"]
fn native_rejected_candidate_inspection_and_capture() {
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/validation/aco_width_inspection");
    let root = source.join("ui");
    std::fs::create_dir_all(&root).unwrap();
    let archive = SearchArchive::load(source.join("reference-rejected.ron")).unwrap();
    let candidate = archive.iterations[0].ants[0].clone();
    let mut app = crate::visual::base_app();
    app.world_mut().resource_mut::<Shell>().mode = crate::visual::Mode::Aco;
    {
        let mut d = app.world_mut().resource_mut::<SearchDashboard>();
        d.seed = archive.state.seed;
        d.config = archive.config.clone();
        d.archive = Some(archive);
        d.tab = 2;
        d.candidate_filter = 4;
        d.select_inspection((0, 0), candidate);
    }
    app.insert_resource(InspectionCapture {
        frame: 0,
        root: root.clone(),
        source,
    });
    app.add_systems(PreUpdate, capture_inspection);
    assert!(app.run().is_success());
    for f in [
        "01-valid-rejected-list.png",
        "02-valid-rejected-track.png",
        "03-invalid-corridor.png",
        "04-invalid-filter.png",
    ] {
        assert!(root.join(f).exists());
    }
}

#[derive(Resource)]
struct ViewportClicks {
    frame: usize,
    phase: usize,
    since: usize,
    root: PathBuf,
    best: String,
    candidate: String,
    key: (usize, usize),
    original_track: Option<String>,
}
fn assert_preview_fits(world: &mut World) {
    let bounds = world
        .resource::<crate::rendering::aco_preview::AcoPreviewBounds>()
        .0
        .unwrap();
    let mut windows = world.query::<&bevy::window::Window>();
    let window = windows.single(world).unwrap();
    let width = window.width() - crate::display::DASHBOARD_WIDTH;
    let height = window.height();
    let mut cameras = world.query_filtered::<(&Camera, &GlobalTransform), With<Camera2d>>();
    let (camera, transform) = cameras.single(world).unwrap();
    for p in [
        bounds.min,
        Vec2::new(bounds.min.x, bounds.max.y),
        bounds.max,
        Vec2::new(bounds.max.x, bounds.min.y),
    ] {
        let point = camera.world_to_viewport(transform, p.extend(0.)).unwrap();
        assert!(
            point.x >= 0.
                && point.x <= width
                && point.y >= crate::display::TOP_BAR_HEIGHT
                && point.y <= height,
            "ACO bounds outside main viewport: {point:?}; available={width}x{height}"
        );
    }
}
fn click_viewport_controls(world: &mut World) {
    world.resource_scope(|world,mut s:Mut<ViewportClicks>| {
        s.frame+=1;assert!(s.frame<1200,"main viewport click smoke timed out at {}",s.phase);
        let d=world.resource::<SearchDashboard>();
        let mut pointer=None;
        match s.phase {
            0 if s.frame>25&&d.buttons.contains_key("Ver melhor pista")=>{s.original_track=Some(world.resource::<crate::simulation::Track>().definition.id.clone());pointer=Some((d.buttons["Ver melhor pista"].center(),true));s.phase=1;}
            1=>{pointer=Some((d.buttons["Ver melhor pista"].center(),false));s.phase=2;s.since=s.frame;}
            2 if s.frame>s.since+8=>{
                assert_eq!(d.view,viewport::AcoView::Best);assert_eq!(d.viewport_scene().road.as_ref().unwrap().id,s.best);
                assert!(world.query::<&crate::rendering::aco_preview::AcoPreviewVisual>().iter(world).count()>0,"best must create real main-scene visuals");
                world.spawn(Screenshot::primary_window()).observe(save_to_disk(s.root.join("01-best-main-viewport.png")));
                world.resource_mut::<SearchDashboard>().tab=2;s.phase=3;s.since=s.frame;
            }
            3 if s.frame>s.since+8&&d.candidate_buttons.contains_key(&s.key)=>{pointer=Some((d.candidate_buttons[&s.key].center(),true));s.phase=4;}
            4=>{pointer=Some((d.candidate_buttons[&s.key].center(),false));s.phase=5;s.since=s.frame;}
            5 if s.frame>s.since+8=>{
                assert_eq!(d.view,viewport::AcoView::Candidate);assert_eq!(d.viewport_scene().road.as_ref().unwrap().id,s.candidate);
                world.spawn(Screenshot::primary_window()).observe(save_to_disk(s.root.join("02-candidate-main-viewport.png")));
                pointer=Some((world.resource::<SearchDashboard>().buttons["Ver geração"].center(),true));s.phase=6;
            }
            6=>{pointer=Some((d.buttons["Ver geração"].center(),false));s.phase=7;s.since=s.frame;}
            7 if s.frame>s.since+8=>{
                assert_eq!(d.view,viewport::AcoView::Generation);
                assert!(!d.viewport_scene().strokes.iter().any(|p|p.kind==viewport::StrokeKind::Population),"other candidates must be hidden by default");
                assert!(world.resource::<crate::rendering::aco_preview::AcoPreviewBounds>().0.is_some());
                assert_preview_fits(world);
                assert_eq!(Some(world.resource::<crate::simulation::Track>().definition.id.clone()),s.original_track,"preview must not replace the simulation Track");
                world.spawn(Screenshot::primary_window()).observe(save_to_disk(s.root.join("03-generation-main-viewport.png")));
                s.phase=8;s.since=s.frame;
            }
            8 if s.frame>s.since+8=>{world.resource_mut::<Shell>().mode=crate::visual::Mode::Menu;s.phase=9;s.since=s.frame;}
            9 if s.frame>s.since+8=>{
                assert_eq!(world.query::<&crate::rendering::aco_preview::AcoPreviewVisual>().iter(world).count(),0);
                assert!(world.resource::<crate::rendering::aco_preview::AcoPreviewBounds>().0.is_none());
                println!("NATIVE MAIN VIEWPORT CLICKS PASSED: best -> candidate -> generation -> mode cleanup; artifacts={}",s.root.display());
                world.write_message(AppExit::Success);s.phase=10;
            }
            _=>()
        }
        if let Some((pos,pressed))=pointer {for mut input in world.query::<&mut EguiInput>().iter_mut(world) {input.0.events.push(egui::Event::PointerMoved(pos));input.0.events.push(egui::Event::PointerButton{pos,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()});}}
    });
}
#[test]
#[ignore = "native GPU click validation of best/candidate/generation in the main viewport; uses saved fixed-seed ACO evidence"]
fn native_main_viewport_best_candidate_generation_buttons() {
    let archive = SearchArchive::load(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/validation/aco_width_inspection/wide-47/search.ron"),
    )
    .unwrap();
    let best = archive.state.selected[0].track.definition.id.clone();
    let (iteration, candidate) = archive
        .iterations
        .iter()
        .flat_map(|i| i.ants.iter().map(move |a| (i.iteration, a)))
        .find(|(_, a)| {
            a.candidate
                .as_ref()
                .is_some_and(|t| t.definition.id != best)
        })
        .unwrap();
    let key = (iteration, candidate.ant);
    let candidate = candidate.candidate.as_ref().unwrap().definition.id.clone();
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/aco-main-viewport")
        .join(format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
    std::fs::create_dir_all(&root).unwrap();
    let mut app = crate::visual::base_app();
    app.world_mut().resource_mut::<Shell>().mode = crate::visual::Mode::Aco;
    {
        let mut d = app.world_mut().resource_mut::<SearchDashboard>();
        d.config = archive.config.clone();
        d.seed = archive.state.seed;
        d.iteration = archive.iterations.len() - 1;
        d.archive = Some(archive);
        d.view_generation();
    }
    app.insert_resource(ViewportClicks {
        frame: 0,
        phase: 0,
        since: 0,
        root: root.clone(),
        best,
        candidate,
        key,
        original_track: None,
    });
    app.add_systems(
        PreUpdate,
        click_viewport_controls
            .after(EguiPreUpdateSet::ProcessInput)
            .before(EguiPreUpdateSet::BeginPass),
    );
    assert!(app.run().is_success());
    for f in [
        "01-best-main-viewport.png",
        "02-candidate-main-viewport.png",
        "03-generation-main-viewport.png",
    ] {
        assert!(root.join(f).is_file());
    }
}

#[derive(Resource)]
struct PpoHandoffSmoke {
    phase: usize,
    frame: usize,
    since: usize,
    started: Instant,
    root: PathBuf,
    checkpoint: PathBuf,
    expected: TrackDefinition,
    initial_position: Option<Vec2>,
}
fn drive_ppo_handoff(world: &mut World) {
    use crate::rl::viewer::{Viewer, ViewerControls};
    world.resource_scope(|world,mut s:Mut<PpoHandoffSmoke>| {
        s.frame+=1;assert!(s.started.elapsed().as_secs()<60,"ACO -> PPO timeout phase {}: {}",s.phase,world.resource::<Shell>().status);
        let mut pointer=None;
        match s.phase {
            0 if s.frame>25&&world.resource::<SearchDashboard>().buttons.contains_key("Testar com PPO")=>{
                world.spawn(Screenshot::primary_window()).observe(save_to_disk(s.root.join("01-selected-aco-track.png")));
                pointer=Some((world.resource::<SearchDashboard>().buttons["Testar com PPO"].center(),true));s.phase=1;
            }
            1=>{pointer=Some((world.resource::<SearchDashboard>().buttons["Testar com PPO"].center(),false));s.phase=2;s.since=s.frame;}
            2 if s.frame>s.since+8&&world.resource::<Shell>().checkpoint_browser.row_rects.contains_key(&s.checkpoint)=>{
                assert_eq!(world.resource::<SearchDashboard>().ppo_test_track.as_ref().unwrap().definition,s.expected);
                assert_eq!(world.resource::<Shell>().mode,crate::visual::Mode::Aco);
                world.spawn(Screenshot::primary_window()).observe(save_to_disk(s.root.join("02-checkpoint-for-selected-track.png")));
                pointer=Some((world.resource::<Shell>().checkpoint_browser.row_rects[&s.checkpoint].center(),true));s.phase=3;
            }
            3=>{pointer=Some((world.resource::<Shell>().checkpoint_browser.row_rects[&s.checkpoint].center(),false));s.phase=4;}
            4 if world.resource::<Shell>().mode==crate::visual::Mode::Ppo=>{
                let viewer=world.non_send::<Viewer>();assert_eq!(viewer.track_id,s.expected.id);assert_eq!(std::fs::canonicalize(&viewer.checkpoint_path).unwrap(),s.checkpoint);assert_eq!(viewer.saved.completed_rollouts,4000);
                assert_eq!(world.resource::<crate::simulation::Track>().definition,s.expected);
                assert!(!world.resource::<ViewerControls>().paused);assert!(world.resource::<ViewerControls>().track_requested.is_none());assert!(world.resource::<ViewerControls>().checkpoint_requested.is_none());
                let mut query=world.query_filtered::<&Transform,With<crate::simulation::SelectedCar>>();s.initial_position=Some(query.single(world).unwrap().translation.truncate());s.phase=5;
            }
            5 if world.non_send::<Viewer>().cars[0].decisions>=5=>{
                let v=world.non_send::<Viewer>();let c=&v.cars[0];
                assert!(c.physical_ticks>0&&c.elapsed>0.);
                let result=serde_json::json!({"track":v.track_id,"width":s.expected.width,"checkpoint":v.checkpoint_path,"rollouts":v.saved.completed_rollouts,"decisions":c.decisions,"physical_ticks":c.physical_ticks,"progress":c.progress,"speed":c.speed,"action":c.action,"reason":format!("{:?}",c.reason),"paused":world.resource::<ViewerControls>().paused});
                std::fs::write(s.root.join("playback.json"),serde_json::to_vec_pretty(&result).unwrap()).unwrap();
                let mut query=world.query_filtered::<&Transform,With<crate::simulation::SelectedCar>>();let p=query.single(world).unwrap().translation.truncate();assert!(p.distance(s.initial_position.unwrap())>0.001,"PPO must physically move on the selected track");
                world.resource_mut::<ViewerControls>().paused=true;s.phase=6;s.since=s.frame;
            }
            6 if s.frame>s.since+8=>{
                assert_eq!(world.query::<&crate::rendering::aco_preview::AcoPreviewVisual>().iter(world).count(),0);
                world.spawn(Screenshot::primary_window()).observe(save_to_disk(s.root.join("03-ppo-running-selected-track.png")));s.phase=7;
            }
            7 if s.root.join("03-ppo-running-selected-track.png").exists()=>{println!("NATIVE ACO -> PPO PASSED: selected track -> checkpoint click -> real inference and motion; artifacts={}",s.root.display());world.write_message(AppExit::Success);s.phase=8;}
            _=>()
        }
        if let Some((pos,pressed))=pointer {for mut input in world.query::<&mut EguiInput>().iter_mut(world) {input.0.events.push(egui::Event::PointerMoved(pos));input.0.events.push(egui::Event::PointerButton{pos,button:egui::PointerButton::Primary,pressed,modifiers:Default::default()});}}
    });
}
#[test]
#[ignore = "native end-to-end ACO track -> checkpoint picker -> PPO motion; requires the archived trained 4000-rollout checkpoint"]
fn native_selected_aco_track_runs_with_checkpoint_despite_previous_pause() {
    use crate::rl::viewer::{ViewerControls, attach};
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let original = repo.join(
        "results_ppo/e_continuation_8h/runs/ppo-1792544758769280200-19876/checkpoint-004000.ron",
    );
    let before = std::fs::read(&original).unwrap();
    let root = repo.join("target/aco-ppo-handoff").join(format!(
        "{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&root).unwrap();
    let checkpoint = root.join("checkpoint-004000.ron");
    std::fs::write(&checkpoint, &before).unwrap();
    let checkpoint = std::fs::canonicalize(checkpoint).unwrap();
    let archive = SearchArchive::load(
        repo.join("docs/validation/aco_width_inspection/reference-rejected.ron"),
    )
    .unwrap();
    let record = archive.iterations[0].ants[0].clone();
    let expected = record.candidate.as_ref().unwrap().definition.clone();
    let mut app = crate::visual::base_app();
    attach(
        app.world_mut(),
        racing_rl::persistence::load(&checkpoint).unwrap(),
        checkpoint.clone(),
        1,
        None,
        None,
    )
    .unwrap();
    {
        let mut shell = app.world_mut().resource_mut::<Shell>();
        shell.mode = crate::visual::Mode::Aco;
        shell.cars = 1;
        shell.policy_path.clear();
        shell.checkpoint_browser =
            crate::ui::ppo_checkpoint_browser::PpoCheckpointBrowser::from_roots(vec![root.clone()]);
    }
    {
        let mut c = app.world_mut().resource_mut::<ViewerControls>();
        c.paused = true;
        c.track_requested = Some("interlagos".into());
        c.checkpoint_requested = Some(PathBuf::from("stale-missing-checkpoint.ron"));
    }
    {
        let mut d = app.world_mut().resource_mut::<SearchDashboard>();
        d.config = archive.config.clone();
        d.seed = archive.state.seed;
        d.archive = Some(archive);
        d.select_inspection((0, 0), record);
        d.tab = 2;
    }
    app.insert_resource(PpoHandoffSmoke {
        phase: 0,
        frame: 0,
        since: 0,
        started: Instant::now(),
        root: root.clone(),
        checkpoint: checkpoint.clone(),
        expected,
        initial_position: None,
    });
    app.add_systems(
        PreUpdate,
        drive_ppo_handoff
            .after(EguiPreUpdateSet::ProcessInput)
            .before(EguiPreUpdateSet::BeginPass),
    );
    assert!(app.run().is_success());
    assert_eq!(std::fs::read(&original).unwrap(), before);
    assert_eq!(std::fs::read(&checkpoint).unwrap(), before);
}
