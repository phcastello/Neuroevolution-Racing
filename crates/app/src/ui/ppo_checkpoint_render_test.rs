use super::*;
use crate::{
    rl::viewer::{Viewer, ViewerControls},
    visual::{Mode, Shell},
};
use bevy::{
    app::AppExit,
    prelude::*,
    render::view::screenshot::{Screenshot, save_to_disk},
};
use bevy_egui::{EguiInput, EguiPreUpdateSet};
#[derive(Resource)]
struct NativePicker {
    phase: u8,
    frame: u64,
    since: u64,
    started: std::time::Instant,
    target: PathBuf,
    root: PathBuf,
}
fn drive(world: &mut World) {
    world.resource_scope(|world,mut smoke:Mut<NativePicker>| {
        smoke.frame+=1;
        assert!(smoke.started.elapsed().as_secs()<40,"picker smoke timeout at phase {}; {}",smoke.phase,world.resource::<Shell>().status);
        match smoke.phase {
            0 if smoke.frame>20 && world.resource::<Shell>().checkpoint_browser.row_rects.contains_key(&smoke.target)=>{
                assert!(!world.contains_non_send::<Viewer>());
                world.spawn(Screenshot::primary_window()).observe(save_to_disk(smoke.root.join("01-initial-ppo-catalog.png")));
                smoke.since=smoke.frame;smoke.phase=1;
            }
            1 if smoke.frame>smoke.since+10=>{click(world,&smoke.target,true);smoke.phase=2;}
            2=>{click(world,&smoke.target,false);smoke.phase=3;}
            3 if world.contains_non_send::<Viewer>()=>{
                let viewer=world.non_send::<Viewer>();assert_eq!(fs::canonicalize(&viewer.checkpoint_path).unwrap(),smoke.target);
                assert_eq!(viewer.saved.completed_rollouts,4000);
                world.resource_mut::<ViewerControls>().paused=true;
                smoke.since=smoke.frame;smoke.phase=4;
            }
            4 if smoke.frame>smoke.since+10=>{
                world.spawn(Screenshot::primary_window()).observe(save_to_disk(smoke.root.join("02-loaded-by-click.png")));
                smoke.phase=5;
            }
            5 if smoke.root.join("02-loaded-by-click.png").exists()=>{
                println!("NATIVE PICKER PASSED: initial list -> click archived 4000-rollout PPO -> real playback; artifacts={}",smoke.root.display());
                world.write_message(AppExit::Success);smoke.phase=6;
            }
            _=>()
        }
    });
}
fn click(world: &mut World, path: &Path, pressed: bool) {
    let pos = world.resource::<Shell>().checkpoint_browser.row_rects[path].center();
    for mut input in world.query::<&mut EguiInput>().iter_mut(world) {
        input.0.events.push(egui::Event::PointerMoved(pos));
        input.0.events.push(egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        });
    }
}
#[test]
#[ignore = "native GPU window; requires the existing archived 4000-rollout PPO"]
fn native_initial_picker_loads_archived_ppo_by_click() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let target = fs::canonicalize(repo.join(
        "results_ppo/e_continuation_8h/runs/ppo-1792544758769280200-19876/checkpoint-004000.ron",
    ))
    .expect("archived PPO required for this opt-in smoke test");
    let root = repo.join("target/ppo-picker-native").join(format!(
        "{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&root).unwrap();
    let mut app = crate::visual::base_app();
    {
        let mut shell = app.world_mut().resource_mut::<Shell>();
        shell.mode = Mode::Ppo;
        shell.cars = 1;
        shell.checkpoint_browser.filter = "004000".into();
    }
    app.insert_resource(NativePicker {
        phase: 0,
        frame: 0,
        since: 0,
        started: std::time::Instant::now(),
        target,
        root,
    });
    app.add_systems(
        PreUpdate,
        drive
            .after(EguiPreUpdateSet::ProcessInput)
            .before(EguiPreUpdateSet::BeginPass),
    );
    assert!(app.run().is_success());
}
