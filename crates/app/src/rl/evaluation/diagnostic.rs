//! Explicit, ignored diagnostic: only the three requested historical CPU episodes.
use super::*;

#[test]
#[ignore = "reads the three real policies only when explicitly requested"]
fn trace_near_finish_timeouts() {
    let output = PathBuf::from(
        std::env::var("PPO_DIAGNOSTIC_DIRECTORY").expect("explicit diagnostic output"),
    );
    std::fs::create_dir_all(&output).unwrap();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../results_ppo/speed_horizon_ablation/E_speed_horizon/ppo-1792524131440514300-32024",
    );
    for (rollout, id) in [
        (200, "interlagos"),
        (400, "interlagos"),
        (200, "silverstone"),
    ] {
        let saved: Checkpoint =
            persistence::load(&source.join(format!("checkpoint-{rollout:06}.ron"))).unwrap();
        let (network, _, physics) = restore::<Cpu>(&saved, &Default::default()).unwrap();
        let library = library().unwrap();
        let scene = Rc::new(Scene::new(
            Track::from_definition(library.definition(id).unwrap()).unwrap(),
        ));
        let mut environment = RacingEnvironment::new(
            0,
            vec![scene.clone()],
            &saved.config,
            &physics,
            ChaCha12Rng::seed_from_u64(saved.config.evaluation_seed),
        )
        .unwrap();
        let mut rng = ChaCha12Rng::seed_from_u64(episode_seed(
            saved.config.evaluation_seed,
            id,
            0,
            ACTION_DOMAIN,
        ));
        let path = output.join(format!("E{rollout}_{id}_trace.csv"));
        let mut file = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap(),
        );
        writeln!(file, "decision,x,y,heading,speed,steering,acceleration,projection,global_projection,segment,cumulative,best,remaining,length,movement,projection_delta,rejected,progress,termination").unwrap();
        let initial = environment.diagnostic_progress();
        writeln!(
            file,
            "0,{},{},{},{},0,0,{},{},{},0,0,{},{},0,0,false,0,initial",
            environment.state.position.x,
            environment.state.position.y,
            environment.state.heading,
            environment.state.speed,
            initial[0],
            initial[0],
            scene
                .track
                .project(environment.state.position)
                .segment_index,
            initial[3],
            initial[3]
        )
        .unwrap();
        for decision in 1..=saved.config.timeout_ticks {
            let before = environment.state;
            let previous = environment.diagnostic_progress();
            let action = action(
                &network,
                &environment.observe().unwrap(),
                EvaluationMode::Deterministic,
                &mut rng,
                &Default::default(),
            )
            .unwrap();
            let step = environment.step(action).unwrap();
            let progress = environment.diagnostic_progress();
            let global = scene.track.project(environment.state.position);
            let movement = before.position.distance(environment.state.position);
            let delta = (progress[0] - previous[0] + progress[3] * 0.5).rem_euclid(progress[3])
                - progress[3] * 0.5;
            writeln!(
                file,
                "{decision},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:?}",
                environment.state.position.x,
                environment.state.position.y,
                environment.state.heading,
                environment.state.speed,
                action[0],
                action[1],
                progress[0],
                global.track_distance,
                global.segment_index,
                progress[1],
                progress[2],
                progress[3] - progress[2],
                progress[3],
                movement,
                delta,
                delta.abs() > 4. * movement + 2.,
                step.progress,
                step.reason
            )
            .unwrap();
            if step.terminated || step.truncated {
                println!(
                    "diagnostic=E{rollout}/{id} steps={decision} reason={:?} progress={} remaining={} trace={}",
                    step.reason,
                    step.progress,
                    progress[3] - progress[2],
                    path.display()
                );
                break;
            }
        }
        file.flush().unwrap();
    }
}

#[test]
#[ignore = "explicit corrected-semantics selection of the three historical candidates"]
fn rescore_e_candidates_after_progress_correction() {
    let output = PathBuf::from(std::env::var("PPO_DIAGNOSTIC_DIRECTORY").unwrap());
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(
        "../../results_ppo/speed_horizon_ablation/E_speed_horizon/ppo-1792524131440514300-32024",
    );
    let mut writer = BufWriter::new(
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(output.join("corrected_candidate_evaluations.csv"))
            .unwrap(),
    );
    write_periodic_header(&mut writer).unwrap();
    for rollout in [200, 400, 500] {
        let path = source.join(format!("checkpoint-{rollout:06}.ron"));
        let saved: Checkpoint = persistence::load(&path).unwrap();
        saved.validate().unwrap();
        periodic(&saved, &path, "E_corrected_selection", &mut writer).unwrap();
    }
}
