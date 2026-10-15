use racing_rl::{
    Error,
    backend::{self, Cpu, Reference, Tensor, TensorData, matrix, read},
    config::{Config, ObservationKind, ObservationSpec, OptimizerKind},
    distribution, manual,
    network::Network,
    optimizer::{Optimizer, clip_global_norm},
    persistence,
    ppo::{self, LossInputs},
    rollout::{EndReason, RewardParts, Rollout, Transition},
};
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha12Rng;
fn m(v: &[f32], r: usize, c: usize) -> Tensor<Cpu, 2> {
    matrix(v, r, c, &Default::default()).unwrap()
}
fn close(a: f32, b: f32) {
    assert!((a - b).abs() < 2e-5, "{a} != {b}");
}
#[test]
fn tensors_shapes_selection_and_cpu_f64() {
    assert!(matrix::<Cpu>(&[1., 2.], 2, 2, &Default::default()).is_err());
    assert!(matrix::<Cpu>(&[f32::NAN], 1, 1, &Default::default()).is_err());
    let x = m(&[1., 2., 3., 4.], 2, 2);
    let indices = Tensor::<Cpu, 1, backend::Int>::from_data(
        TensorData::new(vec![1i64, 0, 1], [3]),
        &Default::default(),
    );
    assert_eq!(
        read(x.clone().select(0, indices)).unwrap(),
        vec![3., 4., 1., 2., 3., 4.]
    );
    assert_eq!(
        read(x.clone().matmul(x.transpose())).unwrap(),
        vec![5., 11., 11., 25.]
    );
    let x = Tensor::<Reference, 2>::from_data(
        TensorData::new(vec![1.0000000001f64, 2.0], [1, 2]),
        &Default::default(),
    );
    assert_eq!(x.into_data().to_vec::<f64>().unwrap()[0], 1.0000000001);
    assert!(!<Cpu as backend::Backend>::ad_enabled());
}
#[test]
fn observation_metadata_versioned_and_has_no_privileged_features() {
    let spec = ObservationSpec::default();
    assert_eq!(spec.dimension(), 25);
    assert_eq!(spec.features().len(), 25);
    let legacy = ObservationSpec {
        kind: ObservationKind::Legacy6,
        ..spec
    };
    assert_eq!(legacy.dimension(), 6);
    assert_eq!(legacy.angles(), vec![60., 30., 0., -30., -60.]);
    let text = ron::to_string(&legacy.features()).unwrap();
    for forbidden in [
        "fitness",
        "track_id",
        "global_position",
        "optimal_action",
        "other_car",
    ] {
        assert!(!text.contains(forbidden));
    }
}
#[test]
fn batched_equals_scalar_and_initialization_is_deterministic() {
    let c = Config {
        hidden: vec![8, 4],
        ..Default::default()
    };
    let d = c.observation.dimension();
    let n = Network::<Cpu>::new(&c, &Default::default()).unwrap();
    let n2 = Network::<Cpu>::new(&c, &Default::default()).unwrap();
    assert_eq!(n.snapshot().unwrap(), n2.snapshot().unwrap());
    let x: Vec<_> = (0..3 * d).map(|i| (i as f32 * 0.1).sin()).collect();
    let batch = n.infer(m(&x, 3, d)).unwrap();
    let mu = read(batch.mean).unwrap();
    let v = read(batch.value).unwrap();
    for i in 0..3 {
        let one = n.infer(m(&x[i * d..(i + 1) * d], 1, d)).unwrap();
        for (a, b) in read(one.mean).unwrap().iter().zip(&mu[2 * i..2 * i + 2]) {
            close(*a, *b);
        }
        close(read(one.value).unwrap()[0], v[i]);
    }
}
#[test]
fn cache_rejects_updates_and_log_std_is_projected() {
    let c = Config::default();
    let mut n = Network::<Cpu>::new(&c, &Default::default()).unwrap();
    let cache = n
        .forward(m(
            &vec![0.; c.observation.dimension()],
            1,
            c.observation.dimension(),
        ))
        .unwrap();
    let mut p = n.parameters();
    *p.last_mut().unwrap() = m(&[-9., 8.], 1, 2);
    n.replace(p, c.log_std_bounds).unwrap();
    assert!(n.validate_cache(&cache).is_err());
    assert_eq!(read(n.log_std).unwrap(), vec![-5., 2.]);
}
#[test]
fn gaussian_tanh_joint_density_and_saturation() {
    let d = distribution::evaluate(
        m(&[0., 0., 0., 0.], 2, 2),
        m(&[0., 0.], 1, 2),
        m(&[0., 0., 30., -30.], 2, 2),
    )
    .unwrap();
    let log = read(d.joint_log_density).unwrap();
    close(log[0], -(2. * std::f32::consts::PI).ln());
    assert!(log[1].is_finite());
    assert_eq!(read(d.action).unwrap(), vec![0., 0., 1., -1.]);
    let h = read(d.latent_entropy).unwrap()[0];
    close(h, (2. * std::f32::consts::PI * std::f32::consts::E).ln());
    let u = [0.7f32, -0.4];
    let density =
        distribution::evaluate(m(&[0.1, -0.2], 1, 2), m(&[-0.3, 0.2], 1, 2), m(&u, 1, 2)).unwrap();
    let expected = (0..2)
        .map(|i| {
            let mu = [0.1f64, -0.2][i];
            let ls = [-0.3f64, 0.2][i];
            let z = (u[i] as f64 - mu) / ls.exp();
            -0.5 * z * z
                - ls
                - 0.5 * (2. * std::f64::consts::PI).ln()
                - (1. - (u[i] as f64).tanh().powi(2)).ln()
        })
        .sum::<f64>();
    close(read(density.joint_log_density).unwrap()[0], expected as f32);
}
#[test]
fn independent_action_noise_and_reproducible_rng() {
    let mut a = ChaCha12Rng::seed_from_u64(3);
    let mut b = a.clone();
    let mu = m(&[0.; 16], 8, 2);
    let ls = m(&[0., 0.], 1, 2);
    let x = distribution::sample(mu.clone(), ls.clone(), &mut a).unwrap();
    let y = distribution::sample(mu, ls, &mut b).unwrap();
    let x = read(x.u).unwrap();
    assert_eq!(x, read(y.u).unwrap());
    assert!(x.chunks(2).skip(1).any(|row| row != &x[..2]));
}
fn transition(
    env: usize,
    episode: u64,
    reward: f32,
    value: f32,
    next: f32,
    reason: Option<EndReason>,
) -> Transition {
    Transition {
        env_id: env,
        episode_id: episode,
        observation: vec![0.; 6],
        action: [0.; 2],
        u: [0.; 2],
        old_log_prob: 0.,
        old_value: value,
        next_observation: vec![0.; 6],
        next_value: next,
        reward: RewardParts {
            progress: reward,
            ..Default::default()
        },
        terminated: matches!(reason, Some(EndReason::Completed | EndReason::Collision)),
        truncated: reason == Some(EndReason::Timeout),
        reason,
        physical_ticks: 1,
        progress: 0.,
    }
}
fn rollout(transitions: Vec<Transition>, n: usize, t: usize) -> Rollout {
    Rollout {
        version: 1,
        policy_version: 0,
        num_envs: n,
        rollout_length: t,
        observation_spec: ObservationSpec {
            kind: ObservationKind::Legacy6,
            ..Default::default()
        },
        transitions,
        raw_advantages: vec![],
        policy_advantages: vec![],
        value_targets: vec![],
        timings: Default::default(),
    }
}
#[test]
fn gae_continuation_termination_timeout_and_rollout_end() {
    let mut r = rollout(
        vec![
            transition(0, 0, 1., 2., 3., None),
            transition(1, 0, 2., 4., 100., Some(EndReason::Collision)),
            transition(0, 0, 2., 3., 4., Some(EndReason::Timeout)),
            transition(1, 1, 3., 5., 6., None),
        ],
        2,
        2,
    );
    r.prepare(0.9, 0.8).unwrap();
    let expected = [1.7 + 0.72 * 2.6, -2., 2.6, 3.4];
    for (a, e) in r.raw_advantages.iter().zip(expected) {
        close(*a, e);
    }
    for i in 0..4 {
        close(
            r.value_targets[i],
            r.raw_advantages[i] + r.transitions[i].old_value,
        );
    }
    assert!((r.policy_advantages.iter().sum::<f32>()).abs() < 1e-5);
    let mut zero = rollout(vec![transition(0, 0, 0., 0., 0., None)], 1, 1);
    zero.prepare(1., 1.).unwrap();
    assert_eq!(zero.policy_advantages, vec![0.]);
}
#[test]
fn minibatches_cover_all_rows_including_smaller_tail() {
    let mut r = rollout(
        (0..10)
            .map(|_| transition(0, 0, 0., 0., 0., None))
            .collect(),
        1,
        10,
    );
    r.prepare(0.99, 0.95).unwrap();
    let mut rng = ChaCha12Rng::seed_from_u64(8);
    let batches = r.minibatches(4, &mut rng).unwrap();
    assert_eq!(
        batches.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![4, 4, 2]
    );
    let mut flat = batches.into_iter().flatten().collect::<Vec<_>>();
    flat.sort();
    assert_eq!(flat, (0..10).collect::<Vec<_>>());
    assert!(r.minibatches(0, &mut rng).is_err());
}
#[test]
fn loss_forward_positive_negative_zero_advantages_and_clipping() {
    let ratios: [f32; 6] = [0.5, 1., 1.5, 0.5, 1.5, 1.7];
    let advantage = [1., 1., 1., -1., -1., 0.];
    let dist =
        distribution::evaluate(m(&[0.; 12], 6, 2), m(&[0.; 2], 1, 2), m(&[0.; 12], 6, 2)).unwrap();
    let logs = read(dist.joint_log_density.clone()).unwrap();
    let old: Vec<_> = logs.iter().zip(ratios).map(|(p, r)| p - r.ln()).collect();
    let inputs = LossInputs {
        old_log_prob: m(&old, 6, 1),
        advantage: m(&advantage, 6, 1),
        value_target: m(&[1.; 6], 6, 1),
    };
    let c = Config::default();
    let (_, loss) = ppo::forward(dist, m(&[0.; 6], 6, 1), inputs, &c).unwrap();
    close(loss.policy_loss, -(0.5 + 1. + 1.2 - 0.8 - 1.5) / 6.);
    close(loss.value_loss, 0.5);
    close(loss.total_loss, loss.policy_loss + 0.25);
    close(loss.clip_fraction, 5. / 6.);
}
#[test]
fn optimizers_known_gradients_and_global_clipping() {
    let p = vec![m(&[1., 2.], 1, 2), m(&[3.], 1, 1)];
    let g = vec![m(&[3., 0.], 1, 2), m(&[4.], 1, 1)];
    let (g, norm) = clip_global_norm(g, 1.).unwrap();
    close(norm, 5.);
    close(read(g[0].clone()).unwrap()[0], 0.6);
    close(read(g[1].clone()).unwrap()[0], 0.8);
    let mut sgd = Optimizer::new(OptimizerKind::Sgd, &p);
    let updated = sgd.update(&p, g, 0.1).unwrap();
    close(read(updated[0].clone()).unwrap()[0], 0.94);
    let kind = OptimizerKind::Adam {
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    };
    let mut adam = Optimizer::new(kind, &p);
    let gradients = || vec![m(&[2., -2.], 1, 2), m(&[0.], 1, 1)];
    let one = adam.update(&p, gradients(), 0.1).unwrap();
    let values = read(one[0].clone()).unwrap();
    close(values[0], 0.9);
    close(values[1], 2.1);
    let saved = adam.snapshot().unwrap();
    let mut resumed = Optimizer::restore(&saved, &one).unwrap();
    let a = adam.update(&one, gradients(), 0.1).unwrap();
    let b = resumed.update(&one, gradients(), 0.1).unwrap();
    assert_eq!(read(a[0].clone()).unwrap(), read(b[0].clone()).unwrap());
    assert_eq!(adam.step, 2);
    assert!(sgd.update(&p, vec![m(&[0.], 1, 1)], 0.1).is_err());
}
#[test]
fn persistence_is_atomic_no_overwrite_and_rng_roundtrip() {
    let c = Config::default();
    let mut rngs = persistence::Rngs::new(&c);
    let serialized = ron::to_string(&rngs).unwrap();
    let mut restored: persistence::Rngs = ron::from_str(&serialized).unwrap();
    assert_eq!(rngs.action.random::<u64>(), restored.action.random::<u64>());
    let dir = persistence::run_directory(&std::env::temp_dir().join("racing-rl-tests")).unwrap();
    let path = dir.join("config.ron");
    persistence::save(&path, &c).unwrap();
    let restored: Config = persistence::load(&path).unwrap();
    assert_eq!(c, restored);
    assert!(persistence::save(&path, &c).is_err());
}
#[test]
fn manual_suite_reports_pending_instead_of_fake_success() {
    for case in racing_rl::verification::CASES
        .iter()
        .filter(|s| **s != "all")
    {
        match racing_rl::verification::check_case(case) {
            Err(Error::MissingManualImplementation { function }) => {
                assert!(manual::FUNCTIONS.contains(&function));
                eprintln!("PENDENTE: {case}: {function}");
            }
            Ok(()) => eprintln!("APROVADO: {case}"),
            Err(e) => panic!("{case}: {e}"),
        }
    }
}

#[test]
fn presets_and_preflight_limits_are_validated() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/rl");
    for name in ["smoke", "base", "stress"] {
        let c = Config::load(&root.join(format!("{name}.ron"))).unwrap();
        assert_eq!(c.observation.dimension(), 25);
        assert!(c.estimated_bytes() < c.memory_limit_mib as u128 * 1024 * 1024);
    }
    let mut c = Config {
        gamma: f32::NAN,
        ..Default::default()
    };
    assert!(c.validate().is_err());
    c = Config {
        memory_limit_mib: 0,
        ..Default::default()
    };
    assert!(c.validate().is_err());
    c = Config {
        physical_dt: 1. / 30.,
        ..Default::default()
    };
    assert!(c.validate().is_err());
    c = Config {
        num_envs: usize::MAX,
        ..Default::default()
    };
    assert!(c.validate().is_err());
}

#[test]
fn ppo_checkpoint_roundtrip_validates_shapes_state_and_cache_restoration() {
    let c = Config {
        hidden: vec![3],
        ..Default::default()
    };
    let n = Network::<Cpu>::new(&c, &Default::default()).unwrap();
    let optimizer = Optimizer::new(c.optimizer.clone(), &n.parameters());
    let checkpoint = persistence::Checkpoint {
        format: "racing-ppo".into(),
        version: 1,
        config: c.clone(),
        features: c.observation.features(),
        parameter_shapes: n.parameters().iter().map(|p| p.dims()).collect(),
        parameters: n.snapshot().unwrap(),
        policy_version: 0,
        optimizer: optimizer.snapshot().unwrap(),
        rngs: persistence::Rngs::new(&c),
        completed_rollouts: 0,
        decisions: 0,
        physical_ticks: 0,
        rendered_frames: 0,
        backend: "CPU test".into(),
        trained: false,
        physics_ron: "application-owned test configuration".into(),
        resume_policy: "restart environments".into(),
    };
    checkpoint.validate().unwrap();
    let text = ron::to_string(&checkpoint).unwrap();
    let mut restored: persistence::Checkpoint = ron::from_str(&text).unwrap();
    restored.validate().unwrap();
    let mut other = Network::<Cpu>::new(&c, &Default::default()).unwrap();
    let cache = other
        .forward(m(
            &vec![0.; c.observation.dimension()],
            1,
            c.observation.dimension(),
        ))
        .unwrap();
    other
        .restore(&restored.parameters, 0, c.log_std_bounds)
        .unwrap();
    assert_eq!(n.snapshot().unwrap(), other.snapshot().unwrap());
    assert!(other.validate_cache(&cache).is_err());
    restored.optimizer.second[0][0] = -1.;
    assert!(restored.validate().is_err());
}
