//! Explicit hardware test: cargo test -p racing-rl --features gpu --test gpu -- --nocapture
//! Fails with Device error if no hardware adapter exists; never reports software as GPU.
#![cfg(feature = "gpu")]
use racing_rl::{
    backend::{self, Cpu, Gpu, matrix, read},
    config::Config,
    distribution,
    network::Network,
    optimizer::{Optimizer, clip_global_norm},
    ppo::{self, LossInputs},
};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
fn compare(a: Vec<f32>, b: Vec<f32>) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() <= 1e-4 + 1e-3 * b.abs(), "{a} vs {b}");
    }
}
#[test]
fn actual_hardware_forward_distribution_loss_optimizer_parity() {
    let (device, label) = backend::gpu_device(0).unwrap();
    println!("{label}");
    let c = Config {
        hidden: vec![8, 8],
        ..Default::default()
    };
    let d = c.observation.dimension();
    let cpu = Network::<Cpu>::new(&c, &Default::default()).unwrap();
    let gpu = Network::<Gpu>::new(&c, &device).unwrap();
    assert_eq!(cpu.snapshot().unwrap(), gpu.snapshot().unwrap());
    for b in [1, 8, 32] {
        let x: Vec<_> = (0..b * d).map(|i| ((i % 13) as f32 - 6.) / 12.).collect();
        let selected = matrix::<Gpu>(&x, b, d, &device).unwrap().select(
            0,
            backend::Tensor::<Gpu, 1, backend::Int>::from_data(
                backend::TensorData::new(vec![(b - 1) as i32, 0], [2]),
                &device,
            ),
        );
        compare(
            read(selected).unwrap(),
            x[(b - 1) * d..b * d]
                .iter()
                .chain(&x[..d])
                .copied()
                .collect(),
        );
        let a = cpu
            .infer(matrix(&x, b, d, &Default::default()).unwrap())
            .unwrap();
        let z = gpu.infer(matrix(&x, b, d, &device).unwrap()).unwrap();
        compare(read(a.mean.clone()).unwrap(), read(z.mean.clone()).unwrap());
        compare(
            read(a.value.clone()).unwrap(),
            read(z.value.clone()).unwrap(),
        );
        let mut r = ChaCha12Rng::seed_from_u64(8);
        let mut s = r.clone();
        let a_dist = distribution::sample(a.mean, cpu.log_std.clone(), &mut r).unwrap();
        let z_dist = distribution::sample(z.mean, gpu.log_std.clone(), &mut s).unwrap();
        compare(
            read(a_dist.u.clone()).unwrap(),
            read(z_dist.u.clone()).unwrap(),
        );
        compare(
            read(a_dist.action.clone()).unwrap(),
            read(z_dist.action.clone()).unwrap(),
        );
        compare(
            read(a_dist.joint_log_density.clone()).unwrap(),
            read(z_dist.joint_log_density.clone()).unwrap(),
        );
        let old = read(a_dist.joint_log_density.clone()).unwrap();
        let advantage: Vec<_> = (0..b)
            .map(|i| if i % 2 == 0 { 0.7 } else { -0.4 })
            .collect();
        let a_in = LossInputs {
            old_log_prob: matrix(&old, b, 1, &Default::default()).unwrap(),
            advantage: matrix(&advantage, b, 1, &Default::default()).unwrap(),
            value_target: matrix(&vec![0.3; b], b, 1, &Default::default()).unwrap(),
        };
        let z_in = LossInputs {
            old_log_prob: matrix(&old, b, 1, &device).unwrap(),
            advantage: matrix(&advantage, b, 1, &device).unwrap(),
            value_target: matrix(&vec![0.3; b], b, 1, &device).unwrap(),
        };
        let (_, a) = ppo::forward(a_dist, a.value, a_in, &c).unwrap();
        let (_, z) = ppo::forward(z_dist, z.value, z_in, &c).unwrap();
        compare(
            vec![a.total_loss, a.mean_ratio, a.latent_entropy],
            vec![z.total_loss, z.mean_ratio, z.latent_entropy],
        );
    }
    let p = vec![matrix::<Gpu>(&[1., 2.], 1, 2, &device).unwrap()];
    let g = vec![matrix::<Gpu>(&[3., 4.], 1, 2, &device).unwrap()];
    let (g, norm) = clip_global_norm(g, 1.).unwrap();
    assert!((norm - 5.).abs() < 1e-5);
    let mut optimizer = Optimizer::new(c.optimizer, &p);
    let updated = optimizer.update(&p, g, 0.1).unwrap();
    compare(read(updated[0].clone()).unwrap(), vec![0.9, 1.9]);
    backend::sync::<Gpu>(&device).unwrap();
}
