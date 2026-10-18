//! Explicit hardware test: cargo test -p racing-rl --features gpu --test gpu -- --nocapture
//! Fails with Device error if no hardware adapter exists; never reports software as GPU.
#![cfg(feature = "gpu")]
use racing_rl::{
    backend::{self, Cpu, Gpu, matrix, read},
    config::Config,
    distribution, manual,
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
fn actual_hardware_backward_clipping_entropy_and_adam_parity() {
    let (device, label) = backend::gpu_device(0).unwrap();
    println!("backward: {label}");
    for batch in [9, 18] {
        for entropy_coef in [0., 0.017] {
            let config = Config {
                hidden: vec![8, 5],
                latent_entropy_coef: entropy_coef,
                ..Default::default()
            };
            let dim = config.observation.dimension();
            let cpu = Network::<Cpu>::new(&config, &Default::default()).unwrap();
            let gpu = Network::<Gpu>::new(&config, &device).unwrap();
            let observations: Vec<_> = (0..batch * dim)
                .map(|i| ((i % 17) as f32 - 8.) / 17.)
                .collect();
            let u: Vec<_> = (0..batch * 2).map(|i| ((i % 7) as f32 - 3.) / 5.).collect();
            let cpu_cache = cpu
                .forward(matrix(&observations, batch, dim, &Default::default()).unwrap())
                .unwrap();
            let gpu_cache = gpu
                .forward(matrix(&observations, batch, dim, &device).unwrap())
                .unwrap();
            let cpu_distribution = distribution::evaluate(
                cpu_cache.mean.clone(),
                cpu.log_std.clone(),
                matrix(&u, batch, 2, &Default::default()).unwrap(),
            )
            .unwrap();
            let gpu_distribution = distribution::evaluate(
                gpu_cache.mean.clone(),
                gpu.log_std.clone(),
                matrix(&u, batch, 2, &device).unwrap(),
            )
            .unwrap();
            // Mesmos dados históricos nos dois dispositivos; grade completa de clipping.
            let old: Vec<_> = read(cpu_distribution.joint_log_density.clone())
                .unwrap()
                .into_iter()
                .enumerate()
                .map(|(i, log_density)| log_density - [0.55_f32, 1.05, 1.55][i % 3].ln())
                .collect();
            let advantages: Vec<_> = (0..batch).map(|i| [0.7, -0.4, 0.][i / 3 % 3]).collect();
            let (cpu_loss, _) = ppo::forward(
                cpu_distribution,
                cpu_cache.value.clone(),
                LossInputs {
                    old_log_prob: matrix(&old, batch, 1, &Default::default()).unwrap(),
                    advantage: matrix(&advantages, batch, 1, &Default::default()).unwrap(),
                    value_target: matrix(&vec![0.3; batch], batch, 1, &Default::default()).unwrap(),
                },
                &config,
            )
            .unwrap();
            let (gpu_loss, _) = ppo::forward(
                gpu_distribution,
                gpu_cache.value.clone(),
                LossInputs {
                    old_log_prob: matrix(&old, batch, 1, &device).unwrap(),
                    advantage: matrix(&advantages, batch, 1, &device).unwrap(),
                    value_target: matrix(&vec![0.3; batch], batch, 1, &device).unwrap(),
                },
                &config,
            )
            .unwrap();
            // Confere as regiões antes de comparar qualquer derivada.
            let expected_ratios: Vec<_> = (0..batch).map(|i| [0.55, 1.05, 1.55][i % 3]).collect();
            compare(
                read(cpu_loss.ratio.clone()).unwrap(),
                expected_ratios.clone(),
            );
            compare(read(gpu_loss.ratio.clone()).unwrap(), expected_ratios);
            let cpu_outputs = manual::ppo_output_gradients(&cpu_loss).unwrap();
            let gpu_outputs = manual::ppo_output_gradients(&gpu_loss).unwrap();
            for (cpu_tensor, gpu_tensor) in [
                (cpu_outputs.mean.clone(), gpu_outputs.mean.clone()),
                (cpu_outputs.value.clone(), gpu_outputs.value.clone()),
                (cpu_outputs.log_std.clone(), gpu_outputs.log_std.clone()),
            ] {
                compare(read(cpu_tensor).unwrap(), read(gpu_tensor).unwrap());
            }
            let cpu_gradients =
                manual::shared_heads_backward(&cpu, cpu_cache, cpu_outputs).unwrap();
            let gpu_gradients =
                manual::shared_heads_backward(&gpu, gpu_cache, gpu_outputs).unwrap();
            for (cpu_tensor, gpu_tensor) in cpu_gradients.iter().zip(&gpu_gradients) {
                compare(
                    read(cpu_tensor.clone()).unwrap(),
                    read(gpu_tensor.clone()).unwrap(),
                );
            }
            let cpu_parameters = cpu.parameters();
            let gpu_parameters = gpu.parameters();
            let mut cpu_adam = Optimizer::new(config.optimizer.clone(), &cpu_parameters);
            let mut gpu_adam = Optimizer::new(config.optimizer.clone(), &gpu_parameters);
            let (cpu_gradients, _) = clip_global_norm(cpu_gradients, 0.5).unwrap();
            let (gpu_gradients, _) = clip_global_norm(gpu_gradients, 0.5).unwrap();
            let cpu_updated = cpu_adam
                .update(&cpu_parameters, cpu_gradients, 0.0003)
                .unwrap();
            let gpu_updated = gpu_adam
                .update(&gpu_parameters, gpu_gradients, 0.0003)
                .unwrap();
            for (cpu_tensor, gpu_tensor) in cpu_updated.into_iter().zip(gpu_updated) {
                compare(read(cpu_tensor).unwrap(), read(gpu_tensor).unwrap());
            }
            println!("paridade backward/Adam: B={batch}, entropia={entropy_coef}");
        }
    }
    backend::sync::<Gpu>(&device).unwrap();
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
