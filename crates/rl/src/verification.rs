//! Test-only finite differences. Never used by trainer/update, never a production gradient source.
//! CPU f64, fixed weights/inputs/u; central step 1e-5, abs 2e-6 + rel 2e-4 tolerance.
use crate::{
    Result,
    backend::{Reference as R, Tensor, TensorData},
    config::Config,
    distribution, ensure,
    manual::{self, Activation, OutputGradients},
    network::{Dense, Network},
    ppo::{self, LossInputs},
};
pub const CASES: &[&str] = &[
    "activation",
    "dense",
    "trunk",
    "heads",
    "log_std",
    "ppo",
    "synthetic",
    "all",
];
/// Infrastructure preflight: invoke every real hook without numerical differences.
/// Finite differences are confined to the explicit check_case validation path.
pub fn probe_hooks() -> Result<()> {
    let z = tensor(&[-0.8, 0.6, 1.2, -0.4], 2, 2);
    let upstream = tensor(&[0.2, -0.3, 0.7, 0.1], 2, 2);
    for kind in [Activation::Relu, Activation::Linear, Activation::Tanh] {
        let activated = match kind {
            Activation::Relu => crate::backend::activation::relu(z.clone()),
            Activation::Linear => z.clone(),
            Activation::Tanh => z.clone().tanh(),
        };
        let gradient = manual::activation_backward(kind, z.clone(), activated, upstream.clone())?;
        crate::backend::validate(&gradient, "activation preflight")?;
    }
    let (n, c) = fixture()?;
    let cache = n.forward(input(&n))?;
    let dense = manual::dense_backward(
        &n.actor,
        cache.head_input.clone(),
        tensor(&[0.2, -0.3, 0.1, 0.4], 2, 2),
    )?;
    let trunk = manual::trunk_backward(&n.trunk, &cache.trunk, dense.input)?;
    for gradient in trunk.parameters {
        crate::backend::validate(&gradient, "trunk preflight")?;
    }
    let (cache, loss) = loss(&n, &c)?;
    let outputs = manual::ppo_output_gradients(&loss)?;
    let gradients = manual::shared_heads_backward(&n, cache, outputs)?;
    crate::optimizer::validate_gradients(&n.parameters(), &gradients)
}
fn tensor(v: &[f64], r: usize, c: usize) -> Tensor<R, 2> {
    Tensor::from_data(TensorData::new(v.to_vec(), [r, c]), &Default::default())
}
fn data(t: Tensor<R, 2>) -> Vec<f64> {
    t.into_data().to_vec::<f64>().unwrap()
}
fn scalar(t: Tensor<R, 1>) -> f64 {
    t.into_data().to_vec::<f64>().unwrap()[0]
}
fn dot(a: Tensor<R, 2>, b: Tensor<R, 2>) -> f64 {
    scalar((a * b).sum())
}
fn compare(actual: &[f64], expected: &[f64]) -> Result<()> {
    ensure(actual.len() == expected.len(), "gradient checker length")?;
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        ensure(
            a.is_finite() && e.is_finite() && (a - e).abs() <= 2e-6 + 2e-4 * e.abs(),
            format!("gradient check index {i}: analytic={a}, numerical={e}"),
        )?;
    }
    Ok(())
}
/// Central finite differences of a frozen scalar objective; only validation calls this.
pub fn central<F: FnMut(&[Vec<f64>]) -> Result<f64>>(
    parameters: &[Vec<f64>],
    mut objective: F,
) -> Result<Vec<Vec<f64>>> {
    let mut gradients = parameters.to_vec();
    for (j, parameter) in parameters.iter().enumerate() {
        for k in 0..parameter.len() {
            let mut plus = parameters.to_vec();
            let mut minus = parameters.to_vec();
            let h = 1e-5;
            plus[j][k] += h;
            minus[j][k] -= h;
            gradients[j][k] = (objective(&plus)? - objective(&minus)?) / (2. * h);
        }
    }
    Ok(gradients)
}
fn fixture() -> Result<(Network<R>, Config)> {
    let c = Config {
        hidden: vec![3, 2],
        ..Default::default()
    };
    let mut n = Network::new(&c, &Default::default())?;
    let mut p = n.parameters();
    // Nonzero positive hidden preactivations keep numerical checks away from ReLU kinks.
    for (i, t) in p.iter_mut().enumerate() {
        let [r, c] = t.dims();
        let v: Vec<_> = (0..r * c)
            .map(|j| {
                if i % 2 == 1 {
                    0.4
                } else {
                    0.02 + 0.001 * j as f64
                }
            })
            .collect();
        *t = tensor(&v, r, c);
    }
    n.replace(p, c.log_std_bounds)?;
    Ok((n, c))
}
fn input(n: &Network<R>) -> Tensor<R, 2> {
    let d = n.input_size;
    tensor(
        &(0..2 * d)
            .map(|i| 0.1 + (i % 5) as f64 * 0.02)
            .collect::<Vec<_>>(),
        2,
        d,
    )
}
fn snapshot(n: &Network<R>) -> Vec<Vec<f64>> {
    n.parameters().into_iter().map(data).collect()
}
fn replace(n: &mut Network<R>, p: &[Vec<f64>], c: &Config) -> Result<()> {
    let old = n.parameters();
    n.replace(
        p.iter()
            .zip(old)
            .map(|(v, t)| {
                let [r, c] = t.dims();
                tensor(v, r, c)
            })
            .collect(),
        c.log_std_bounds,
    )
}
fn loss(
    n: &Network<R>,
    c: &Config,
) -> Result<(crate::network::ForwardCache<R>, crate::ppo::LossCache<R>)> {
    let cache = n.forward(input(n))?;
    let dist = distribution::evaluate(
        cache.mean.clone(),
        cache.log_std.clone(),
        tensor(&[0.24, -0.31, 0.37, -0.27], 2, 2),
    )?;
    let inputs = LossInputs {
        old_log_prob: tensor(&[-2.1, -2.0], 2, 1),
        advantage: tensor(&[0.7, -0.4], 2, 1),
        value_target: tensor(&[0.9, -0.7], 2, 1),
    };
    let (loss, _) = ppo::forward(dist, cache.value.clone(), inputs, c)?;
    Ok((cache, loss))
}
pub fn check_case(case: &str) -> Result<()> {
    ensure(
        CASES.contains(&case),
        format!("unknown gradient case {case}; cases={CASES:?}"),
    )?;
    if case == "all" {
        for name in &CASES[..CASES.len() - 1] {
            check_case(name)?;
        }
        return Ok(());
    }
    match case {
        "activation" => {
            for kind in [Activation::Relu, Activation::Linear, Activation::Tanh] {
                let x = vec![vec![-0.8, 0.6, 1.2, -0.4]];
                let upstream = tensor(&[0.2, -0.3, 0.7, 0.1], 2, 2);
                let apply = |x: Tensor<R, 2>| match kind {
                    Activation::Relu => crate::backend::activation::relu(x),
                    Activation::Linear => x,
                    Activation::Tanh => x.tanh(),
                };
                let z = tensor(&x[0], 2, 2);
                let grad =
                    manual::activation_backward(kind, z.clone(), apply(z), upstream.clone())?;
                let numeric = central(&x, |p| {
                    Ok(dot(apply(tensor(&p[0], 2, 2)), upstream.clone()))
                })?;
                compare(&data(grad), &numeric[0])?;
            }
            Ok(())
        }
        "dense" => {
            let p = vec![
                vec![0.2, -0.3, 0.1, 0.4],
                vec![0.5, -0.2],
                vec![0.6, -0.4, 0.7, 0.2],
            ];
            let make = |p: &[Vec<f64>]| Dense {
                weights: tensor(&p[0], 2, 2),
                bias: tensor(&p[1], 1, 2),
            };
            let upstream = tensor(&[0.3, 0.7, -0.2, 0.5], 2, 2);
            let g = manual::dense_backward(&make(&p), tensor(&p[2], 2, 2), upstream.clone())?;
            let numerical = central(&p, |p| {
                Ok(dot(make(p).forward(tensor(&p[2], 2, 2)), upstream.clone()))
            })?;
            for (a, e) in [g.weights, g.bias, g.input].into_iter().zip(numerical) {
                compare(&data(a), &e)?;
            }
            Ok(())
        }
        "trunk" => {
            let (mut n, c) = fixture()?;
            let cache = n.forward(input(&n))?;
            let up = tensor(&[0.3, -0.2, 0.1, 0.4], 2, 2);
            let g = manual::trunk_backward(&n.trunk, &cache.trunk, up.clone())?;
            let original = snapshot(&n);
            let mut p = original.clone();
            let count = n.trunk.len() * 2;
            p.truncate(count);
            let numerical = central(&p, |p| {
                let mut all = snapshot(&n);
                all[..count].clone_from_slice(p);
                replace(&mut n, &all, &c)?;
                Ok(dot(n.forward(input(&n))?.head_input, up.clone()))
            })?;
            ensure(g.parameters.len() == count, "trunk gradient count")?;
            for (a, e) in g.parameters.into_iter().zip(numerical) {
                compare(&data(a), &e)?;
            }
            replace(&mut n, &original, &c)?;
            let x = vec![data(input(&n))];
            let numerical_input = central(&x, |p| {
                Ok(dot(
                    n.forward(tensor(&p[0], 2, n.input_size))?.head_input,
                    up.clone(),
                ))
            })?;
            compare(&data(g.input), &numerical_input[0])?;
            Ok(())
        }
        "heads" => {
            let (mut n, c) = fixture()?;
            let cache = n.forward(input(&n))?;
            let mu = tensor(&[0.2, -0.3, 0.1, 0.4], 2, 2);
            let value = tensor(&[0.5, -0.2], 2, 1);
            let std = tensor(&[0.3, -0.1], 1, 2);
            let g = manual::shared_heads_backward(
                &n,
                cache,
                OutputGradients {
                    mean: mu.clone(),
                    value: value.clone(),
                    log_std: std.clone(),
                },
            )?;
            let p = snapshot(&n);
            let numerical = central(&p, |p| {
                replace(&mut n, p, &c)?;
                let out = n.infer(input(&n))?;
                Ok(dot(out.mean, mu.clone())
                    + dot(out.value, value.clone())
                    + dot(n.log_std.clone(), std.clone()))
            })?;
            ensure(g.len() == p.len(), "heads gradient count")?;
            for (a, e) in g.into_iter().zip(numerical) {
                compare(&data(a), &e)?;
            }
            Ok(())
        }
        "log_std" | "ppo" => {
            let (mut n, mut c) = fixture()?;
            c.latent_entropy_coef = 0.017;
            let (cache, l) = loss(&n, &c)?;
            let outputs = manual::ppo_output_gradients(&l)?;
            if case == "log_std" {
                let p = vec![data(n.log_std.clone())];
                let g = data(outputs.log_std);
                let numerical = central(&p, |p| {
                    n.log_std = tensor(&p[0], 1, 2);
                    Ok(scalar(loss(&n, &c)?.1.total))
                })?;
                compare(&g, &numerical[0])
            } else {
                let g = manual::shared_heads_backward(&n, cache, outputs)?;
                let p = snapshot(&n);
                let numerical = central(&p, |p| {
                    replace(&mut n, p, &c)?;
                    Ok(scalar(loss(&n, &c)?.1.total))
                })?;
                ensure(g.len() == p.len(), "PPO gradient count")?;
                for (a, e) in g.into_iter().zip(numerical) {
                    compare(&data(a), &e)?;
                }
                Ok(())
            }
        }
        "synthetic" => synthetic(),
        _ => unreachable!(),
    }
}
/// A small frozen PPO objective must decrease using the real manual hooks and optimizer.
fn synthetic() -> Result<()> {
    let (mut n, mut c) = fixture()?;
    c.target_kl = 100.;
    c.learning_rate = 0.001;
    let before = scalar(loss(&n, &c)?.1.total);
    let mut opt = crate::optimizer::Optimizer::new(c.optimizer.clone(), &n.parameters());
    for _ in 0..10 {
        let (cache, l) = loss(&n, &c)?;
        let out = manual::ppo_output_gradients(&l)?;
        let g = manual::shared_heads_backward(&n, cache, out)?;
        let (g, _) = crate::optimizer::clip_global_norm(g, 1.)?;
        let p = opt.update(&n.parameters(), g, c.learning_rate)?;
        n.replace(p, c.log_std_bounds)?;
    }
    ensure(
        scalar(loss(&n, &c)?.1.total) < before,
        "synthetic objective did not decrease",
    )
}
