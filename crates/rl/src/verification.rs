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
    "mu",
    "value",
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
            let h = DIFFERENCE_STEP;
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
// Only the lightweight preflight and synthetic optimization smoke use this small
// objective. Numerical PPO checks use the frozen sign/ratio grid below instead.
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

const DIFFERENCE_STEP: f64 = 1e-5;
const PPO_BATCH_SIZES: [usize; 2] = [9, 18];
const ENTROPY_COEFFICIENTS: [f32; 2] = [0., 0.017];
const REGION_MARGIN: f64 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RatioRegion {
    Below,
    Inside,
    Above,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PolicyBranch {
    Unclipped,
    Clipped,
    ZeroAdvantage,
}
#[derive(Clone, Copy, Debug)]
struct ExpectedRow {
    advantage: f64,
    ratio: f64,
    region: RatioRegion,
    branch: PolicyBranch,
}

/// Immutable historical data, created ONCE from the unperturbed network.
/// Numerical objectives receive this same fixture; they never calibrate old_log_prob again.
struct FrozenPpoFixture {
    observations: Tensor<R, 2>,
    u: Tensor<R, 2>,
    old_log_prob: Tensor<R, 2>,
    advantages: Tensor<R, 2>,
    targets: Tensor<R, 2>,
    expected: Vec<ExpectedRow>,
    config: Config,
}
impl FrozenPpoFixture {
    fn new(network: &Network<R>, mut config: Config, batch: usize, entropy: f32) -> Result<Self> {
        ensure(
            batch > 0 && batch.is_multiple_of(9),
            "PPO fixture requires complete nine-row grids",
        )?;
        config.minibatch_size = batch;
        config.latent_entropy_coef = entropy;
        let d = network.input_size;
        let observations = tensor(
            &(0..batch * d)
                .map(|i| 0.1 + ((i / d * 3 + i % d) % 11) as f64 * 0.015)
                .collect::<Vec<_>>(),
            batch,
            d,
        );
        let u = tensor(
            &(0..batch)
                .flat_map(|i| [0.21 + 0.027 * i as f64, -0.46 + 0.019 * i as f64])
                .collect::<Vec<_>>(),
            batch,
            2,
        );
        // All nine sign/region combinations, including the two branches where PPO
        // actually limits the objective. Zero advantage gives a constant policy term.
        let expected: Vec<_> = (0..batch)
            .map(|i| {
                let repeat = (i / 9) as f64;
                let advantage = [0.7 + 0.2 * repeat, -0.4 - 0.15 * repeat, 0.][i % 9 / 3];
                let ratio = [0.55, 1.05, 1.55][i % 3];
                let region = [RatioRegion::Below, RatioRegion::Inside, RatioRegion::Above][i % 3];
                let branch = match (i % 9 / 3, region) {
                    (0, RatioRegion::Above) | (1, RatioRegion::Below) => PolicyBranch::Clipped,
                    (2, _) => PolicyBranch::ZeroAdvantage,
                    _ => PolicyBranch::Unclipped,
                };
                ExpectedRow {
                    advantage,
                    ratio,
                    region,
                    branch,
                }
            })
            .collect();
        let initial = network.forward(observations.clone())?;
        validate_relu_region(&initial)?;
        let initial_log_prob = data(
            distribution::evaluate(initial.mean, initial.log_std, u.clone())?.joint_log_density,
        );
        // This is a FORWARD fixture calibration, not a gradient formula. These values
        // are stored and cloned unchanged for every plus/minus numerical objective.
        let old_log_prob = tensor(
            &initial_log_prob
                .iter()
                .zip(&expected)
                .map(|(log, row)| log - row.ratio.ln())
                .collect::<Vec<_>>(),
            batch,
            1,
        );
        let advantages = tensor(
            &expected.iter().map(|row| row.advantage).collect::<Vec<_>>(),
            batch,
            1,
        );
        let targets = tensor(
            &(0..batch)
                .map(|i| -0.8 + 0.13 * (i % 7) as f64)
                .collect::<Vec<_>>(),
            batch,
            1,
        );
        let fixture = Self {
            observations,
            u,
            old_log_prob,
            advantages,
            targets,
            expected,
            config,
        };
        let (_, cache) = fixture.network_loss(network)?;
        fixture.validate_initial(&cache)?;
        Ok(fixture)
    }

    fn loss(
        &self,
        mean: Tensor<R, 2>,
        value: Tensor<R, 2>,
        log_std: Tensor<R, 2>,
    ) -> Result<crate::ppo::LossCache<R>> {
        let distribution = distribution::evaluate(mean, log_std, self.u.clone())?;
        let inputs = LossInputs {
            old_log_prob: self.old_log_prob.clone(),
            advantage: self.advantages.clone(),
            value_target: self.targets.clone(),
        };
        let (cache, _) = ppo::forward(distribution, value, inputs, &self.config)?;
        self.validate_regions(&cache)?;
        Ok(cache)
    }

    fn network_loss(
        &self,
        network: &Network<R>,
    ) -> Result<(crate::network::ForwardCache<R>, crate::ppo::LossCache<R>)> {
        let forward = network.forward(self.observations.clone())?;
        validate_relu_region(&forward)?;
        let loss = self.loss(
            forward.mean.clone(),
            forward.value.clone(),
            forward.log_std.clone(),
        )?;
        Ok((forward, loss))
    }

    fn validate_initial(&self, cache: &crate::ppo::LossCache<R>) -> Result<()> {
        self.validate_regions(cache)?;
        for (i, (actual, expected)) in data(cache.ratio.clone())
            .iter()
            .zip(&self.expected)
            .enumerate()
        {
            ensure(
                (actual - expected.ratio).abs() < 1e-12,
                format!(
                    "PPO fixture row {i}: ratio {actual}, expected {}",
                    expected.ratio
                ),
            )?;
        }
        Ok(())
    }

    /// Check actual forward values/branch selection, without providing any derivatives.
    /// Also runs on every perturbed objective so the finite-difference stencil cannot
    /// cross a clip/min boundary silently.
    fn validate_regions(&self, cache: &crate::ppo::LossCache<R>) -> Result<()> {
        ensure(
            cache.batch_size == self.expected.len(),
            "PPO fixture minibatch size",
        )?;
        ensure(
            data(cache.inputs.old_log_prob.clone()) == data(self.old_log_prob.clone()),
            "PPO fixture old_log_prob changed",
        )?;
        ensure(
            data(cache.distribution.u.clone()) == data(self.u.clone()),
            "PPO fixture historical u changed",
        )?;
        ensure(
            data(cache.inputs.value_target.clone()) == data(self.targets.clone()),
            "PPO fixture value targets changed",
        )?;
        let ratios = data(cache.ratio.clone());
        let clipped = data(cache.clipped_ratio.clone());
        let advantages = data(cache.inputs.advantage.clone());
        let low = (1. - self.config.clip_epsilon) as f64;
        let high = (1. + self.config.clip_epsilon) as f64;
        for (i, expected) in self.expected.iter().enumerate() {
            let ratio = ratios[i];
            let in_region = match expected.region {
                RatioRegion::Below => ratio < low - REGION_MARGIN,
                RatioRegion::Inside => ratio > low + REGION_MARGIN && ratio < high - REGION_MARGIN,
                RatioRegion::Above => ratio > high + REGION_MARGIN,
            };
            ensure(
                ratio.is_finite() && in_region,
                format!(
                    "PPO fixture row {i}: ratio {ratio} outside intended {:?} region",
                    expected.region
                ),
            )?;
            ensure(
                advantages[i] == expected.advantage,
                format!("PPO fixture row {i}: advantage changed"),
            )?;
            ensure(
                (clipped[i] - ratio.clamp(low, high)).abs() < 1e-12,
                format!("PPO fixture row {i}: clipped ratio"),
            )?;
            let raw_objective = ratio * advantages[i];
            let clipped_objective = clipped[i] * advantages[i];
            let branch_valid = match expected.branch {
                PolicyBranch::Clipped => clipped_objective < raw_objective,
                PolicyBranch::Unclipped => raw_objective <= clipped_objective,
                PolicyBranch::ZeroAdvantage => raw_objective == 0. && clipped_objective == 0.,
            };
            ensure(
                branch_valid,
                format!(
                    "PPO fixture row {i}: intended {:?} objective branch not selected",
                    expected.branch
                ),
            )?;
        }
        Ok(())
    }
}

fn validate_relu_region(cache: &crate::network::ForwardCache<R>) -> Result<()> {
    for layer in &cache.trunk {
        ensure(
            data(layer.preactivation.clone())
                .iter()
                .all(|z| z.is_finite() && *z > 0.05),
            "gradient fixture crossed the positive ReLU region",
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum PpoOutput {
    Mean,
    Value,
    LogStd,
}

/// Perturb one output group while freezing the other two and all historical data.
fn output_differences(
    fixture: &FrozenPpoFixture,
    forward: &crate::network::ForwardCache<R>,
    output: PpoOutput,
) -> Result<Vec<f64>> {
    let (values, rows, columns) = match output {
        PpoOutput::Mean => (data(forward.mean.clone()), fixture.expected.len(), 2),
        PpoOutput::Value => (data(forward.value.clone()), fixture.expected.len(), 1),
        PpoOutput::LogStd => (data(forward.log_std.clone()), 1, 2),
    };
    let numerical = central(&[values], |p| {
        let mut mean = forward.mean.clone();
        let mut value = forward.value.clone();
        let mut log_std = forward.log_std.clone();
        match output {
            PpoOutput::Mean => mean = tensor(&p[0], rows, columns),
            PpoOutput::Value => value = tensor(&p[0], rows, columns),
            PpoOutput::LogStd => log_std = tensor(&p[0], rows, columns),
        }
        Ok(scalar(fixture.loss(mean, value, log_std)?.total))
    })?;
    Ok(numerical.into_iter().next().unwrap())
}

fn network_differences(
    network: &mut Network<R>,
    fixture: &FrozenPpoFixture,
) -> Result<Vec<Vec<f64>>> {
    let parameters = snapshot(network);
    let numerical = central(&parameters, |p| {
        replace(network, p, &fixture.config)?;
        Ok(scalar(fixture.network_loss(network)?.1.total))
    });
    replace(network, &parameters, &fixture.config)?;
    numerical
}

fn check_ppo(case: &str) -> Result<()> {
    for batch in PPO_BATCH_SIZES {
        for entropy in ENTROPY_COEFFICIENTS {
            let (mut network, config) = fixture()?;
            let fixture = FrozenPpoFixture::new(&network, config.clone(), batch, entropy)?;
            let (cache, loss) = fixture.network_loss(&network)?;
            fixture.validate_initial(&loss)?;
            let outputs = manual::ppo_output_gradients(&loss)?;
            ensure(
                outputs.mean.dims() == [batch, 2]
                    && outputs.value.dims() == [batch, 1]
                    && outputs.log_std.dims() == [1, 2],
                "manual PPO output gradient shapes",
            )?;
            match case {
                "mu" => compare(
                    &data(outputs.mean),
                    &output_differences(&fixture, &cache, PpoOutput::Mean)?,
                )?,
                "value" => compare(
                    &data(outputs.value),
                    &output_differences(&fixture, &cache, PpoOutput::Value)?,
                )?,
                "log_std" => compare(
                    &data(outputs.log_std),
                    &output_differences(&fixture, &cache, PpoOutput::LogStd)?,
                )?,
                "ppo" => {
                    let gradients = manual::shared_heads_backward(&network, cache, outputs)?;
                    crate::optimizer::validate_gradients(&network.parameters(), &gradients)?;
                    let numerical = network_differences(&mut network, &fixture)?;
                    for (actual, expected) in gradients.into_iter().zip(numerical) {
                        compare(&data(actual), &expected)?;
                    }
                }
                _ => unreachable!(),
            }
        }
    }
    Ok(())
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
            for (batch, input, output) in [(2, 3, 4), (5, 4, 2)] {
                let p = vec![
                    (0..output * input)
                        .map(|i| -0.3 + 0.017 * i as f64)
                        .collect(),
                    (0..output).map(|i| 0.1 - 0.04 * i as f64).collect(),
                    (0..batch * input)
                        .map(|i| -0.4 + 0.03 * (i % 13) as f64)
                        .collect(),
                ];
                let make = |p: &[Vec<f64>]| Dense {
                    weights: tensor(&p[0], output, input),
                    bias: tensor(&p[1], 1, output),
                };
                let upstream = tensor(
                    &(0..batch * output)
                        .map(|i| 0.07 * ((i * 7 % 17) as f64 - 8.))
                        .collect::<Vec<_>>(),
                    batch,
                    output,
                );
                let g = manual::dense_backward(
                    &make(&p),
                    tensor(&p[2], batch, input),
                    upstream.clone(),
                )?;
                ensure(
                    g.weights.dims() == [output, input]
                        && g.bias.dims() == [1, output]
                        && g.input.dims() == [batch, input],
                    "manual dense gradient shapes",
                )?;
                let numerical = central(&p, |p| {
                    Ok(dot(
                        make(p).forward(tensor(&p[2], batch, input)),
                        upstream.clone(),
                    ))
                })?;
                for (a, e) in [g.weights, g.bias, g.input].into_iter().zip(numerical) {
                    compare(&data(a), &e)?;
                }
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
        "mu" | "value" | "log_std" | "ppo" => check_ppo(case),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppo_fixture_grid_covers_signs_regions_and_limited_objective_branches() {
        for batch in PPO_BATCH_SIZES {
            for entropy in ENTROPY_COEFFICIENTS {
                let (network, config) = fixture().unwrap();
                let fixture = FrozenPpoFixture::new(&network, config, batch, entropy).unwrap();
                let (_, cache) = fixture.network_loss(&network).unwrap();
                fixture.validate_initial(&cache).unwrap();
                assert_eq!(cache.batch_size, batch);
                assert_eq!(cache.latent_entropy_coef, entropy);
                assert_eq!(data(cache.ratio.clone()).len(), batch);
                for grid in fixture.expected.chunks_exact(9) {
                    for (i, row) in grid.iter().enumerate() {
                        assert_eq!(
                            row.region,
                            [RatioRegion::Below, RatioRegion::Inside, RatioRegion::Above][i % 3]
                        );
                        assert_eq!(
                            row.advantage.total_cmp(&0.),
                            [
                                std::cmp::Ordering::Greater,
                                std::cmp::Ordering::Less,
                                std::cmp::Ordering::Equal
                            ][i / 3]
                        );
                    }
                    assert_eq!(
                        grid.iter().map(|row| row.branch).collect::<Vec<_>>(),
                        vec![
                            PolicyBranch::Unclipped,
                            PolicyBranch::Unclipped,
                            PolicyBranch::Clipped,
                            PolicyBranch::Clipped,
                            PolicyBranch::Unclipped,
                            PolicyBranch::Unclipped,
                            PolicyBranch::ZeroAdvantage,
                            PolicyBranch::ZeroAdvantage,
                            PolicyBranch::ZeroAdvantage,
                        ]
                    );
                }
            }
        }
    }

    #[test]
    fn frozen_ppo_objectives_stay_in_region_for_all_output_and_network_perturbations() {
        // This verifies the numeric TEST harness, not manual backprop. No analytic
        // gradient is supplied or compared while the derivative hooks are pending.
        for batch in PPO_BATCH_SIZES {
            for entropy in ENTROPY_COEFFICIENTS {
                let (mut network, config) = fixture().unwrap();
                let fixture = FrozenPpoFixture::new(&network, config, batch, entropy).unwrap();
                let before = snapshot(&network);
                let old_log_prob = data(fixture.old_log_prob.clone());
                let observations = data(fixture.observations.clone());
                let u = data(fixture.u.clone());
                let targets = data(fixture.targets.clone());
                let advantages = data(fixture.advantages.clone());
                let (cache, _) = fixture.network_loss(&network).unwrap();
                for output in [PpoOutput::Mean, PpoOutput::Value, PpoOutput::LogStd] {
                    let numerical = output_differences(&fixture, &cache, output).unwrap();
                    let length = match output {
                        PpoOutput::Mean => batch * 2,
                        PpoOutput::Value => batch,
                        PpoOutput::LogStd => 2,
                    };
                    assert_eq!(numerical.len(), length);
                    assert!(numerical.iter().all(|x| x.is_finite()));
                }
                let numerical = network_differences(&mut network, &fixture).unwrap();
                assert_eq!(
                    numerical.iter().map(Vec::len).collect::<Vec<_>>(),
                    before.iter().map(Vec::len).collect::<Vec<_>>()
                );
                assert!(numerical.iter().flatten().all(|x| x.is_finite()));
                assert_eq!(snapshot(&network), before);
                assert_eq!(data(fixture.old_log_prob.clone()), old_log_prob);
                assert_eq!(data(fixture.observations.clone()), observations);
                assert_eq!(data(fixture.u.clone()), u);
                assert_eq!(data(fixture.targets.clone()), targets);
                assert_eq!(data(fixture.advantages.clone()), advantages);
            }
        }
    }

    #[test]
    fn fixture_guard_rejects_wrong_region_changed_history_and_clip_boundary() {
        let (network, config) = fixture().unwrap();
        let fixture = FrozenPpoFixture::new(&network, config, 9, 0.).unwrap();
        let (_, mut cache) = fixture.network_loss(&network).unwrap();
        let correct = cache.ratio.clone();
        cache.ratio = tensor(&[1.05; 9], 9, 1);
        assert!(fixture.validate_regions(&cache).is_err());
        cache.ratio = correct;
        cache.inputs.old_log_prob = cache.inputs.old_log_prob.clone() + 0.1;
        assert!(fixture.validate_regions(&cache).is_err());

        let (_, mut cache) = fixture.network_loss(&network).unwrap();
        let mut ratios = data(cache.ratio.clone());
        ratios[0] = (1. - fixture.config.clip_epsilon) as f64 - DIFFERENCE_STEP;
        cache.ratio = tensor(&ratios, 9, 1);
        assert!(fixture.validate_regions(&cache).is_err());

        let (_, mut cache) = fixture.network_loss(&network).unwrap();
        cache.inputs.advantage = cache.inputs.advantage.clone() + 0.1;
        assert!(fixture.validate_regions(&cache).is_err());
    }
}
