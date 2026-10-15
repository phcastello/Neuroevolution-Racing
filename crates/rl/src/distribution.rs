//! Forward only: joint log-density of a diagonal Gaussian transformed with tanh.
use crate::{
    Result,
    backend::{Backend, Tensor, matrix},
    ensure,
};
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, StandardNormal};
pub struct DistributionCache<B: Backend> {
    pub mean: Tensor<B, 2>,
    pub log_std: Tensor<B, 2>,
    pub u: Tensor<B, 2>,
    pub action: Tensor<B, 2>,
    pub joint_log_density: Tensor<B, 2>,
    pub latent_entropy: Tensor<B, 2>,
}
pub fn evaluate<B: Backend>(
    mean: Tensor<B, 2>,
    log_std: Tensor<B, 2>,
    u: Tensor<B, 2>,
) -> Result<DistributionCache<B>> {
    ensure(
        mean.dims() == u.dims() && mean.dims()[1] == 2 && log_std.dims() == [1, 2],
        "distribution shapes [B,2],[1,2],[B,2]",
    )?;
    let standardized = (u.clone() - mean.clone()) / log_std.clone().exp();
    let log_normal = standardized.powf_scalar(2.) * (-0.5)
        - log_std.clone()
        - 0.5 * (2. * std::f64::consts::PI).ln();
    // log(1-tanh(u)^2) = 2*(ln(2)-u-softplus(-2u)). Stable even when tanh rounds to +/-1.
    let x = u.clone() * (-2.);
    let softplus = x.clone().clamp_min(0.) + ((-x.abs()).exp() + 1.).log();
    let correction = (u.clone().neg() + std::f64::consts::LN_2 - softplus) * 2.;
    let joint_log_density = (log_normal - correction).sum_dim(1);
    let latent_entropy =
        (log_std.clone() + 0.5 * (2. * std::f64::consts::PI * std::f64::consts::E).ln()).sum_dim(1);
    Ok(DistributionCache {
        mean,
        log_std,
        action: u.clone().tanh(),
        u,
        joint_log_density,
        latent_entropy,
    })
}
pub fn sample<B: Backend>(
    mean: Tensor<B, 2>,
    log_std: Tensor<B, 2>,
    rng: &mut ChaCha12Rng,
) -> Result<DistributionCache<B>> {
    let [b, d] = mean.dims();
    let eps: Vec<f32> = (0..b * d).map(|_| StandardNormal.sample(rng)).collect();
    let noise = matrix(&eps, b, d, &mean.device())?;
    let u = mean.clone() + log_std.clone().exp() * noise;
    evaluate(mean, log_std, u)
}
