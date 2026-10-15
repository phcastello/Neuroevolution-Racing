use crate::{
    Result,
    backend::{Backend, Tensor, matrix, read, validate},
    config::OptimizerKind,
    ensure,
};
use serde::{Deserialize, Serialize};
pub struct Optimizer<B: Backend> {
    pub kind: OptimizerKind,
    pub step: u64,
    pub first: Vec<Tensor<B, 2>>,
    pub second: Vec<Tensor<B, 2>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizerState {
    pub kind: OptimizerKind,
    pub step: u64,
    pub first: Vec<Vec<f32>>,
    pub second: Vec<Vec<f32>>,
}
impl<B: Backend> Optimizer<B> {
    pub fn new(kind: OptimizerKind, params: &[Tensor<B, 2>]) -> Self {
        Self {
            kind,
            step: 0,
            first: params
                .iter()
                .map(|p| Tensor::zeros(p.shape(), &p.device()))
                .collect(),
            second: params
                .iter()
                .map(|p| Tensor::zeros(p.shape(), &p.device()))
                .collect(),
        }
    }
    pub fn update(
        &mut self,
        params: &[Tensor<B, 2>],
        grads: Vec<Tensor<B, 2>>,
        lr: f32,
    ) -> Result<Vec<Tensor<B, 2>>> {
        validate_gradients(params, &grads)?;
        ensure(lr.is_finite() && lr > 0., "learning rate")?;
        ensure(
            params.len() == self.first.len()
                && params.len() == self.second.len()
                && params
                    .iter()
                    .zip(&self.first)
                    .zip(&self.second)
                    .all(|((p, m), v)| p.dims() == m.dims() && p.dims() == v.dims()),
            "optimizer state shapes",
        )?;
        let step = self
            .step
            .checked_add(1)
            .ok_or_else(|| crate::Error::Invalid("optimizer step overflow".into()))?;
        let mut updated = vec![];
        let mut first = vec![];
        let mut second = vec![];
        for (i, (p, g)) in params.iter().zip(grads).enumerate() {
            let (next, m, v) = match self.kind {
                OptimizerKind::Sgd => (
                    p.clone() - g * lr,
                    self.first[i].clone(),
                    self.second[i].clone(),
                ),
                OptimizerKind::Adam {
                    beta1,
                    beta2,
                    epsilon,
                } => {
                    ensure(
                        (0.0..1.0).contains(&beta1)
                            && (0.0..1.0).contains(&beta2)
                            && epsilon.is_finite()
                            && epsilon > 0.,
                        "Adam coefficients",
                    )?;
                    let m = self.first[i].clone() * beta1 + g.clone() * (1. - beta1);
                    let v = self.second[i].clone() * beta2 + g.powf_scalar(2.) * (1. - beta2);
                    let mhat = m.clone() / (1. - (beta1 as f64).powf(step as f64));
                    let vhat = v.clone() / (1. - (beta2 as f64).powf(step as f64));
                    (p.clone() - mhat / (vhat.sqrt() + epsilon) * lr, m, v)
                }
            };
            validate(&next, "optimizer output")?;
            first.push(m);
            second.push(v);
            updated.push(next);
        }
        self.first = first;
        self.second = second;
        self.step = step;
        Ok(updated)
    }
    pub fn snapshot(&self) -> Result<OptimizerState> {
        Ok(OptimizerState {
            kind: self.kind.clone(),
            step: self.step,
            first: self
                .first
                .iter()
                .cloned()
                .map(read)
                .collect::<Result<_>>()?,
            second: self
                .second
                .iter()
                .cloned()
                .map(read)
                .collect::<Result<_>>()?,
        })
    }
    pub fn restore(state: &OptimizerState, params: &[Tensor<B, 2>]) -> Result<Self> {
        ensure(
            state.first.len() == params.len() && state.second.len() == params.len(),
            "optimizer state count",
        )?;
        let load = |values: &[Vec<f32>]| -> Result<Vec<Tensor<B, 2>>> {
            values
                .iter()
                .zip(params)
                .map(|(v, p)| {
                    let [r, c] = p.dims();
                    matrix(v, r, c, &p.device())
                })
                .collect()
        };
        Ok(Self {
            kind: state.kind.clone(),
            step: state.step,
            first: load(&state.first)?,
            second: load(&state.second)?,
        })
    }
}
pub fn validate_gradients<B: Backend>(
    params: &[Tensor<B, 2>],
    grads: &[Tensor<B, 2>],
) -> Result<()> {
    ensure(
        params.len() == grads.len() && params.iter().zip(grads).all(|(p, g)| p.dims() == g.dims()),
        "gradient shapes/count",
    )?;
    for g in grads {
        validate(g, "gradient")?;
    }
    Ok(())
}
/// One global norm across weights, biases AND log_std, not one norm per tensor.
pub fn clip_global_norm<B: Backend>(
    grads: Vec<Tensor<B, 2>>,
    maximum: f32,
) -> Result<(Vec<Tensor<B, 2>>, f32)> {
    ensure(
        maximum.is_finite() && maximum > 0. && !grads.is_empty(),
        "gradient norm configuration",
    )?;
    let mut sum = Tensor::<B, 1>::zeros([1], &grads[0].device());
    for g in &grads {
        validate(g, "gradient clipping")?;
        sum = sum + g.clone().powf_scalar(2.).sum();
    }
    let norm = read(sum.sqrt())?[0];
    let scale = if norm > maximum { maximum / norm } else { 1. };
    Ok((grads.into_iter().map(|g| g * scale).collect(), norm))
}
