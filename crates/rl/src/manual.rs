//! Backpropagation explícito: Burn executa os tensores, sem autodiferenciação.
//! Somente a origem da loss aplica médias; nenhuma função atualiza os parâmetros.
use crate::{
    Error, Result,
    backend::{Backend, Tensor},
    ensure,
    network::{Dense, ForwardCache, LayerCache, Network},
    ppo::LossCache,
};

pub const FUNCTIONS: &[&str] = &[
    "activation_backward",
    "dense_backward",
    "trunk_backward",
    "shared_heads_backward",
    "ppo_output_gradients",
];
pub fn missing(function: &'static str) -> Error {
    Error::MissingManualImplementation { function }
}
/// Call real hooks on a tiny frozen case, never a success flag or dummy gradient.
pub fn preflight() -> Result<()> {
    crate::verification::probe_hooks()
}

#[derive(Clone, Copy, Debug)]
pub enum Activation {
    Relu,
    Linear,
    Tanh,
}
// 1. Ativações ---------------------------------------------------------------
/// Recebe z, ativação e grad_output [B,H]; regra da cadeia: dL/dz = dL/da * f'(z).
/// Retorna [B,H], sem redução. Para ReLU em zero escolhemos derivada zero.
pub fn activation_backward<B: Backend>(
    kind: Activation,
    z: Tensor<B, 2>,
    activation: Tensor<B, 2>,
    grad_output: Tensor<B, 2>,
) -> Result<Tensor<B, 2>> {
    ensure(
        z.dims() == activation.dims() && z.dims() == grad_output.dims(),
        "activation backward shapes [B,H]",
    )?;
    Ok(match kind {
        Activation::Linear => grad_output, // f'(z) = 1
        Activation::Relu => grad_output * z.greater_elem(0.).float(),
        Activation::Tanh => grad_output * (activation.powf_scalar(2.).neg() + 1.),
    })
}
pub struct DenseGradients<B: Backend> {
    pub input: Tensor<B, 2>,
    pub weights: Tensor<B, 2>,
    pub bias: Tensor<B, 2>,
}
// 2. Camada densa -----------------------------------------------------------
/// Recebe X[B,I], W[O,I], grad_output[B,O] para Y = X Wᵀ + b.
/// Retorna dX[B,I], dW[O,I], db[1,O]; a média da loss já está em grad_output.
pub fn dense_backward<B: Backend>(
    layer: &Dense<B>,
    input: Tensor<B, 2>,
    grad_output: Tensor<B, 2>,
) -> Result<DenseGradients<B>> {
    let [batch, inputs] = input.dims();
    let [outputs, weight_inputs] = layer.weights.dims();
    ensure(
        batch > 0
            && inputs == weight_inputs
            && layer.bias.dims() == [1, outputs]
            && grad_output.dims() == [batch, outputs],
        "dense backward shapes X[B,I], W[O,I], b[1,O], grad_output[B,O]",
    )?;
    let grad_input = grad_output.clone().matmul(layer.weights.clone()); // dX = dY W
    let grad_weights = grad_output.clone().transpose().matmul(input); // dW = dYᵀ X
    let grad_bias = grad_output.sum_dim(0); // db = soma das linhas de dY
    Ok(DenseGradients {
        input: grad_input,
        weights: grad_weights,
        bias: grad_bias,
    })
}
pub struct TrunkGradients<B: Backend> {
    pub input: Tensor<B, 2>,
    pub parameters: Vec<Tensor<B, 2>>,
}
// 3. Tronco compartilhado ---------------------------------------------------
/// Recebe camadas/caches em ordem forward e dL/dH_final [B,H_final].
/// Propaga dL/dH_anterior pela ReLU e pela densa; retorna dX[B,D] e W,b em ordem forward.
pub fn trunk_backward<B: Backend>(
    layers: &[Dense<B>],
    caches: &[LayerCache<B>],
    grad_output: Tensor<B, 2>,
) -> Result<TrunkGradients<B>> {
    ensure(layers.len() == caches.len(), "trunk backward cache count")?;
    let mut grad_input = grad_output;
    let mut parameters = Vec::with_capacity(2 * layers.len());
    for (layer, cache) in layers.iter().zip(caches).rev() {
        let grad_preactivation = activation_backward(
            Activation::Relu,
            cache.preactivation.clone(),
            cache.activation.clone(),
            grad_input,
        )?;
        let gradients = dense_backward(layer, cache.input.clone(), grad_preactivation)?;
        grad_input = gradients.input;
        // Percorremos de trás para frente: guardamos b,W e invertemos ao terminar.
        parameters.push(gradients.bias);
        parameters.push(gradients.weights);
    }
    parameters.reverse();
    Ok(TrunkGradients {
        input: grad_input,
        parameters,
    })
}
pub struct OutputGradients<B: Backend> {
    pub mean: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
    pub log_std: Tensor<B, 2>,
}
// 4. Loss PPO e distribuição ------------------------------------------------
/// Recebe o cache da loss de minimização, com u, old_log_prob, vantagens e alvos fixos.
/// Deriva -mean(min(rA,clip(r)A)) + c_v/2 mean((V-alvo)²) - c_H H_latente.
/// Retorna dmu[B,2], dV[B,1], dlog_std[1,2]; aplica a média exatamente aqui.
pub fn ppo_output_gradients<B: Backend>(cache: &LossCache<B>) -> Result<OutputGradients<B>> {
    let batch = cache.batch_size;
    ensure(
        batch > 0
            && cache.distribution.mean.dims() == [batch, 2]
            && cache.distribution.u.dims() == [batch, 2]
            && cache.distribution.log_std.dims() == [1, 2]
            && cache.value.dims() == [batch, 1]
            && cache.inputs.value_target.dims() == [batch, 1]
            && cache.inputs.advantage.dims() == [batch, 1]
            && cache.inputs.old_log_prob.dims() == [batch, 1]
            && cache.ratio.dims() == [batch, 1]
            && cache.clipped_ratio.dims() == [batch, 1],
        "PPO backward shapes mu/u[B,2], log_std[1,2], scalar fields[B,1]",
    )?;
    let advantage = cache.inputs.advantage.clone();
    let ratio = cache.ratio.clone();
    let objective = ratio.clone() * advantage.clone();
    let clipped_objective = cache.clipped_ratio.clone() * advantage.clone();
    // O mínimo limita A>0 acima de 1+eps e A<0 abaixo de 1-eps; A=0 dá zero.
    // Nesse ramo o clip é constante. No empate (inclusive nas fronteiras), usamos
    // o ramo sem clip, como no forward. Os checks evitam pontos não diferenciáveis.
    let clipped_branch = objective.greater(clipped_objective);
    // dr/d(log_density) = r; old_log_prob não recebe gradiente.
    let grad_log_density = (advantage * ratio * (-1. / batch as f64)).mask_fill(clipped_branch, 0.);

    let residual = cache.distribution.u.clone() - cache.distribution.mean.clone();
    let inverse_variance = (cache.distribution.log_std.clone() * -2.).exp();
    // Com u fixo, o Jacobiano da Tanh é constante em relação a mu e log_std.
    // d log_density/dmu = (u-mu)/sigma².
    let grad_mean = grad_log_density.clone() * residual.clone() * inverse_variance.clone();
    // d log_density/dlog_std = (u-mu)²/sigma² - 1; soma porque log_std é global.
    let grad_log_std = (grad_log_density
        * (residual.powf_scalar(2.) * inverse_variance - 1.))
        .sum_dim(0)
        // H_latente = soma(log_std + constante): -c_H por dimensão, sem outro 1/B.
        - cache.latent_entropy_coef;
    // O 2 do quadrado cancela o 1/2 da value loss; a média entra uma única vez.
    let grad_value = (cache.value.clone() - cache.inputs.value_target.clone())
        * (cache.value_coef as f64 / batch as f64);
    Ok(OutputGradients {
        mean: grad_mean,
        value: grad_value,
        log_std: grad_log_std,
    })
}
// 5. Cabeças e combinação dos ramos -----------------------------------------
/// Recebe o cache consumido do forward e dmu[B,2], dV[B,1], dlog_std[1,2].
/// dH = dH_actor + dH_critic; retorna todos os gradientes como Network::parameters().
/// Nenhuma atualização ocorre aqui; o cache deve pertencer aos pesos atuais.
pub fn shared_heads_backward<B: Backend>(
    network: &Network<B>,
    cache: ForwardCache<B>,
    outputs: OutputGradients<B>,
) -> Result<Vec<Tensor<B, 2>>> {
    network.validate_cache(&cache)?;
    ensure(
        outputs.log_std.dims() == network.log_std.dims(),
        "shared heads backward log_std shape [1,2]",
    )?;
    let actor = dense_backward(&network.actor, cache.head_input.clone(), outputs.mean)?;
    let critic = dense_backward(&network.critic, cache.head_input, outputs.value)?;
    let trunk = trunk_backward(&network.trunk, &cache.trunk, actor.input + critic.input)?;
    let mut parameters = trunk.parameters;
    parameters.extend([
        actor.weights,
        actor.bias,
        critic.weights,
        critic.bias,
        outputs.log_std,
    ]);
    Ok(parameters)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Cpu, matrix, read};

    #[test]
    fn completed_hooks_and_relu_zero_convention() {
        // Uma regressão para stub agora deve falhar, mesmo na suíte de infraestrutura.
        preflight().unwrap();
        let z = matrix::<Cpu>(&[-1., 0., 1.], 1, 3, &Default::default()).unwrap();
        let activation = crate::backend::activation::relu(z.clone());
        let grad_output = matrix(&[2., 3., 4.], 1, 3, &Default::default()).unwrap();
        let gradient = activation_backward(Activation::Relu, z, activation, grad_output).unwrap();
        assert_eq!(read(gradient).unwrap(), vec![0., 0., 4.]);
    }
}
