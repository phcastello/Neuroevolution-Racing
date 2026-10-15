use crate::{
    Result,
    backend::{Backend, Tensor, activation, matrix, read},
    config::Config,
    ensure,
};
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use rand_distr::{Distribution, StandardNormal};

/// All parameter tensors are rank 2: W[out,in], bias[1,out], log_std[1,2].
/// Flat order: trunk layer W,b ... actor W,b, critic W,b, log_std.
#[derive(Clone)]
pub struct Dense<B: Backend> {
    pub weights: Tensor<B, 2>,
    pub bias: Tensor<B, 2>,
}
impl<B: Backend> Dense<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        x.matmul(self.weights.clone().transpose()) + self.bias.clone()
    }
}
pub struct LayerCache<B: Backend> {
    pub input: Tensor<B, 2>,
    pub preactivation: Tensor<B, 2>,
    pub activation: Tensor<B, 2>,
}
pub struct ForwardCache<B: Backend> {
    pub version: u64,
    network_id: u64,
    revision: u64,
    pub trunk: Vec<LayerCache<B>>,
    pub head_input: Tensor<B, 2>,
    pub mean: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
    pub log_std: Tensor<B, 2>,
}
pub struct Output<B: Backend> {
    pub mean: Tensor<B, 2>,
    pub value: Tensor<B, 2>,
}
pub struct Network<B: Backend> {
    pub trunk: Vec<Dense<B>>,
    pub actor: Dense<B>,
    pub critic: Dense<B>,
    pub log_std: Tensor<B, 2>,
    version: u64,
    network_id: u64,
    revision: u64,
    pub input_size: usize,
}
impl<B: Backend> Network<B> {
    pub fn new(config: &Config, device: &B::Device) -> Result<Self> {
        config.validate()?;
        let mut rng = ChaCha12Rng::seed_from_u64(config.network_seed);
        let mut build = |input: usize, out: usize, gain: f32| -> Result<Dense<B>> {
            let scale = gain / (input as f32).sqrt();
            let w: Vec<f32> = (0..input * out)
                .map(|_| {
                    let z: f32 = StandardNormal.sample(&mut rng);
                    z * scale
                })
                .collect();
            Ok(Dense {
                weights: matrix(&w, out, input, device)?,
                bias: matrix(&vec![0.; out], 1, out, device)?,
            })
        };
        let mut input = config.observation.dimension();
        let mut trunk = vec![];
        for &out in &config.hidden {
            trunk.push(build(input, out, 2f32.sqrt())?);
            input = out;
        }
        Ok(Self {
            trunk,
            actor: build(input, 2, 0.01)?,
            critic: build(input, 1, 1.)?,
            log_std: matrix(&config.initial_log_std, 1, 2, device)?,
            version: 0,
            network_id: {
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
            revision: 0,
            input_size: config.observation.dimension(),
        })
    }
    pub fn version(&self) -> u64 {
        self.version
    }
    pub fn infer(&self, x: Tensor<B, 2>) -> Result<Output<B>> {
        ensure(x.dims()[1] == self.input_size, "network input D")?;
        let mut h = x;
        for layer in &self.trunk {
            h = activation::relu(layer.forward(h));
        }
        Ok(Output {
            mean: self.actor.forward(h.clone()),
            value: self.critic.forward(h),
        })
    }
    pub fn forward(&self, x: Tensor<B, 2>) -> Result<ForwardCache<B>> {
        ensure(x.dims()[1] == self.input_size, "network input D")?;
        let mut h = x;
        let mut trunk = vec![];
        for layer in &self.trunk {
            let input = h;
            let z = layer.forward(input.clone());
            h = activation::relu(z.clone());
            trunk.push(LayerCache {
                input,
                preactivation: z,
                activation: h.clone(),
            });
        }
        Ok(ForwardCache {
            version: self.version,
            network_id: self.network_id,
            revision: self.revision,
            trunk,
            head_input: h.clone(),
            mean: self.actor.forward(h.clone()),
            value: self.critic.forward(h),
            log_std: self.log_std.clone(),
        })
    }
    pub fn validate_cache(&self, cache: &ForwardCache<B>) -> Result<()> {
        ensure(
            cache.version == self.version
                && cache.network_id == self.network_id
                && cache.revision == self.revision,
            "stale forward cache: weights changed",
        )
    }
    pub fn parameters(&self) -> Vec<Tensor<B, 2>> {
        let mut p = vec![];
        for layer in self.trunk.iter().chain([&self.actor, &self.critic]) {
            p.push(layer.weights.clone());
            p.push(layer.bias.clone());
        }
        p.push(self.log_std.clone());
        p
    }
    /// Replaces a complete parameter set only after a complete backward/update. Bumps cache version.
    pub fn replace(&mut self, p: Vec<Tensor<B, 2>>, bounds: [f32; 2]) -> Result<()> {
        let old = self.parameters();
        ensure(
            old.len() == p.len() && old.iter().zip(&p).all(|(a, b)| a.dims() == b.dims()),
            "parameter shapes",
        )?;
        for t in &p {
            crate::backend::validate(t, "parameters")?;
        }
        let mut it = p.into_iter();
        for layer in self
            .trunk
            .iter_mut()
            .chain([&mut self.actor, &mut self.critic])
        {
            layer.weights = it.next().unwrap();
            layer.bias = it.next().unwrap();
        }
        self.log_std = it.next().unwrap().clamp(bounds[0], bounds[1]);
        self.version += 1;
        self.revision += 1;
        Ok(())
    }
    pub fn snapshot(&self) -> Result<Vec<Vec<f32>>> {
        self.parameters().into_iter().map(read).collect()
    }
    pub fn restore(&mut self, values: &[Vec<f32>], version: u64, bounds: [f32; 2]) -> Result<()> {
        let templates = self.parameters();
        ensure(values.len() == templates.len(), "checkpoint tensor count")?;
        let p = values
            .iter()
            .zip(templates)
            .map(|(v, t)| {
                let [r, c] = t.dims();
                matrix(v, r, c, &t.device())
            })
            .collect::<Result<Vec<_>>>()?;
        self.replace(p, bounds)?;
        self.version = version;
        Ok(())
    }
}
