//! Contiguous row-major tensors; no autodiff decorator. Weights stay on the device.
use crate::{Result, ensure, finite};
use burn_tensor::ElementConversion;
pub use burn_tensor::backend::Backend;
pub use burn_tensor::{Int, Tensor, TensorData, activation};
pub type Cpu = burn_ndarray::NdArray<f32>;
pub type Reference = burn_ndarray::NdArray<f64>;
/// Upstream Rayon and matrixmultiply threading are an explicit build choice.
pub const fn cpu_parallel_enabled() -> bool {
    cfg!(feature = "cpu-parallel")
}
/// CPU-only system metadata. Never creates a graphics instance or queries a GPU.
pub fn cpu_label() -> String {
    let model = std::env::var("PROCESSOR_IDENTIFIER")
        .ok()
        .or_else(|| {
            std::fs::read_to_string("/proc/cpuinfo")
                .ok()
                .and_then(|text| {
                    text.lines().find_map(|line| {
                        line.strip_prefix("model name")?
                            .split_once(':')
                            .map(|(_, value)| value.trim().to_owned())
                    })
                })
        })
        .unwrap_or_else(|| "unreported".into());
    format!(
        "Burn 0.20.1 NdArray f32; CPU={model}; OS={}; logical_available={}; cpu_parallel={}; RAYON_NUM_THREADS={}; MATMUL_NUM_THREADS={}",
        std::env::consts::OS,
        std::thread::available_parallelism()
            .map_or_else(|_| "unreported".into(), |n| n.to_string()),
        cpu_parallel_enabled(),
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "upstream default".into()),
        std::env::var("MATMUL_NUM_THREADS").unwrap_or_else(|_| "upstream default".into()),
    )
}
pub fn sync<B: Backend>(device: &B::Device) -> Result<()> {
    B::sync(device).map_err(|e| crate::Error::Device(format!("synchronization: {e:?}")))
}
#[cfg(feature = "gpu")]
pub type Gpu = burn_wgpu::Wgpu<f32, i32>;

pub fn matrix<B: Backend>(
    data: &[f32],
    rows: usize,
    cols: usize,
    device: &B::Device,
) -> Result<Tensor<B, 2>> {
    ensure(
        rows > 0 && cols > 0 && rows.checked_mul(cols) == Some(data.len()),
        "matrix shape",
    )?;
    finite(data, "matrix upload")?;
    Ok(Tensor::from_data(
        TensorData::new(data.to_vec(), [rows, cols]).convert::<B::FloatElem>(),
        device,
    ))
}
pub fn read<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Result<Vec<f32>> {
    let data = tensor.into_data().convert::<f32>();
    let values = data
        .to_vec::<f32>()
        .map_err(|e| crate::Error::Invalid(format!("tensor read: {e:?}")))?;
    finite(&values, "tensor download")?;
    Ok(values)
}
pub fn validate<B: Backend, const D: usize>(tensor: &Tensor<B, D>, context: &str) -> Result<()> {
    let valid = tensor.clone().is_finite().all().into_scalar();
    if valid.elem::<bool>() {
        Ok(())
    } else {
        Err(crate::Error::Numerical(context.into()))
    }
}

#[cfg(feature = "gpu")]
pub fn gpu_device(index: usize) -> Result<(burn_wgpu::WgpuDevice, String)> {
    // Select an actual hardware adapter. No software/CPU fallback, even on explicit GPU requests.
    // Linux deployments use Vulkan; DX12 remains available for historical Windows runs.
    let backends = if cfg!(target_os = "windows") {
        wgpu::Backends::VULKAN | wgpu::Backends::DX12
    } else {
        wgpu::Backends::VULKAN
    };
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });
    let adapters = instance.enumerate_adapters(backends);
    let adapters: Vec<_> = adapters
        .into_iter()
        .filter(|a| {
            matches!(
                a.get_info().device_type,
                wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
            )
        })
        .collect();
    let inventory = adapters
        .iter()
        .enumerate()
        .map(|(i, a)| format!("{i}: {:?}", a.get_info()))
        .collect::<Vec<_>>()
        .join("\n");
    let adapter = adapters.into_iter().nth(index).ok_or_else(|| {
        crate::Error::Device(format!(
            "explicit GPU request failed: hardware adapter {index} unavailable for {backends:?}. Install/check the Vulkan driver or choose an available hardware index. No CPU fallback. Hardware adapters:\n{inventory}"
        ))
    })?;
    let info = adapter.get_info();
    // Match CubeCL's WGSL setup requirements, retaining adapter limits/features.
    let descriptor = wgpu::DeviceDescriptor {
        label: Some("racing-ppo-f32"),
        required_features: adapter
            .features()
            .difference(wgpu::Features::MAPPABLE_PRIMARY_BUFFERS),
        required_limits: adapter.limits(),
        memory_hints: wgpu::MemoryHints::MemoryUsage,
        trace: wgpu::Trace::Off,
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&descriptor)).map_err(|e| {
        crate::Error::Device(format!(
            "GPU {index} ({info:?}) device creation failed; no CPU fallback: {e}"
        ))
    })?;
    let setup = burn_wgpu::WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend: info.backend,
    };
    let device = burn_wgpu::init_device(setup, Default::default());
    Ok((
        device,
        format!(
            "{inventory}\nselected_index={index}; selected={info:?}; Burn 0.20.1 WGPU WGSL f32"
        ),
    ))
}
