use std::{collections::HashMap, num::NonZeroU64, sync::Arc};

use wgpu::{ShaderModel, ShaderModule};

type WgpuResult<T> = Result<T, WgpuError>;

#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    #[error("Wgpu backend could not be initialized")]
    InitializationError,
    #[error("Wgpu device could not be requested")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
}

#[derive(Debug, Clone)]
pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    shaders: WgpuShaders,
}

impl WgpuBackend {
    pub fn new() -> WgpuResult<Self> {
        smol::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::default(),
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .ok_or(Err(WgpuError::InitializationError));

            match adapter {
                Ok(adapter) => {
                    let (device, queue) = adapter.request_device(&Default::default(), None).await?;

                    Ok(Self {
                        device: Arc::new(device),
                        queue: Arc::new(queue),
                    })
                }
                Err(e) => e,
            }
        })
    }

    // DType::F32 => "rand_uniform_f32",
    // DType::F16 => "rand_uniform_f16",
    // DType::BF16 => "rand_uniform_bf16",
    pub fn rand_uniform_f32(&self) -> Vec<u8> {
        let shader = self.shaders.get_uniform_f32(self);
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                // XXX - some graphics cards do not support empty bind layout groups, so
                // create a dummy entry.
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    count: None,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: Some(NonZeroU64::new(1).unwrap()),
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                    },
                },
            ],
        });
        todo!()
    }

    pub fn rand_uniform_f16(&self) -> Vec<u8> {
        todo!()
    }

    pub fn rand_uniform_bf16(&self) -> Vec<u8> {
        todo!()
    }
}

struct WgpuShaders {
    rand_uniform_f32: Option<ShaderModule>,
}

impl WgpuShaders {
    fn get_uniform_f32(&mut self, backend: &WgpuBackend) -> ShaderModule {
        match self.rand_uniform_f32 {
            Some(shader) => shader,
            None => self.compile(&backend, include_str!("random.wgsl")),
        }
    }

    fn compile(&mut self, backend: &WgpuBackend, module: &str) -> ShaderModule {
        backend
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(module)),
            })
    }
}

#[cfg(test)]
pub mod tests {
    use crate::WgpuBackend;

    #[test]
    fn test_init() {
        WgpuBackend::new().unwrap();
    }
}
