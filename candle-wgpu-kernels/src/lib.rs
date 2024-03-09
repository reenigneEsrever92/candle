use std::sync::Arc;

mod fill;

type WgpuResult<T> = Result<T, WgpuError>;

#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    #[error("Wgpu backend could not be initialized")]
    InitializationError,
    #[error("Wgpu device could not be requested")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error("Error awaiting buffer")]
    BufferError(#[from] wgpu::BufferAsyncError),
}

#[derive(Debug, Clone)]
pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
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
}

#[cfg(test)]
pub mod tests {
    use crate::WgpuBackend;

    #[test]
    fn test_init() {
        WgpuBackend::new().unwrap();
    }
}
