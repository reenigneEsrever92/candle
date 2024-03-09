use wgpu::{Adapter, Device, Instance, Queue};

#[derive(Debug)]
pub enum Error {}

pub type WgpuResult<T> = Result<T, Error>;

#[derive(Default)]
struct WgpuConfig {}

struct WgpuBackend {
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
}

impl WgpuBackend {
    pub fn new(config: WgpuConfig) -> WgpuResult<Self> {
        smol::block_on(async { Self::init(config).await })
    }

    async fn init(config: WgpuConfig) -> WgpuResult<Self> {
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
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    ..Default::default()
                },
                None, // Trace path
            )
            .await
            .unwrap();

        // device.create_shader_module(wgpu::ShaderModuleDescriptor {});

        Ok(Self {
            adapter,
            device,
            instance,
            queue,
        })
    }
}

#[cfg(test)]
mod test {
    use crate::{WgpuBackend, WgpuConfig};

    #[test]
    fn test_init() {
        WgpuBackend::new(WgpuConfig::default()).unwrap();
    }
}
