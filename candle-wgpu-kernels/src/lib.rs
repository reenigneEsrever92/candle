use core::slice;
use std::sync::Arc;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MaintainResult,
};

mod fill;

type WgpuBackendResult<T> = Result<T, WgpuBackendError>;

#[derive(Debug, thiserror::Error)]
pub enum WgpuBackendError {
    #[error("Wgpu backend could not be initialized")]
    InitializationError,
    #[error("Wgpu device could not be requested")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error("Error awaiting buffer")]
    BufferError(#[from] wgpu::BufferAsyncError),
    #[error("Wgpu submission queue empty: {0}")]
    SubmissionQueueEmpty(String),
}

#[derive(Debug, Clone)]
pub struct WgpuBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

impl WgpuBackend {
    pub fn new() -> WgpuBackendResult<Self> {
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
                .ok_or(Err(WgpuBackendError::InitializationError));

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

    pub fn get_gpu_id(&self) -> usize {
        self.device.global_id().inner() as usize
    }

    pub fn create_buffer_with_data<T>(&self, data: &[T]) -> WgpuBackendResult<wgpu::Id<Buffer>> {
        let data = self.cast_to_bytes(data);
        let input_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: data,
            usage: BufferUsages::COPY_SRC,
        });
        let output_buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: data.len() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Create Buffer Encoder"),
            });

        encoder.copy_buffer_to_buffer(&input_buffer, 0, &output_buffer, 0, data.len() as u64);

        self.queue.submit(Some(encoder.finish()));

        match self.device.poll(wgpu::Maintain::wait()) {
            // this an error!?
            MaintainResult::SubmissionQueueEmpty => Ok(output_buffer.global_id()),
            MaintainResult::Ok => Ok(output_buffer.global_id()),
        }
    }

    #[inline]
    fn cast_to_bytes<T>(&self, data: &[T]) -> &[u8] {
        let size = core::mem::size_of_val(data);
        let ptr = data.as_ptr() as *const u8;
        unsafe { slice::from_raw_parts(ptr, size) }
    }
}

#[cfg(test)]
pub mod tests {
    use crate::WgpuBackend;

    #[test]
    fn test_init() {
        WgpuBackend::new().unwrap();
    }

    #[test]
    fn test_create_buffer() {
        let data: [f32; 1024] = [1.0; 1024];
        let backend = WgpuBackend::new().unwrap();
        backend.create_buffer_with_data(&data).unwrap();
    }
}
