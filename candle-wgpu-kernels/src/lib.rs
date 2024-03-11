use bytemuck::Pod;
use core::slice;
use std::{cell::RefCell, collections::HashMap, ops::DerefMut, rc::Rc, sync::Arc};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindingResource, Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Id,
    MaintainResult,
};

mod conv;
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
    buffers: Arc<RefCell<Vec<Buffer>>>,
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
                        buffers: Arc::new(RefCell::new(Vec::new())),
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

        self.buffers.borrow_mut().push(output_buffer);

        match self.device.poll(wgpu::Maintain::wait()) {
            // this an error!?
            MaintainResult::SubmissionQueueEmpty => {
                Ok(self.buffers.borrow().last().unwrap().global_id())
            }
            MaintainResult::Ok => Ok(self.buffers.borrow().last().unwrap().global_id()),
        }
    }

    pub fn create_buffer(&mut self, size: u64) -> WgpuBackendResult<Id<Buffer>> {
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.buffers.borrow_mut().push(buffer);

        Ok(self.buffers.borrow().last().unwrap().global_id())
    }

    pub fn read_buf(&mut self, buf_id: Id<Buffer>) -> WgpuBackendResult<Vec<u8>> {
        Ok(smol::block_on(async {
            let output_buffer = {
                let borrow = self.buffers.borrow();
                let buffer = borrow.iter().find(|buf| buf.global_id() == buf_id).unwrap();

                let output_buffer = self.device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: buffer.size(),
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

                encoder.copy_buffer_to_buffer(buffer, 0, &output_buffer, 0, buffer.size());

                self.queue.submit(Some(encoder.finish()));

                output_buffer
                // drop borrow before await is called
            };

            let buffer_slice = output_buffer.slice(..);
            // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
            let (sender, receiver) = flume::bounded(1);
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            // Poll the device in a blocking manner so that our future resolves.
            // In an actual application, `device.poll(...)` should
            // be called in an event loop or on another thread.
            self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

            // Awaits until `buffer_future` can be read from
            receiver.recv_async().await.unwrap().unwrap();
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            output_buffer.unmap(); // Unmaps buffer from memory
            result
        }))
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
        let mut backend = WgpuBackend::new().unwrap();
        backend.create_buffer_with_data(&data).unwrap();
    }
}
