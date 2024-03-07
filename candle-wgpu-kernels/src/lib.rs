use std::{
    collections::HashMap,
    io::Read,
    num::NonZeroU64,
    ops::Deref,
    sync::{Arc, Mutex},
};

use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, ShaderModule,
};

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
    shaders: Arc<Mutex<WgpuShaders>>,
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
                        shaders: Arc::new(Mutex::new(WgpuShaders::default())),
                    })
                }
                Err(e) => e,
            }
        })
    }

    // DType::F32 => "rand_uniform_f32",
    // DType::F16 => "rand_uniform_f16",
    // DType::BF16 => "rand_uniform_bf16",
    pub fn rand_uniform_f32(&self, amount: usize) -> Vec<u8> {
        let mut shaders = self.shaders.lock().unwrap();

        // Gets the size in bytes of the buffer.
        let buffer_size = amount * std::mem::size_of::<f32>();
        let size = buffer_size as wgpu::BufferAddress;
        let buffer = Vec::with_capacity(buffer_size);

        // Instantiates buffer without data.
        // `usage` of buffer specifies how it can be used:
        //   `BufferUsage::MAP_READ` allows it to be read (outside the shader).
        //   `BufferUsage::COPY_DST` allows it to be the destination of the copy.
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Instantiates buffer with data (`numbers`).
        // Usage allowing the buffer to be:
        //   A storage buffer (can be bound within a bind group and thus available to a shader).
        //   The destination of a copy.
        //   The source of a copy
        let storage_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Storage Buffer"),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            contents: &buffer,
        });

        // A bind group defines how buffers are accessed by shaders.
        // It is to WebGPU what a descriptor set is to Vulkan.
        // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

        // A pipeline specifies the operation of a shader

        // Instantiates the pipeline.
        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: None,
                    module: shaders.get_uniform_f32(self),
                    entry_point: "main",
                });

        // Instantiates the bind group, once again specifying the binding of buffers.
        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: storage_buffer.as_entire_binding(),
            }],
        });

        // A command encoder executes one or many pipelines.
        // It is to WebGPU what a command buffer is to Vulkan.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("compute collatz iterations");
            cpass.dispatch_workgroups(amount as u32, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
        }
        // Sets adds copy operation to command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

        // Submits command encoder for processing
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        self.device.poll(wgpu::Maintain::Wait);

        // Gets contents of buffer
        let data = buffer_slice.get_mapped_range();

        // Since contents are got in bytes, this converts these bytes back to u32
        let result = data.iter().cloned().collect();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap(); // Unmaps buffer from memory
                                // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                //   delete myPointer;
                                //   myPointer = NULL;
                                // It effectively frees the memory

        // Returns data from buffer
        result
    }

    pub fn rand_uniform_f16(&self) -> Vec<u8> {
        todo!()
    }

    pub fn rand_uniform_bf16(&self) -> Vec<u8> {
        todo!()
    }
}

#[derive(Debug, Default)]
struct WgpuShaders {
    rand_uniform_f32: Option<ShaderModule>,
}

impl WgpuShaders {
    fn get_uniform_f32(&mut self, backend: &WgpuBackend) -> &ShaderModule {
        match self.rand_uniform_f32 {
            Some(shader) => &shader,
            None => {
                let shader = self.compile(&backend, include_str!("random.wgsl"));
                self.rand_uniform_f32 = Some(shader);
                &shader
            }
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
