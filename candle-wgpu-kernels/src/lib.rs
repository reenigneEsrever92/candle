use std::{
    borrow::Cow,
    collections::HashMap,
    fs::read,
    io::Read,
    mem::size_of,
    num::NonZeroU64,
    ops::Deref,
    sync::{Arc, Mutex},
};

use bytemuck::{Pod, Zeroable};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    Buffer, Features, PipelineLayoutDescriptor, ShaderModule,
};

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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Input {
    seed: u32,
}

pub enum FillFunc {
    Ones,
    RandUniformF32,
}

impl From<FillFunc> for &str {
    fn from(value: FillFunc) -> Self {
        match value {
            FillFunc::Ones => "ones",
            FillFunc::RandUniformF32 => "rand_uniform_f32",
        }
    }
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

    // DType::F32 => "rand_uniform_f32",
    // DType::F16 => "rand_uniform_f16",
    // DType::BF16 => "rand_uniform_bf16",
    // pub fn rand_uniform_f32(&self, numbers: &[f32]) -> WgpuResult<>
    pub fn run_kernel_fill(&self, func: nums: u64, func: &str, seed: u32) -> WgpuResult<Vec<u8>> {
        smol::block_on(async {
            let inp = Input { seed };

            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                        "fill.wgsl"
                    ))),
                });

            // Instantiates buffer without data.
            // `usage` of buffer specifies how it can be used:
            //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
            //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size_of::<f32>() as u64 * nums,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            // Instantiates buffer with data (`numbers`).
            // Usage allowing the buffer to be:
            //   A storage buffer (can be bound within a bind group and thus available to a shader).
            //   The destination of a copy.
            //   The source of a copy.
            let storage_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Storage Buffer"),
                size: size_of::<f32>() as u64 * nums,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

            let input_buffer = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Input Buffer"),
                    contents: bytemuck::bytes_of(&inp),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

            // A bind group defines how buffers are accessed by shaders.
            // It is to WebGPU what a descriptor set is to Vulkan.
            // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

            // A pipeline specifies the operation of a shader

            // Instantiates the bind group, once again specifying the binding of buffers
            // let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
            let bind_group_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Uniform,
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                        ],
                    });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: storage_buffer.as_entire_binding(),
                    },
                ],
            });

            let pipeline_layout = self
                .device
                .create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

            // Instantiates the pipeline.
            let compute_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: func,
                    });

            // A command encoder executes one or many pipelines.
            // It is to WebGPU what a command buffer is to Vulkan.
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&compute_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.insert_debug_marker("compute");
                cpass.dispatch_workgroups(64, 1, 1); // Number of cells to run, the (x,y,z) size of item being processed
            }
            // Sets adds copy operation to command encoder.
            // Will copy data from storage buffer on GPU to staging buffer on CPU.
            encoder.copy_buffer_to_buffer(
                &storage_buffer,
                0,
                &output_buffer,
                0,
                size_of::<f32>() as u64 * nums,
            );

            // Submits command encoder for processing
            self.queue.submit(Some(encoder.finish()));

            // Note that we're not calling `.await` here.
            let buffer_slice = output_buffer.slice(..);
            // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
            let (sender, receiver) = flume::bounded(1);
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            // Poll the device in a blocking manner so that our future resolves.
            // In an actual application, `device.poll(...)` should
            // be called in an event loop or on another thread.
            self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

            // Awaits until `buffer_future` can be read from
            receiver.recv_async().await.unwrap()?;
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are got in bytes, this converts these bytes back to u32
            let result = bytemuck::cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            output_buffer.unmap(); // Unmaps buffer from memory
                                   // If you are familiar with C++ these 2 lines can be thought of similarly to:
                                   //   delete myPointer;
                                   //   myPointer = NULL;
                                   // It effectively frees the memory

            // Returns data from buffer
            Ok(result)
        })
    }

    pub fn rand_uniform_f16(&self) -> Vec<u8> {
        todo!()
    }

    pub fn rand_uniform_bf16(&self) -> Vec<u8> {
        todo!()
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
    fn test_fill_ones() {
        let backend = WgpuBackend::new().unwrap();
        let rand = backend.run_kernel_fill(512, "ones").unwrap();

        let floats: Vec<f32> = bytemuck::cast_slice(&rand).to_vec();

        assert_eq!(floats.len(), 512);
        assert_eq!(floats.get(0).unwrap(), &0.0);
        assert_eq!(floats.get(511).unwrap(), &511.0);
    }
}
