use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, PipelineLayoutDescriptor,
};

use crate::{WgpuBackend, WgpuResult};

const SHADER_CODE_F32: &str = concat!(include_str!("fill.wgsl"), include_str!("fill_f32.wgsl"));
const SHADER_CODE_F16: &str = concat!(include_str!("fill.wgsl"), include_str!("fill_f16.wgsl"));

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FillInput {
    seed: u32,
    min: f32,
    max: f32,
}

impl WgpuBackend {
    pub fn ones_f32(&self, amount: u64) -> WgpuResult<Vec<f32>> {
        self.run_kernel_f32("ones", 1, amount, 0.0, 0.0)
            .map(|bytes| bytemuck::cast_slice(&bytes).to_vec())
    }

    pub fn rand_uniform_f32(
        &self,
        seed: u32,
        amount: u64,
        min: f32,
        max: f32,
    ) -> WgpuResult<Vec<f32>> {
        self.run_kernel_f32("rand_uniform_f32", seed, amount, min, max)
            .map(|bytes| bytemuck::cast_slice(&bytes).to_vec())
    }

    pub fn ones_f16(&self, amount: u64) -> WgpuResult<Vec<u8>> {
        self.run_kernel_f16("ones", 1, amount, 0.0, 0.0)
    }

    fn run_kernel_f16(
        &self,
        entry_point: &str,
        seed: u32,
        amount: u64,
        min: f32,
        max: f32,
    ) -> WgpuResult<Vec<u8>> {
        smol::block_on(async {
            let inp = FillInput { seed, min, max };

            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                        "fill_f16.wgsl"
                    ))),
                });

            // Instantiates buffer without data.
            // `usage` of buffer specifies how it can be used:
            //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
            //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size_of::<f32>() as u64 * amount,
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
                size: size_of::<f32>() as u64 * amount,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

            let input_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
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

            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
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
                        entry_point,
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
                size_of::<f32>() as u64 * amount,
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

    fn run_kernel_f32(
        &self,
        entry_point: &str,
        seed: u32,
        amount: u64,
        min: f32,
        max: f32,
    ) -> WgpuResult<Vec<u8>> {
        smol::block_on(async {
            let inp = FillInput { seed, min, max };

            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_CODE_F32)),
                });

            // Instantiates buffer without data.
            // `usage` of buffer specifies how it can be used:
            //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
            //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: size_of::<f32>() as u64 * amount,
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
                size: size_of::<f32>() as u64 * amount,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

            let input_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
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

            let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
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
                        entry_point,
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
                size_of::<f32>() as u64 * amount,
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
}

#[cfg(test)]
pub mod tests {
    use half::f16;

    use crate::WgpuBackend;

    #[test]
    fn test_fill_ones_f16() {
        let backend = WgpuBackend::new().unwrap();
        let rand = backend.ones_f16(512).unwrap();

        let floats: Vec<f32> = bytemuck::cast_slice(&rand).to_vec();

        assert_eq!(floats.len(), 512);
        // assert_eq!(floats.get(0).unwrap(), &f16::from_f32(1.0));
        // assert_eq!(floats.get(511).unwrap(), &f16::from_f32(1.0));
    }

    #[test]
    fn test_fill_ones_f32() {
        let backend = WgpuBackend::new().unwrap();
        let rand = backend.ones_f32(512).unwrap();

        let floats: Vec<f32> = bytemuck::cast_slice(&rand).to_vec();

        assert_eq!(floats.len(), 512);
        assert_eq!(floats.get(0).unwrap(), &1.0);
        assert_eq!(floats.get(511).unwrap(), &1.0);
    }

    #[test]
    fn test_fill_random() {
        let (min, max) = (-30.0, 30.0);
        let shape = vec![1024, 10];
        let length = shape.iter().product::<u64>();
        let seed = 299792458;

        let backend = WgpuBackend::new().unwrap();
        let rand = backend.rand_uniform_f32(seed, length, min, max).unwrap();

        let floats: Vec<f32> = bytemuck::cast_slice(&rand).to_vec();
        let mean = floats.iter().sum::<f32>() / floats.len() as f32;

        assert_eq!(floats.len() as u64, length);
        assert!(floats.iter().all(|v| { *v >= min && *v <= max }));
        // mean must approach zero given a large enough sample size
        assert!(mean >= -1.0);
        assert!(mean <= 1.0);
    }
}
