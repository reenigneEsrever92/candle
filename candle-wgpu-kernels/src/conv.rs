use std::borrow::BorrowMut;

use bytemuck::{Pod, Zeroable};
use wgpu::{
    util::BufferInitDescriptor, util::DeviceExt, BindGroupDescriptor, Buffer, BufferUsages, Id,
    PipelineLayoutDescriptor,
};

use crate::{WgpuBackend, WgpuBackendResult};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct WgpuConvParams {
    batch_size: u32,
    input_length: u32,
    channels_in: u32,
    channels_out: u32,
    kernel_size: u32,
    groups: u32,
    padding: u32,
    stride: u32,
    dilation: u32,
}

impl Default for WgpuConvParams {
    fn default() -> Self {
        Self {
            batch_size: 1,
            input_length: 1,
            channels_in: 1,
            channels_out: 1,
            kernel_size: 1,
            groups: 1,
            padding: 0,
            stride: 1,
            dilation: 0,
        }
    }
}

impl WgpuBackend {
    pub fn conv1d(
        &mut self,
        input: Id<Buffer>,
        params: &WgpuConvParams,
    ) -> WgpuBackendResult<Id<Buffer>> {
        smol::block_on(async {
            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                        "conv1d_f32.wgsl"
                    ))),
                });

            let output_buffer_id = self.create_buffer(8).unwrap();

            let buffers = self.buffers.borrow();

            let input_buffer = buffers.iter().find(|buf| buf.global_id() == input).unwrap();

            let output_buffer = buffers
                .iter()
                .find(|buf| buf.global_id() == output_buffer_id)
                .unwrap();

            let params_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(params),
                usage: BufferUsages::UNIFORM,
            });

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
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
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
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
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

            let compute_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: "conv1d",
                    });

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

            self.queue.submit(Some(encoder.finish()));

            Ok(output_buffer_id)
        })
    }

    pub fn conv2d(
        &mut self,
        input: Id<Buffer>,
        params: &WgpuConvParams,
    ) -> WgpuBackendResult<Id<Buffer>> {
        smol::block_on(async {
            let output_buffer_size =
                (params.input_length * params.batch_size * params.channels_out
                    + params.padding * 2)
                    * 4; // 4 bytes for f32

            let module = self
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                        "conv1d_f32.wgsl"
                    ))),
                });

            let output_buffer_id = self.create_buffer(output_buffer_size as u64).unwrap();

            let buffers = self.buffers.lock().unwrap();

            let input_buffer = buffers.iter().find(|buf| buf.global_id() == input).unwrap();

            let output_buffer = buffers
                .iter()
                .find(|buf| buf.global_id() == output_buffer_id)
                .unwrap();

            let params_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
                label: None,
                contents: bytemuck::bytes_of(params),
                usage: BufferUsages::UNIFORM,
            });

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
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
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
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
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

            let compute_pipeline =
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: Some(&pipeline_layout),
                        module: &module,
                        entry_point: "conv1d",
                    });

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

            self.queue.submit(Some(encoder.finish()));

            Ok(output_buffer_id)
        })
    }
}

#[cfg(test)]
mod test {
    use crate::WgpuBackend;

    use super::WgpuConvParams;

    #[test]
    fn test_conv2d() {
        let tensor: [f32; 25] = [2.0; 25];
        let kernel = [0.5f32, 1.0, 0.5, 1.0, 2.0, 1.0, 0.5, 1.0, 0.5];
        let expected = [16f32; 9];

        println!("tensor: {tensor:?}");
    }

    #[test]
    fn test_conv1d() {
        // dims (1, 4, 5)
        let tensor = &[
            0.4056f32, -0.8689, -0.0773, -1.5630, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866, 0.4145,
            1.8025, -0.1536, 2.2013, -0.6836, 0.2477, 1.3127, -0.6957, 0.3278, -1.0124, 0.5599,
        ];
        // dims (2, 4, 3)
        let kernel = &[
            -0.8404f32, -0.3490, 0.0130, 1.3123, 0.1763, -1.9249, 1.4270, 0.9421, 0.8670, -0.7181,
            -1.1111, 0.8869, -1.2429, 1.8357, 1.6052, -1.3844, 0.3951, -1.2036, 0.6686, 1.6261,
            -0.6451, -0.0840, -1.4247, 0.5512,
        ];
        // dims (1, 2, 5)
        let expected = [
            2.4509315, 2.6357481, -1.3335553, 4.1392756, 0.56572014, 1.809062, -1.1783935,
            3.567513, 0.5069167, 3.3352304,
        ];

        let mut backend = WgpuBackend::new().unwrap();
        let buffer_id = backend.create_buffer_with_data(tensor).unwrap();
        let result = backend
            .conv1d(
                buffer_id,
                &WgpuConvParams {
                    input_length: 5, // tensor dim 3
                    batch_size: 1,   // tensor dim 1
                    channels_in: 4,  // kernel dim 2 - tensor dim 2
                    channels_out: 2, // kernel dim 1
                    kernel_size: 3,  // kernel dim 3
                    ..Default::default()
                },
            )
            .unwrap();
        let contents = backend.read_buf(result).unwrap();
        let floats: &[f32] = bytemuck::cast_slice(&contents);

        assert_eq!(floats.len(), 10);
        assert_eq!(floats, expected);
        println!("Result: {floats:?}");
    }
}
