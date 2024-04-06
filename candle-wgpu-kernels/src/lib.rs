use bytemuck::Pod;
use core::slice;
use kernel::{Kernels, Shader};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::core::id::BufferId;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor,
    DeviceDescriptor, DeviceType, Id, Limits, MaintainResult, PipelineLayoutDescriptor,
    ShaderModule,
};

pub mod conv;
mod copy;
pub mod fill;
mod kernel;
mod random;

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
    buffers: Arc<Mutex<Vec<Buffer>>>,
    kernels: Arc<Kernels>,
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
                    let (device, queue) = adapter
                        .request_device(
                            &DeviceDescriptor {
                                required_limits: Limits::downlevel_defaults(),
                                ..Default::default()
                            },
                            None,
                        )
                        .await?;

                    let kernels = Arc::new(Kernels::new(&device));

                    Ok(Self {
                        device: Arc::new(device),
                        queue: Arc::new(queue),
                        buffers: Arc::new(Mutex::new(Vec::new())),
                        kernels,
                    })
                }
                Err(e) => e,
            }
        })
    }

    pub fn get_gpu_id(&self) -> usize {
        self.device.global_id().inner() as usize
    }

    pub fn create_buffer_with_data<T>(&self, data: &[T]) -> WgpuBackendResult<Id<Buffer>> {
        let data = self.cast_to_bytes(data);
        let input_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: data,
            usage: BufferUsages::COPY_SRC,
        });
        let output_buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size: data.len() as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Create Buffer Encoder"),
            });

        encoder.copy_buffer_to_buffer(&input_buffer, 0, &output_buffer, 0, data.len() as u64);

        self.queue.submit(Some(encoder.finish()));

        let mut buffers = self.buffers.lock().unwrap();

        buffers.push(output_buffer);

        match self.device.poll(wgpu::Maintain::wait()) {
            // this an error!?
            MaintainResult::SubmissionQueueEmpty => Ok(buffers.last().unwrap().global_id()),
            MaintainResult::Ok => Ok(buffers.last().unwrap().global_id()),
        }
    }

    pub fn create_buffer(&self, size: u64) -> WgpuBackendResult<Id<Buffer>> {
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut buffers = self.buffers.lock().unwrap();

        buffers.push(buffer);

        Ok(buffers.last().unwrap().global_id())
    }

    pub fn read_buf_as<T>(&self, buf_id: Id<Buffer>) -> WgpuBackendResult<Vec<T>> {
        // TODO check bounds and stuff
        let result = self.read_buf(buf_id)?;
        let ptr = result[..].as_ptr() as *mut T;
        let len = result.len() / core::mem::size_of::<T>();

        core::mem::forget(result);

        Ok(unsafe { Vec::from_raw_parts(ptr, len, len) })
    }

    pub fn read_buf(&self, buf_id: Id<Buffer>) -> WgpuBackendResult<Vec<u8>> {
        Ok(smol::block_on(async {
            let output_buffer = {
                let buffers = self.buffers.lock().unwrap();
                let buffer = buffers
                    .iter()
                    .find(|buf| buf.global_id() == buf_id)
                    .unwrap();

                let output_buffer = self.device.create_buffer(&BufferDescriptor {
                    label: None,
                    size: buffer.size(),
                    usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });

                let mut encoder = self
                    .device
                    .create_command_encoder(&CommandEncoderDescriptor { label: None });

                encoder.copy_buffer_to_buffer(buffer, 0, &output_buffer, 0, buffer.size());

                self.queue.submit(Some(encoder.finish()));

                output_buffer
                // drop borrow before await is called
            };

            let buffer_slice = output_buffer.slice(..);

            let (sender, receiver) = flume::bounded(1);
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

            self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();

            receiver.recv_async().await.unwrap().unwrap();
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();

            drop(data);
            output_buffer.unmap();
            result
        }))
    }

    pub fn run_shader<P: Pod>(
        &self,
        shader: Shader,
        entry_point: &str,
        output_buffer_id: Id<Buffer>,
        params: &P,
    ) -> WgpuBackendResult<Id<Buffer>> {
        smol::block_on(async {
            let module = self.get_shader(shader);

            let buffers = self.buffers.lock().unwrap();

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
                        module,
                        entry_point,
                    });

            let mut encoder = self
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
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

    pub fn run_shader_with_input<P: Pod>(
        &self,
        shader: Shader,
        entry_point: &str,
        input_buffer_id: Id<Buffer>,
        output_buffer_id: Id<Buffer>,
        params: &P,
    ) -> WgpuBackendResult<Id<Buffer>> {
        smol::block_on(async {
            let module = self.get_shader(shader);

            let buffers = self.buffers.lock().unwrap();

            let input_buffer = buffers
                .iter()
                .find(|buf| buf.global_id() == input_buffer_id)
                .unwrap();

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
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        module,
                        entry_point,
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

    pub fn run_shader_with_input_2<P: Pod>(
        &self,
        shader: Shader,
        entry_point: &str,
        input_buffer_id: Id<Buffer>,
        input_buffer_id_2: Id<Buffer>,
        output_buffer_id: Id<Buffer>,
        params: &P,
    ) -> WgpuBackendResult<Id<Buffer>> {
        smol::block_on(async {
            let module = self.get_shader(shader);

            let buffers = self.buffers.lock().unwrap();

            let input_buffer = buffers
                .iter()
                .find(|buf| buf.global_id() == input_buffer_id)
                .unwrap();

            let input_buffer_2 = buffers
                .iter()
                .find(|buf| buf.global_id() == input_buffer_id_2)
                .unwrap();

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
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: None,
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
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
                        resource: input_buffer_2.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
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
                        module,
                        entry_point,
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

    #[inline]
    fn get_shader(&self, shader: Shader) -> &ShaderModule {
        match shader {
            Shader::Random => &self.kernels.random,
            Shader::Conv => &self.kernels.conv,
            Shader::Fill => &self.kernels.fill,
            Shader::FillU8 => &self.kernels.fill_u8,
            Shader::Copy => &self.kernels.copy,
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
        let mut backend = WgpuBackend::new().unwrap();
        backend.create_buffer_with_data(&data).unwrap();
    }
}
