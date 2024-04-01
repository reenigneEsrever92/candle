use std::mem::size_of;

use bytemuck::{Pod, Zeroable};
use wgpu::core::id::BufferId;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, Buffer, Id, PipelineLayoutDescriptor,
};

use crate::{WgpuBackend, WgpuBackendResult};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RandUniformInput {
    seed: u32,
    min: f32,
    max: f32,
}

impl WgpuBackend {
    pub fn fill_ones(&self, buffer: Id<Buffer>) -> WgpuBackendResult<()> {
        todo!()
    }

    pub fn fill_zeroes(&self, buffer: Id<Buffer>) -> WgpuBackendResult<()> {
        todo!()
    }
}

#[cfg(test)]
pub mod tests {

    use crate::WgpuBackend;

    #[test]
    fn test_fill_ones() {
        let backend = WgpuBackend::new().unwrap();
        let output_buffer = backend.create_buffer(1024).unwrap();

        backend.fill_ones(output_buffer).unwrap();

        let floats: Vec<f32> = backend.read_buf_as::<f32>(output_buffer).unwrap();

        assert_eq!(floats.len(), 256);
        assert_eq!(floats.get(0).unwrap(), &1.0);
        assert_eq!(floats.get(511).unwrap(), &1.0);
    }
}
