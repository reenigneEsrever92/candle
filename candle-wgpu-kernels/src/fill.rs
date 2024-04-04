use bytemuck::{Pod, Zeroable};
use wgpu::{util::DeviceExt, Buffer, Id};

use crate::kernel::Shader;
use crate::{WgpuBackend, WgpuBackendResult};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
struct FillParams {
    dummy: u32,
}

impl WgpuBackend {
    pub fn fill_ones(&self, buffer: Id<Buffer>) -> WgpuBackendResult<()> {
        self.run_shader(Shader::Fill, "ones", buffer, &FillParams::default())?;
        Ok(())
    }

    pub fn fill_zeroes(&self, buffer: Id<Buffer>) -> WgpuBackendResult<()> {
        self.run_shader(Shader::Fill, "zeroes", buffer, &FillParams::default())?;
        Ok(())
    }
}

#[cfg(test)]
pub mod tests {
    use crate::WgpuBackend;

    #[test]
    fn test_fill_ones() {
        let backend = WgpuBackend::new().unwrap();
        let output_buffer = backend.create_buffer(256 * 4).unwrap();

        backend.fill_ones(output_buffer).unwrap();

        let floats: Vec<f32> = backend.read_buf_as::<f32>(output_buffer).unwrap();

        assert_eq!(floats.len(), 256);
        assert_eq!(floats.iter().sum::<f32>(), 256.0);
        assert_eq!(floats.first().unwrap(), &1.0);
        assert_eq!(floats.last().unwrap(), &1.0);
    }

    #[test]
    fn test_fill_zeroes() {
        let backend = WgpuBackend::new().unwrap();
        let output_buffer = backend.create_buffer(256 * 4).unwrap();

        backend.fill_zeroes(output_buffer).unwrap();

        let floats: Vec<f32> = backend.read_buf_as::<f32>(output_buffer).unwrap();

        assert_eq!(floats.len(), 256);
        assert_eq!(floats.iter().sum::<f32>(), 0.0);
    }
}
