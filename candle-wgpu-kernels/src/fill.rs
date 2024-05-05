use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, Id};

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

    pub fn fill_ones_u8(&self, buffer: Id<Buffer>) -> WgpuBackendResult<()> {
        self.run_shader(Shader::FillU8, "ones", buffer, &FillParams::default())?;
        Ok(())
    }

    pub fn fill_zeroes_u8(&self, buffer: Id<Buffer>) -> WgpuBackendResult<()> {
        self.run_shader(Shader::FillU8, "zeroes", buffer, &FillParams::default())?;
        Ok(())
    }

    pub fn fill_ones_u32(&self, buffer: Id<Buffer>) -> WgpuBackendResult<()> {
        self.run_shader(Shader::FillU32, "ones", buffer, &FillParams::default())?;
        Ok(())
    }

    pub fn fill_zeroes_u32(&self, buffer: Id<Buffer>) -> WgpuBackendResult<()> {
        self.run_shader(Shader::FillU32, "zeroes", buffer, &FillParams::default())?;
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

    #[test]
    fn test_fill_ones_u8() {
        let backend = WgpuBackend::new().unwrap();
        let output_buffer = backend.create_buffer(256).unwrap();

        backend.fill_ones_u8(output_buffer).unwrap();

        let result = backend.read_buf_as::<u8>(output_buffer).unwrap();

        assert_eq!(result.len(), 256);
        assert_eq!(result.iter().map(|v| u32::from(*v)).sum::<u32>(), 256);
        assert_eq!(result.first().unwrap(), &1);
        assert_eq!(result.last().unwrap(), &1);
    }

    #[test]
    fn test_fill_zeroes_u8() {
        let backend = WgpuBackend::new().unwrap();
        let output_buffer = backend.create_buffer(256).unwrap();

        backend.fill_zeroes_u8(output_buffer).unwrap();

        let result = backend.read_buf_as::<u8>(output_buffer).unwrap();

        assert_eq!(result.len(), 256);
        assert_eq!(result.iter().map(|v| u32::from(*v)).sum::<u32>(), 0);
        assert_eq!(result.first().unwrap(), &0);
        assert_eq!(result.last().unwrap(), &0);
    }
}
