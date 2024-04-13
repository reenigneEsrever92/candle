use crate::kernel::Shader;
use crate::{WgpuBackend, WgpuBackendResult};
use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, Id};

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, Pod, Zeroable)]
struct ConvertParams {
    dummy: u32,
}

impl WgpuBackend {
    pub fn convert_u8_to_f32(
        &self,
        input_buffer_id: Id<Buffer>,
        output_buffer: Id<Buffer>,
    ) -> WgpuBackendResult<()> {
        self.run_shader_with_input(
            Shader::ConvertU8ToF32,
            "convert",
            input_buffer_id,
            output_buffer,
            &ConvertParams::default(),
        )?;

        Ok(())
    }

    pub fn convert_u32_to_f32(
        &self,
        input_buffer_id: Id<Buffer>,
        output_buffer: Id<Buffer>,
    ) -> WgpuBackendResult<()> {
        self.run_shader_with_input(
            Shader::ConvertU8ToF32,
            "convert",
            input_buffer_id,
            output_buffer,
            &ConvertParams::default(),
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::WgpuBackend;

    #[test]
    fn test_convert_u8_to_f32() {
        let backend = WgpuBackend::new().unwrap();
        let input_buffer = backend.create_buffer_with_data(&[1u8; 8]).unwrap();
        let output_buffer = backend.create_buffer(8 * 4).unwrap();

        backend
            .convert_u8_to_f32(input_buffer, output_buffer)
            .unwrap();

        let result = backend.read_buf_as::<f32>(output_buffer).unwrap();

        assert_eq!(result, &[1.0f32; 8])
    }
}
