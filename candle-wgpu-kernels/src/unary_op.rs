use crate::kernel::Shader;
use crate::{WgpuBackend, WgpuBackendResult};
use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, Id};

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, Pod, Zeroable)]
struct UnaryOpParams {
    dummy: u32,
}

impl WgpuBackend {
    pub fn sqrt(&self, operand: Id<Buffer>, output: Id<Buffer>) -> WgpuBackendResult<()> {
        self.run_shader_with_input(
            Shader::UnaryOp,
            "sqrt_f32",
            operand,
            output,
            &UnaryOpParams::default(),
        )?;

        Ok(())
    }

    pub fn neg(&self, operand: Id<Buffer>, output: Id<Buffer>) -> WgpuBackendResult<()> {
        self.run_shader_with_input(
            Shader::UnaryOp,
            "neg",
            operand,
            output,
            &UnaryOpParams::default(),
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::WgpuBackend;

    #[test]
    fn test_sqrt() {
        let backend = WgpuBackend::new().unwrap();
        let operand_buffer = backend.create_buffer_with_data(&[9f32, 25.0]).unwrap();
        let output_buffer = backend.create_buffer(4 * 2).unwrap();

        backend.sqrt(operand_buffer, output_buffer).unwrap();

        let result = backend.read_buf_as::<f32>(output_buffer).unwrap();

        assert_eq!(result, [3f32, 5.0]);
    }
}
