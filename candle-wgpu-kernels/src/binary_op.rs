use crate::kernel::Shader;
use crate::{WgpuBackend, WgpuBackendResult};
use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, Id};

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, Pod, Zeroable)]
struct BinaryOpParams {
    dummy: u32,
}

impl WgpuBackend {
    pub fn binary_max(
        &self,
        lhs_buffer: Id<Buffer>,
        rhs_buffer: Id<Buffer>,
        output_buffer: Id<Buffer>,
    ) -> WgpuBackendResult<()> {
        self.run_shader_with_input_2(
            Shader::BinaryOp,
            "max",
            lhs_buffer,
            rhs_buffer,
            output_buffer,
            &BinaryOpParams::default(),
        )?;

        Ok(())
    }

    pub fn binary_add(
        &self,
        lhs_buffer: Id<Buffer>,
        rhs_buffer: Id<Buffer>,
        output_buffer: Id<Buffer>,
    ) -> WgpuBackendResult<()> {
        self.run_shader_with_input_2(
            Shader::BinaryOp,
            "add",
            lhs_buffer,
            rhs_buffer,
            output_buffer,
            &BinaryOpParams::default(),
        )?;

        Ok(())
    }

    pub fn binary_sub(
        &self,
        lhs_buffer: Id<Buffer>,
        rhs_buffer: Id<Buffer>,
        output_buffer: Id<Buffer>,
    ) -> WgpuBackendResult<()> {
        self.run_shader_with_input_2(
            Shader::BinaryOp,
            "sub",
            lhs_buffer,
            rhs_buffer,
            output_buffer,
            &BinaryOpParams::default(),
        )?;

        Ok(())
    }

    pub fn binary_mul(
        &self,
        lhs_buffer: Id<Buffer>,
        rhs_buffer: Id<Buffer>,
        output_buffer: Id<Buffer>,
    ) -> WgpuBackendResult<()> {
        self.run_shader_with_input_2(
            Shader::BinaryOp,
            "mul",
            lhs_buffer,
            rhs_buffer,
            output_buffer,
            &BinaryOpParams::default(),
        )?;

        Ok(())
    }

    pub fn binary_div(
        &self,
        lhs_buffer: Id<Buffer>,
        rhs_buffer: Id<Buffer>,
        output_buffer: Id<Buffer>,
    ) -> WgpuBackendResult<()> {
        self.run_shader_with_input_2(
            Shader::BinaryOp,
            "div",
            lhs_buffer,
            rhs_buffer,
            output_buffer,
            &BinaryOpParams::default(),
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::WgpuBackend;

    #[test]
    fn test_binary_mul() {
        let backend = WgpuBackend::new().unwrap();
        let lhs_buffer = backend.create_buffer_with_data(&[1f32, 2.25]).unwrap();
        let rhs_buffer = backend.create_buffer_with_data(&[1f32, 0.5]).unwrap();
        let output_buffer = backend.create_buffer(4 * 2).unwrap();

        backend
            .binary_mul(lhs_buffer, rhs_buffer, output_buffer)
            .unwrap();

        let result = backend.read_buf_as::<f32>(output_buffer).unwrap();

        assert_eq!(result, [1f32, 1.125]);
    }
}
