use crate::kernel::Shader;
use crate::{WgpuBackend, WgpuBackendResult};
use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, Id};

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, Pod, Zeroable)]
struct AffineParams {
    mul: f32,
    add: f32,
}

impl WgpuBackend {
    pub fn affine(
        &self,
        input_buffer_id: Id<Buffer>,
        output_buffer_id: Id<Buffer>,
        mul: f32,
        add: f32,
    ) -> WgpuBackendResult<()> {
        self.run_shader_with_input(
            Shader::Affine,
            "affine",
            input_buffer_id,
            output_buffer_id,
            &AffineParams { mul, add },
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::WgpuBackend;

    #[test]
    fn test_affine() {
        let backend = WgpuBackend::new().unwrap();
        let input_buffer = backend
            .create_buffer_with_data(&[1.0f32, 2.007f32])
            .unwrap();
        let output_buffer = backend.create_buffer(2 * 4).unwrap();

        backend
            .affine(input_buffer, output_buffer, 2.4, 5.0)
            .unwrap();

        let result = backend.read_buf_as::<f32>(output_buffer).unwrap();

        assert_eq!(result, [7.4f32, 9.8168]);
    }
}
