use crate::kernel::Shader;
use crate::{WgpuBackend, WgpuBackendResult};
use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, Id};

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
struct CopySparseParams {
    batch_size: u32,
    out_stride: u32,
}

impl WgpuBackend {
    pub fn copy_sparse(
        &self,
        input_buffer: Id<Buffer>,
        output_buffer: Id<Buffer>,
        batch_size: u32,
        out_stride: u32,
    ) -> WgpuBackendResult<()> {
        self.run_shader_with_input(
            Shader::CopySparse,
            "copy_sparse",
            input_buffer,
            output_buffer,
            &CopySparseParams {
                batch_size,
                out_stride,
            },
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::WgpuBackend;

    #[test]
    fn test_copy_sparse() {
        let backend = WgpuBackend::new().unwrap();
        let input_buffer = backend
            .create_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let output_buffer = backend.create_buffer(8 * 4).unwrap();

        backend
            .copy_sparse(input_buffer, output_buffer, 2, 4)
            .unwrap();

        let result = backend.read_buf_as::<f32>(output_buffer).unwrap();

        assert_eq!(result, [1.0f32, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]);
    }
}
