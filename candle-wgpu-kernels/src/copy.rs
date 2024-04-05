use crate::kernel::Shader;
use crate::{WgpuBackend, WgpuBackendResult};
use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, Id};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CopyParams {
    input_w: u32,
    input_h: u32,
    stride_x: u32,
    stride_y: u32,
}

impl WgpuBackend {
    pub fn copy(
        &self,
        input_buffer_id: Id<Buffer>,
        output_buffer_id: Id<Buffer>,
        input_w: u32,
        input_h: u32,
        stride_x: u32,
        stride_y: u32,
    ) -> WgpuBackendResult<()> {
        let params = CopyParams {
            input_w,
            input_h,
            stride_x,
            stride_y,
        };

        self.run_shader_with_input(
            Shader::Copy,
            "copy",
            input_buffer_id,
            output_buffer_id,
            &params,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::WgpuBackend;

    #[test]
    fn test_copy() {
        let backend = WgpuBackend::new().unwrap();
        let input = [
            1f32, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let input_buffer = backend.create_buffer_with_data(&input).unwrap();
        let output_buffer = backend.create_buffer(3 * 3 * 4).unwrap();

        backend
            .copy(input_buffer, output_buffer, 5, 5, 2, 2)
            .unwrap();

        let result = backend.read_buf_as::<f32>(output_buffer).unwrap();

        assert_eq!(result.len(), 9);
        assert_eq!(result, [1f32; 9]);
    }

    #[test]
    fn test_copy_2() {
        let backend = WgpuBackend::new().unwrap();
        let input = [
            1f32, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ];
        let input_buffer = backend.create_buffer_with_data(&input).unwrap();
        let output_buffer = backend.create_buffer(3 * 3 * 4).unwrap();

        backend
            .copy(input_buffer, output_buffer, 6, 5, 2, 2)
            .unwrap();

        let result = backend.read_buf_as::<f32>(output_buffer).unwrap();

        assert_eq!(result.len(), 9);
        assert_eq!(result, [1f32; 9]);
    }

    #[test]
    fn test_copy_3() {
        let backend = WgpuBackend::new().unwrap();
        let input = [
            1f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0,
        ];
        let input_buffer = backend.create_buffer_with_data(&input).unwrap();
        let output_buffer = backend.create_buffer(3 * 3 * 4).unwrap();

        backend
            .copy(input_buffer, output_buffer, 6, 3, 2, 1)
            .unwrap();

        let result = backend.read_buf_as::<f32>(output_buffer).unwrap();

        assert_eq!(result.len(), 9);
        assert_eq!(result, [1f32; 9]);
    }
}
