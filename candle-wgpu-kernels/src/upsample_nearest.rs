use crate::kernel::Shader;
use crate::{WgpuBackend, WgpuBackendResult};
use bytemuck::{Pod, Zeroable};
use wgpu::core::id::BufferId;
use wgpu::{Buffer, Id};

#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct UpsampleNearestParams {
    no_batches: u32,
    no_channels: u32,
    input_w: u32,
    input_h: u32,
    output_w: u32,
    output_h: u32,
}

impl WgpuBackend {
    #[allow(clippy::too_many_arguments)]
    pub fn upsample_nearest(
        &self,
        no_batches: u32,
        no_channels: u32,
        input_w: u32,
        input_h: u32,
        output_w: u32,
        output_h: u32,
        input: Id<Buffer>,
        output: Id<Buffer>,
    ) -> WgpuBackendResult<()> {
        let params = UpsampleNearestParams {
            no_batches,
            no_channels,
            input_w,
            input_h,
            output_w,
            output_h,
        };

        self.run_shader_with_input(
            Shader::UpsampleNearest,
            "upsample_nearest",
            input,
            output,
            &params,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::WgpuBackend;

    #[test]
    fn test_upsample_nearest_2d() {
        let backend = WgpuBackend::new().unwrap();
        let input = backend
            .create_buffer_with_data(&[1.0f32, 2.0, 3.0, 4.0])
            .unwrap();
        let output = backend.create_buffer(4 * 4 * 4).unwrap();

        backend
            .upsample_nearest(1, 1, 2, 2, 4, 4, input, output)
            .unwrap();

        let result = backend.read_buf_as::<f32>(output).unwrap();

        assert_eq!(
            result,
            [1f32, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0]
        );
    }
}
