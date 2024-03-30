use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, Id};

use crate::{kernel::Shader, WgpuBackend, WgpuBackendResult};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct WgpuConvParams {
    pub batch_size: u32,
    pub channels_in: u32,
    pub input_w: u32,
    pub input_h: u32,
    pub channels_out: u32,
    pub kernel_w: u32,
    pub kernel_h: u32,
    pub groups: u32,
    pub padding: u32,
    pub stride: u32,
    pub dilation: u32,
}

impl Default for WgpuConvParams {
    fn default() -> Self {
        Self {
            batch_size: 1,
            input_w: 1,
            input_h: 1,
            channels_in: 1,
            channels_out: 1,
            kernel_w: 1,
            kernel_h: 1,
            groups: 1,
            padding: 0,
            stride: 1,
            dilation: 1,
        }
    }
}

impl WgpuBackend {
    // pub fn conv1d(
    //     &mut self,
    //     input: Id<Buffer>,
    //     params: &WgpuConvParams,
    // ) -> WgpuBackendResult<Id<Buffer>> {
    //     self.run_shader_with_input(Shader::Conv, input, 8, params)
    // }

    pub fn conv2d(
        &self,
        input: Id<Buffer>,
        kernel: Id<Buffer>,
        params: &WgpuConvParams,
    ) -> WgpuBackendResult<Id<Buffer>> {
        let out_w = params.input_w - (params.kernel_w - 1) * params.dilation + params.padding * 2;
        let out_h = params.input_h - (params.kernel_h - 1) * params.dilation + params.padding * 2;

        let output_buffer_size = params.batch_size * params.channels_out * out_h * out_w * 4; // 4 bytes for f32

        self.run_shader_with_input(
            Shader::Conv,
            input,
            kernel,
            output_buffer_size as u64,
            params,
        )
    }
}

#[cfg(test)]
mod test {
    use crate::WgpuBackend;

    use super::WgpuConvParams;

    #[test]
    fn test_conv2d() {
        // dims: (1, 1, 5, 5)
        let tensor: [f32; 25] = [2.0; 25];
        // dims: (1, 3, 3)
        let kernel = [0.5f32, 1.0, 0.5, 1.0, 2.0, 1.0, 0.5, 1.0, 0.5];
        // dims: (1, 1, 3, 3)
        let expected = [16f32; 9];

        let backend = WgpuBackend::new().unwrap();
        let input_buffer = backend.create_buffer_with_data(&tensor).unwrap();
        let kernel_buffer = backend.create_buffer_with_data(&kernel).unwrap();
        let result_buffer = backend
            .conv2d(
                input_buffer,
                kernel_buffer,
                &WgpuConvParams {
                    input_w: 5,
                    input_h: 5,
                    kernel_w: 3,
                    kernel_h: 3,
                    ..Default::default()
                },
            )
            .unwrap();
        let result = backend.read_buf_as::<f32>(result_buffer).unwrap();

        println!("result: {result:?}");

        assert_eq!(result.len(), 9);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_conv1d() {
        // dims (1, 4, 5, 1)
        let tensor = [
            0.4f32, -0.8, -0.1, -1.6, 1.2, -0.9, -1.7, 0.1, 0.2, 0.4, 1.8, -0.2, 2.2, -0.7, 0.2,
            1.3, -0.7, 0.3, -1.0, 0.6,
        ];
        // dims (2, 4, 3, 1)
        let kernel = [
            -0.8f32, -0.3, 0.0, 1.3, 0.2, -1.9, 1.4, 0.9, 0.9, -0.7, -1.1, 0.9, -1.2, 1.8, 1.6,
            -1.4, 0.4, -1.2, 0.7, 1.6, -0.6, -0.1, -1.4, 0.6,
        ];
        // dims (1, 2, 3, 1)
        let expected = [1.95, -2.55, 4.03];

        let backend = WgpuBackend::new().unwrap();
        let input_buffer = backend.create_buffer_with_data(&tensor).unwrap();
        let kernel_buffer = backend.create_buffer_with_data(&kernel).unwrap();
        let output_buffer = backend
            .conv2d(
                input_buffer,
                kernel_buffer,
                &WgpuConvParams {
                    input_w: 4,
                    input_h: 5,
                    kernel_w: 3,
                    kernel_h: 3,
                    ..Default::default()
                },
            )
            .unwrap();
        let contents = backend.read_buf_as::<f32>(output_buffer).unwrap();

        assert_eq!(contents.len(), 6);
        assert_eq!(contents[0..=5], expected);
    }
}
