use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, Id};

use crate::{kernel::Shader, WgpuBackend, WgpuBackendResult};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct WgpuConvParams {
    batch_size: u32,
    channels_in: u32,
    input_size: [u32; 2],
    channels_out: u32,
    kernel_size: [u32; 2],
    groups: u32,
    padding: u32,
    stride: u32,
    dilation: u32,
}

impl Default for WgpuConvParams {
    fn default() -> Self {
        Self {
            batch_size: 1,
            input_size: [1, 1],
            channels_in: 1,
            channels_out: 1,
            kernel_size: [1, 1],
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
        &mut self,
        input: Id<Buffer>,
        kernel: Id<Buffer>,
        params: &WgpuConvParams,
    ) -> WgpuBackendResult<Id<Buffer>> {
        let out_w = params.input_size[0] - (params.kernel_size[0] - 1) * params.dilation
            + params.padding * 2;
        let out_h = params.input_size[1] - (params.kernel_size[1] - 1) * params.dilation
            + params.padding * 2;

        let output_buffer_size = (out_w * out_h * params.batch_size * params.channels_out) * 4; // 4 bytes for f32

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
        let tensor: [f32; 25] = [2.0; 25];
        let kernel = [0.5f32, 1.0, 0.5, 1.0, 2.0, 1.0, 0.5, 1.0, 0.5];
        let expected = [16f32; 9];

        let mut backend = WgpuBackend::new().unwrap();
        let input_buffer = backend.create_buffer_with_data(&tensor).unwrap();
        let kernel_buffer = backend.create_buffer_with_data(&kernel).unwrap();
        let result_buffer = backend
            .conv2d(
                input_buffer,
                kernel_buffer,
                &WgpuConvParams {
                    input_size: [5, 5],
                    kernel_size: [3, 3],
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
        // // dims (1, 4, 5)
        // let tensor = &[
        //     0.4056f32, -0.8689, -0.0773, -1.5630, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866, 0.4145,
        //     1.8025, -0.1536, 2.2013, -0.6836, 0.2477, 1.3127, -0.6957, 0.3278, -1.0124, 0.5599,
        // ];
        // // dims (2, 4, 3)
        // let kernel = &[
        //     -0.8404f32, -0.3490, 0.0130, 1.3123, 0.1763, -1.9249, 1.4270, 0.9421, 0.8670, -0.7181,
        //     -1.1111, 0.8869, -1.2429, 1.8357, 1.6052, -1.3844, 0.3951, -1.2036, 0.6686, 1.6261,
        //     -0.6451, -0.0840, -1.4247, 0.5512,
        // ];
        // // dims (1, 2, 5)
        // let expected = [
        //     2.4509315, 2.6357481, -1.3335553, 4.1392756, 0.56572014, 1.809062, -1.1783935,
        //     3.567513, 0.5069167, 3.3352304,
        // ];

        // let mut backend = WgpuBackend::new().unwrap();
        // let buffer_id = backend.create_buffer_with_data(tensor).unwrap();
        // let result = backend
        //     .conv1d(
        //         buffer_id,
        //         &WgpuConvParams {
        //             input_size: [5, 1],  // tensor dim 3
        //             batch_size: 1,       // tensor dim 1
        //             channels_in: 4,      // kernel dim 2 - tensor dim 2
        //             channels_out: 2,     // kernel dim 1
        //             kernel_size: [3, 1], // kernel dim 3
        //             ..Default::default()
        //         },
        //     )
        //     .unwrap();
        // let contents = backend.read_buf(result).unwrap();
        // let floats: &[f32] = bytemuck::cast_slice(&contents);

        // assert_eq!(floats.len(), 10);
        // assert_eq!(floats, expected);
        // println!("Result: {floats:?}");
    }
}
