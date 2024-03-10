use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, Id};

use crate::{WgpuBackend, WgpuBackendResult};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct WgpuConv1DParams {
    batch_size: u32,
    layers_in: u32,
    conv_in: u32,
    conv_out: u32,
    kernel_size: u32,
    padding: u32,
    stride: u32,
    dilation: u32,
}

impl WgpuBackend {
    pub fn conv1d(
        &self,
        input: Id<Buffer>,
        params: &WgpuConv1DParams,
    ) -> WgpuBackendResult<Id<Buffer>> {
        // smol::block_on(async {})
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::WgpuBackend;

    #[test]
    fn test_conv1d() {
        // dims (1, 4, 5)
        let tensor = &[
            0.4056f32, -0.8689, -0.0773, -1.5630, 1.2279, -0.9287, -1.7030, 0.1370, 0.1866, 0.4145,
            1.8025, -0.1536, 2.2013, -0.6836, 0.2477, 1.3127, -0.6957, 0.3278, -1.0124, 0.5599,
        ];
        // dims (2, 3, 4)
        let kernel = &[
            -0.8404f32, -0.3490, 0.0130, 1.3123, 0.1763, -1.9249, 1.4270, 0.9421, 0.8670, -0.7181,
            -1.1111, 0.8869, -1.2429, 1.8357, 1.6052, -1.3844, 0.3951, -1.2036, 0.6686, 1.6261,
            -0.6451, -0.0840, -1.4247, 0.5512,
        ];
        // dims (1, 2, 5)
        let expected = [
            2.4509315, 2.6357481, -1.3335553, 4.1392756, 0.56572014, 1.809062, -1.1783935,
            3.567513, 0.5069167, 3.3352304,
        ];

        let backend = WgpuBackend::new().unwrap();
        let buffer_id = backend.create_buffer_with_data(tensor);
        let result = backend.conv1d(input);
    }
}
