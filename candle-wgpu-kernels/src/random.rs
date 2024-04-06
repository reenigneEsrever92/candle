use crate::kernel::Shader;
use crate::{WgpuBackend, WgpuBackendResult};
use bytemuck::{Pod, Zeroable};
use wgpu::{Buffer, Id};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RandUniformParams {
    seed: u32,
    min: f32,
    max: f32,
}

impl WgpuBackend {
    pub fn rand_uniform(
        &self,
        output_buffer_id: Id<Buffer>,
        params: &RandUniformParams,
    ) -> WgpuBackendResult<()> {
        self.run_shader(Shader::Random, "rand_uniform", output_buffer_id, params)?;

        Ok(())
    }
}

#[cfg(test)]
pub mod test {
    use crate::random::RandUniformParams;
    use crate::WgpuBackend;
    use std::time::Instant;

    #[test]
    fn test_fill_random() {
        let backend = WgpuBackend::new().unwrap();

        let params = RandUniformParams {
            min: -30f32,
            max: 30f32,
            seed: 299792458,
        };

        let output_buffer_size = 16_000_000 * 4;
        let output_buffer_id = backend.create_buffer(output_buffer_size as u64).unwrap();

        let start = Instant::now();
        backend.rand_uniform(output_buffer_id, &params).unwrap();
        println!("Took: {:?}", Instant::now().duration_since(start));

        let start = Instant::now();
        backend.rand_uniform(output_buffer_id, &params).unwrap();
        println!("Took: {:?}", Instant::now().duration_since(start));

        let result = backend.read_buf_as::<f32>(output_buffer_id).unwrap();

        assert_eq!(result.len(), 16_000_000);
        assert!(result
            .iter()
            .all(|v| { *v >= params.min && *v <= params.max }));

        // mean must approach zero given a large enough sample size
        let mean = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean >= -1.0);
        assert!(mean <= 1.0);
    }
}
