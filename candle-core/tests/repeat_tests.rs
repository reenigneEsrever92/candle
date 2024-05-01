use anyhow::Result;
use candle_core::{test_device, test_utils, Device, IndexOp, Tensor};

fn repeat(dev: &Device) -> Result<()> {
    let input = Tensor::new(&[1f32, 2.0, 3.0], dev)?;

    let expected = Tensor::new(&[1f32, 2.0, 3.0, 1.0, 2.0, 3.0], dev)?;
    let result = input.repeat(2)?;
    assert_eq!(result.to_vec1::<f32>()?, expected.to_vec1::<f32>()?);

    let input = Tensor::new(&[[1f32, 2.0], [3.0, 4.0]], dev)?;
    let expected = Tensor::new(
        &[
            [1f32, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            [3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            [3.0, 4.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
        ],
        dev,
    )?;
    let result = input.repeat((2, 4))?;
    assert_eq!(result.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);

    Ok(())
}

test_device!(repeat, repeat_cpu, repeat_gpu, repeat_metal, repeat_wgpu);
