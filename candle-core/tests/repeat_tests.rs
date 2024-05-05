use anyhow::Result;
use candle_core::{test_device, Device, Shape, Tensor};

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

    let input = Tensor::new(&[[1f32, 2.0], [3.0, 4.0]], dev)?;
    let expected = Tensor::new(
        &[[
            [1f32, 2.0, 1.0, 2.0],
            [3.0, 4.0, 3.0, 4.0],
            [1.0, 2.0, 1.0, 2.0],
            [3.0, 4.0, 3.0, 4.0],
            [1.0, 2.0, 1.0, 2.0],
            [3.0, 4.0, 3.0, 4.0],
        ]; 4],
        dev,
    )?;
    let result = input.repeat((4, 3, 2))?;
    assert_eq!(result.to_vec3::<f32>()?, expected.to_vec3::<f32>()?);

    let input = Tensor::new(&[[1f32, 2.0, 3.0], [4.0, 5.0, 6.0]], dev)?;

    let expected = Tensor::new(
        &[
            [1f32, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dev,
    )?;
    let result = input.repeat((2, 1))?;
    assert_eq!(result.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);

    let input = Tensor::new(&[1f32, 2.0], dev)?;

    let expected = Tensor::new(&[[1f32, 2.0], [1f32, 2.0], [1f32, 2.0], [1f32, 2.0]], dev)?;
    let result = input.repeat((4, 1))?;
    assert_eq!(result.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);

    Ok(())
}

test_device!(repeat, repeat_cpu, repeat_gpu, repeat_metal, repeat_wgpu);
