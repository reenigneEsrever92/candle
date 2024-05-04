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

fn repeat2(dev: &Device) -> Result<()> {
    let input = Tensor::from_slice(
        &[
            [[
                0f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]; 13]
                .concat(),
            [
                [0f32; 13],
                [1f32; 13],
                [2f32; 13],
                [3f32; 13],
                [4f32; 13],
                [5f32; 13],
                [6f32; 13],
                [7f32; 13],
                [8f32; 13],
                [9f32; 13],
                [10f32; 13],
                [11f32; 13],
                [12f32; 13],
            ]
            .concat(),
        ]
        .concat(),
        (169, 2),
        dev,
    )?;

    let result_old = input.repeat_old((1, 3))?;
    let result = input.repeat((1, 3))?;
    assert_eq!(result.to_vec2::<f32>()?, result_old.to_vec2::<f32>()?);

    Ok(())
}

test_device!(repeat, repeat_cpu, repeat_gpu, repeat_metal, repeat_wgpu);
test_device!(
    repeat2,
    repeat2_cpu,
    repeat2_gpu,
    repeat2_metal,
    repeat2_wgpu
);
