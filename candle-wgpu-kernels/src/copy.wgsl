struct CopyParams {
    input_w: u32,
    input_h: u32,
    input_stride_x: u32,
    input_stride_y: u32,
    output_stride_x: u32,
    output_stride_y: u32
}

@group(0) @binding(0)
var<uniform> params: CopyParams;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn copy(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let input_offset = gid.x * params.input_stride % params.input_w
    output[gid.x] = 0.0;
}

@compute @workgroup_size(64)
fn ones(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = 1.0;
}
