struct CopyParams {
    input_w: u32,
    input_h: u32,
    stride_x: u32,
    stride_y: u32,
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
    let out_w = (params.input_w + 1) / params.stride_x;

    let row = gid.x / out_w;
    let col = gid.x % out_w;

    output[gid.x] = input[row * params.stride_y * params.input_w + col * params.stride_x];
}

@compute @workgroup_size(64)
fn ones(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = 1.0;
}
