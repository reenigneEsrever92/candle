struct RepeatParams {
    batch_size: u32,
    out_batch_size: u32,
}

@group(0) @binding(0)
var<uniform> params: RepeatParams;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn repeat(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let input_offset = gid.x / params.out_batch_size * params.batch_size + gid.x % params.batch_size;
    output[gid.x] = input[input_offset];
}
