struct ConvertParams {
    dummy: u32
}

@group(0) @binding(0)
var<uniform> params: ConvertParams;

@group(0) @binding(1)
var<storage, read> input: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn convert(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = f32(input[gid.x]);
}
