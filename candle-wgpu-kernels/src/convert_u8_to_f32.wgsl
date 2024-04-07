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
    let input_offset = gid.x / 4u;
    let shift = gid.x % 4u;
    let shift_mask = shift * 8u;
    let shift_val = (3u - shift) * 8u;
    let mask = u32(0xFF000000) >> shift_mask;
    let val = (input[input_offset] & mask) >> shift_val;

    output[gid.x] = f32(val);
}
