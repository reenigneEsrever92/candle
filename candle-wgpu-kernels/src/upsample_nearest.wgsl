struct Params {
    no_batches: u32,
    no_channels: u32,
    input_w: u32,
    input_h: u32,
    output_w: u32,
    output_h: u32,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn upsample_nearest(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let out_x = gid.x % params.output_w;
    let out_y = gid.x / params.output_w;

    let in_x = u32(round_2(f32(params.input_w) / f32(params.output_w) * f32(out_x)));
    let in_y = u32(round_2(f32(params.input_h) / f32(params.output_h) * f32(out_y)));

    let offset = in_y * params.input_w + in_x;

    output[gid.x] = input[offset];
}

fn round_2(val: f32) -> f32 {
    if(val - floor(val) <= 0.5) {
        return floor(val);
    } else {
        return ceil(val);
    }
}
