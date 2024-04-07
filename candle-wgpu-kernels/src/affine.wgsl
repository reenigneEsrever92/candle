struct AffineParams {
    mul: f32,
    add: f32,
}

@group(0) @binding(0)
var<uniform> params: AffineParams;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn affine(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = input[gid.x] * params.mul + params.add;
}
