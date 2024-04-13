struct UnaryOpParams {
    dummy: u32
}

@group(0) @binding(0)
var<uniform> params: UnaryOpParams;

@group(0) @binding(1)
var<storage, read> lhs: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn sqrt_f32(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = sqrt(lhs[gid.x]);
}

@compute @workgroup_size(64)
fn neg(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = lhs[gid.x] * f32(-1);
}

@compute @workgroup_size(64)
fn exp_f32(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = exp(lhs[gid.x]);
}

@compute @workgroup_size(64)
fn recip(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = 1.0 / lhs[gid.x];
}
