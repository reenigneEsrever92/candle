struct BinaryOpParams {
    dummy: u32
}

@group(0) @binding(0)
var<uniform> params: BinaryOpParams;

@group(0) @binding(1)
var<storage, read> lhs: array<f32>;

@group(0) @binding(2)
var<storage, read> rhs: array<f32>;

@group(0) @binding(3)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn add(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = lhs[gid.x] + rhs[gid.x];
}

@compute @workgroup_size(64)
fn sub(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = lhs[gid.x] - rhs[gid.x];
}

@compute @workgroup_size(64)
fn mul(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = lhs[gid.x] * rhs[gid.x];
}

@compute @workgroup_size(64)
fn div(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = lhs[gid.x] / rhs[gid.x];
}

@compute @workgroup_size(64)
fn max(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    if (lhs[gid.x] > rhs[gid.x]) {
        output[gid.x] = lhs[gid.x];
    } else {
        output[gid.x] = rhs[gid.x];
    }
}
