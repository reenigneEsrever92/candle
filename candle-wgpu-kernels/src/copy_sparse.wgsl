struct CopySparseParams {
    batch_size: u32,
    out_stride: u32
}

@group(0) @binding(0)
var<uniform> params: CopySparseParams;

@group(0) @binding(1)
var<storage, read> input: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn copy_sparse(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_no = gid.x / params.batch_size;
    let batch_offset = gid.x % params.batch_size;
    output[batch_no * params.out_stride + batch_offset] = input[gid.x];
}
