struct Params {
    stride: u32
};

@group(0) @binding(0) 
var<uniform> params: Params;

@group(0) @binding(1) 
var<storage, read_write> input: array<f32>;

@group(0) @binding(2) 
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn conv1d(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = 1.0;
}