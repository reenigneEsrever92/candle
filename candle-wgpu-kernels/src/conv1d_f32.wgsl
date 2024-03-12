struct Params {
    batch_size: u32,
    input_length: u32,
    channels_in: u32,
    channels_out: u32,
    kernel_size: u32,
    groups: u32,
    padding: u32,
    stride: u32,
    dilation: u32,
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
     if(gid.x == 1) {
        output[gid.x] = f32(params.batch_size);
    } else {
        output[gid.x] = f32(params.channels_in);
    }
}