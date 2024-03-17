struct Params {
    batch_size: u32,
    channels_in: u32,
    input_size_w: u32,
    input_size_h: u32,
    channels_out: u32,
    kernel_size_w: u32,
    kernel_size_h: u32,
    groups: u32,
    padding: u32,
    stride: u32,
    dilation: u32,
};

@group(0) @binding(0) 
var<uniform> params: Params;

@group(0) @binding(1) 
var<storage, read> input: array<f32>;

@group(0) @binding(2) 
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn conv1d(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // new tensor shape: (batch_size, channels_out, tensor_w, tensor_h)
    let tid = gid.x;
    let output_index = tid / params.batch_size + tid;
    let offset_x = (params.kernel_size_w - 1) * params.dilation / 2;
    let offset_y = (params.kernel_size_h - 1) * params.dilation / 2;


     if(gid.x == 0) {
        output[tid] = f32(offset_x);
    } else if (gid.x == 1) {
        output[tid] = f32(offset_y);
    }
}