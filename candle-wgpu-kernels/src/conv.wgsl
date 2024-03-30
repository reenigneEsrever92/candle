struct Params {
    batch_size: u32,
    channels_in: u32,
    input_w: u32,
    input_h: u32,
    channels_out: u32,
    kernel_w: u32,
    kernel_h: u32,
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
var<storage, read> kernel: array<f32>;

@group(0) @binding(3) 
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn conv1d(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // new tensor shape: (batch_size, channels_out, tensor_w, tensor_h)

    let k_cs = params.kernel_w * params.kernel_h;
    let k_bs = k_cs * params.channels_out;

    let i_cs = params.input_w * params.input_h;

    let ci = gid.x / k_bs;

    let cs = params.input_w * params.input_h;
    let bs = params.channels_in * cs;

    let bi = gid.x / bs;

    let row = gid.x / params.input_w;
    let row_offset = params.input_w * params.channels_in * params.batch_size;
    let global_offset = row * row_offset;

    for(var c_in = u32(0); c_in < params.channels_in; c_in++) {
        for(var x = u32(0); x < params.kernel_w; x++) {
            for(var y = u32(0); y < params.kernel_h; y++) {
                let kernel_offset = c_in * k_cs + y * params.kernel_w + x;
                let input_offset = c_in * i_cs + y * params.input_w + x;
                let weight = kernel[kernel_offset];
                let value = weight * input[input_offset];
                output[gid.x] = output[gid.x] + value;
            }
        }
    }
}