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
    let i_cs = params.input_w * params.input_h;
    let i_bs = i_cs * params.channels_in;

    let out_w = params.input_w - (params.kernel_w - 1) * params.dilation + params.padding * 2;
    let out_h = params.input_h - (params.kernel_h - 1) * params.dilation + params.padding * 2;
    let out_channel_size = out_w * out_h;

    let out_batch_size = out_channel_size * params.channels_out;

    let out_batch = gid.x / out_batch_size;
    let out_channel = gid.x % out_batch_size / out_channel_size;

    let k_cs = params.kernel_w * params.kernel_h;
    let k_bs = k_cs * params.channels_in;

    // FIXME
    let row_offset = gid.x % params.input_w;

    for(var c_in = u32(0); c_in < params.channels_in; c_in++) {
        for(var y = u32(0); y < params.kernel_h; y++) {
            for(var x = u32(0); x < params.kernel_w; x++) {
                let kernel_offset = out_channel * k_bs + c_in * k_cs + y * params.kernel_w + x;
                let input_offset = out_batch * i_bs + c_in * i_cs + y * params.dilation * params.input_w + x * params.dilation + row_offset;
                let weight = kernel[kernel_offset];
                let value = input[input_offset] * weight;

                output[gid.x] = f32(row_offset);
            }
        }
    }
}