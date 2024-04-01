struct Params {
    batch_size: u32,
    channels_in: u32,
    input_w: u32,
    input_h: u32,
    channels_out: u32,
    kernel_w: u32,
    kernel_h: u32,
    groups: u32,
    padding_x: u32,
    padding_y: u32,
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
fn conv2d(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    // TODO move calculations from shader to backend
    let i_cs = params.input_w * params.input_h;
    let i_bs = i_cs * params.channels_in;

    let out_w = params.input_w - (params.kernel_w - 1) * params.dilation + params.padding_x * 2;
    let out_h = params.input_h - (params.kernel_h - 1) * params.dilation + params.padding_y * 2;
    let out_channel_size = out_w * out_h;

    let out_batch_size = out_channel_size * params.channels_out;

    let out_batch = gid.x / out_batch_size;
    let out_channel = gid.x % out_batch_size / out_channel_size;

    let k_cs = params.kernel_w * params.kernel_h;
    let k_bs = k_cs * params.channels_in;

    let col_offset = gid.x % (out_batch_size * out_batch + out_channel * out_channel_size) / out_w - params.padding_y;
    let row_offset = gid.x % out_w - params.padding_x;

    for(var c_in = u32(0); c_in < params.channels_in; c_in++) {
        for(var y = u32(0); y < params.kernel_h; y++) {
            for(var x = u32(0); x < params.kernel_w; x++) {
                var value = f32(0);

                let kernel_offset = out_channel * k_bs + c_in * k_cs + y * params.kernel_w + x;
                let input_offset = out_batch * i_bs + c_in * i_cs + col_offset * out_w + y * params.dilation * params.input_w + x * params.dilation + row_offset;

                if(!(col_offset + y < 0 || row_offset + x < 0 || col_offset + y >= params.input_h || row_offset + x >= params.input_w)) {
                    value = input[input_offset];
                }

                output[gid.x] += kernel[kernel_offset] * value;
            }
        }
    }
}