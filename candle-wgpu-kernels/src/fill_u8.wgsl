struct Input {
    dummy: u32
}

@group(0) @binding(0)
var<uniform> input: Input;

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(64)
fn zeroes(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = u32(0);
}

@compute @workgroup_size(64)
fn ones(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    output[gid.x] = u32(0x01010101);
}
