

struct Input {
    seed: u32,
    min: f32,
    max: f32
}

@group(0) @binding(0) 
var<uniform> input: Input;
@group(0) @binding(1) 
var<storage, read_write> buffer: array<f32>;

@compute @workgroup_size(64)
fn ones(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    buffer[gid.x] = 1.0;
}

@compute @workgroup_size(64)
fn rand_uniform_f32(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let diff = abs(input.min - input.max);
    let seed = vec4(input.seed, gid.x, 1, 1);
    buffer[gid.x] = f32(rand(rng(seed))) * unif01_inv32 * diff + input.min;
}
