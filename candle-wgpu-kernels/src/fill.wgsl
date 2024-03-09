
const s1 = vec3(u32(13), u32(19), u32(12));
const s2 = vec3(u32(2), u32(25), u32(4));
const s3 = vec3(u32(3), u32(11), u32(17));

const phi = array(
    u32(0x9E3779B9),
    u32(0xF39CC060),
    u32(0x1082276B),
    u32(0xF86C6A11)
);

// used for conversion between int and float
const unif01_norm32 = f32(4294967295);
const unif01_inv32 = f32(2.328306436538696289e-10);

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
    buffer[gid.x] = f32(gid.x);
}

@compute @workgroup_size(64)
fn rand_uniform_f32(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let diff = abs(input.min - input.max);
    let seed = vec4(input.seed, gid.x, 1, 1);
    buffer[gid.x] = f32(rand(rng(seed))) * unif01_inv32 * diff + input.min;
}

fn rand(rng: u32) -> u32 {
        let z1 = taus(rng, s1, u32(429496729));
        let z2 = taus(rng, s2, u32(4294967288));
        let z3 = taus(rng, s3, u32(429496280));
        let z4 = lcg(rng);

        let s = z1^z2^z3^z4;

        return s;
}

fn taus(z: u32, s: vec3<u32>, M: u32) -> u32 {
    let b: u32 = (((z << s.x) ^ z) >> s.y);
    return (((z & M) << s.z) ^ b);
}

fn lcg(z: u32) -> u32 {
    return (1664525 * z + u32(1013904223));
}

fn rng(seed_in: vec4<u32>) -> u32 {
    let seed: vec4<u32> = vec4(seed_in * vec4(phi[0], phi[1], phi[2], phi[3]) * u32(1099087573));
    
    var z1 = taus(seed.x, s1, u32(4294967294));
    var z2 = taus(seed.y, s2, u32(4294967288));
    var z3 = taus(seed.z, s3, u32(4294967280));
    var z4 = lcg(seed.x);

    var r1 = z1^z2^z3^z4^seed.y;
    z1 = taus(r1, s1, u32(4294967294));
    z2 = taus(r1, s2, u32(4294967288));
    z3 = taus(r1, s3, u32(4294967280));
    z4 = lcg(r1);

    r1 = z1^z2^z3^z4^seed.z;
    z1 = taus(r1, s1, u32(4294967294));
    z2 = taus(r1, s2, u32(4294967288));
    z3 = taus(r1, s3, u32(4294967280));
    z4 = lcg(r1);

    r1 = z1^z2^z3^z4^seed.w;
    z1 = taus(r1, s1, u32(4294967294));
    z2 = taus(r1, s2, u32(4294967288));
    z3 = taus(r1, s3, u32(4294967280));
    z4 = lcg(r1);

    return z1^z2^z3^z4;
}