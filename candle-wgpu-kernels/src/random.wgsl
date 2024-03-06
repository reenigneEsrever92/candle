@compute @workgroup_size(64)
fn rand_f32(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let d = vec2f(0., 1.);
    let b = floor(n);
    let f = smoothStep(vec2f(0.), vec2f(1.), fract(n));
    return mix(mix(rand22(b), rand22(b + d.yx), f.x), mix(rand22(b + d.xy), rand22(b + d.yy), f.x), f.y);
}