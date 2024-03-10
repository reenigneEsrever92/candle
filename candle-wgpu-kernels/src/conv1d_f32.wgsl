struct Params {
    stride: u32
};

@group(0) @binding(0) 
var<uniform> params: Params;

@group(0) @binding(1) 
var<storage, read_write> input: array<f32>;

@group(0) @binding(2) 
var<storage, read_write> output: array<f32>;

