use wgpu::{Device, ShaderModule};

const CONV_SHADER: &str = include_str!("conv.wgsl");
const FILL_SHADER: &str = include_str!("fill.wgsl");
const FILL_SHADER_U8: &str = include_str!("fill_u8.wgsl");
const RANDOM_SHADER: &str = include_str!("random.wgsl");
const COPY_SHADER: &str = include_str!("copy.wgsl");

#[derive(Debug)]
pub struct Kernels {
    pub(crate) conv: ShaderModule,
    pub(crate) fill: ShaderModule,
    pub(crate) fill_u8: ShaderModule,
    pub(crate) random: ShaderModule,
    pub(crate) copy: ShaderModule,
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum Shader {
    Conv,
    Fill,
    FillU8,
    Random,
    Copy,
}

impl Kernels {
    pub fn new(device: &Device) -> Self {
        Self {
            conv: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(CONV_SHADER)),
            }),
            fill: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(FILL_SHADER)),
            }),
            fill_u8: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(FILL_SHADER_U8)),
            }),
            random: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(RANDOM_SHADER)),
            }),
            copy: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(COPY_SHADER)),
            }),
        }
    }
}
