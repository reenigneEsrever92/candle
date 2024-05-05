use wgpu::{Device, ShaderModule};

const CONV_SHADER: &str = include_str!("conv.wgsl");
const FILL_SHADER: &str = include_str!("fill.wgsl");
const FILL_SHADER_U8: &str = include_str!("fill_u8.wgsl");
const FILL_SHADER_U32: &str = include_str!("fill_u32.wgsl");
const RANDOM_SHADER: &str = include_str!("random.wgsl");
const COPY_SHADER: &str = include_str!("copy.wgsl");
const COPY_SPARSE_SHADER: &str = include_str!("copy_sparse.wgsl");
const CONVERT_U8_TO_F32: &str = include_str!("convert_u8_to_f32.wgsl");
const CONVERT_U32_TO_F32: &str = include_str!("convert_u32_to_f32.wgsl");
const AFFINE: &str = include_str!("affine.wgsl");
const BINARY_OP: &str = include_str!("binary_op.wgsl");
const UNARY_OP: &str = include_str!("unary_op.wgsl");
const UPSAMPLE_NEAREST: &str = include_str!("upsample_nearest.wgsl");

#[derive(Debug)]
pub struct Kernels {
    pub(crate) conv: ShaderModule,
    pub(crate) fill: ShaderModule,
    pub(crate) fill_u8: ShaderModule,
    pub(crate) fill_u32: ShaderModule,
    pub(crate) random: ShaderModule,
    pub(crate) copy: ShaderModule,
    pub(crate) copy_sparse: ShaderModule,
    pub(crate) convert_u8_to_f32: ShaderModule,
    pub(crate) convert_u32_to_f32: ShaderModule,
    pub(crate) affine: ShaderModule,
    pub(crate) binary_op: ShaderModule,
    pub(crate) unary_op: ShaderModule,
    pub(crate) uspample_nearest: ShaderModule,
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum Shader {
    Conv,
    Fill,
    FillU8,
    FillU32,
    Random,
    Copy,
    CopySparse,
    ConvertU8ToF32,
    ConvertU32ToF32,
    Affine,
    BinaryOp,
    UnaryOp,
    UpsampleNearest,
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
            fill_u32: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(FILL_SHADER_U32)),
            }),
            random: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(RANDOM_SHADER)),
            }),
            copy: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(COPY_SHADER)),
            }),
            copy_sparse: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(COPY_SPARSE_SHADER)),
            }),
            convert_u8_to_f32: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(CONVERT_U8_TO_F32)),
            }),
            convert_u32_to_f32: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(CONVERT_U32_TO_F32)),
            }),
            affine: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(AFFINE)),
            }),
            binary_op: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(BINARY_OP)),
            }),
            unary_op: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(UNARY_OP)),
            }),
            uspample_nearest: device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(UPSAMPLE_NEAREST)),
            }),
        }
    }
}
