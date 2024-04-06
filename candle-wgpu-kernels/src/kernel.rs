const CONV_SHADER: &str = include_str!("conv.wgsl");
const FILL_SHADER: &str = include_str!("fill.wgsl");
const FILL_SHADER_U8: &str = include_str!("fill_u8.wgsl");
const RANDOM_SHADER: &str = include_str!("random.wgsl");
const COPY_SHADER: &str = include_str!("copy.wgsl");

#[derive(Debug, Clone, Default)]
pub struct Kernels;

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum Shader {
    Conv,
    Fill,
    FillU8,
    Random,
    Copy,
}

impl Kernels {
    pub(crate) fn get_shader(&self, shader: Shader) -> &'static str {
        match shader {
            Shader::Conv => CONV_SHADER,
            Shader::Fill => FILL_SHADER,
            Shader::FillU8 => FILL_SHADER_U8,
            Shader::Random => RANDOM_SHADER,
            Shader::Copy => COPY_SHADER,
        }
    }
}
