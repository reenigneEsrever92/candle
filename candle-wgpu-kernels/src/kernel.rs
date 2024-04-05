#[derive(Debug, Clone, Default)]
pub struct Kernels;

const CONV_SHADER: &str = include_str!("conv.wgsl");
const FILL_SHADER: &str = include_str!("fill.wgsl");
const RANDOM_SHADER: &str = include_str!("random.wgsl");
const COPY_SHADER: &str = include_str!("copy.wgsl");

pub enum Shader {
    Conv,
    Fill,
    Random,
    Copy,
}

impl Kernels {
    pub(crate) fn get_shader(&self, shader: Shader) -> &str {
        match shader {
            Shader::Conv => CONV_SHADER,
            Shader::Fill => FILL_SHADER,
            Shader::Random => RANDOM_SHADER,
            Shader::Copy => COPY_SHADER,
        }
    }
}
