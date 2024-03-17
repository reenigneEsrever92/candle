#[derive(Debug, Clone, Default)]
pub struct Kernels;

const CONV_SHADER: &str = include_str!("conv.wgsl");

pub enum Shader {
    Conv,
}

impl Kernels {
    pub(crate) fn get_shader(&self, shader: Shader) -> &str {
        match shader {
            Shader::Conv => CONV_SHADER,
        }
    }
}
