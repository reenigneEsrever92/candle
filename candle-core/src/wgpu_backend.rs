use crate::{
    backend::{BackendDevice, BackendStorage},
    layout::Layout,
    CpuStorage, DType, Result,
};

#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for WgpuError {
    fn from(e: String) -> Self {
        WgpuError::Message(e)
    }
}

#[derive(Clone)]
pub struct WgpuDevice {}

impl BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;

    fn new(_ordinal: usize) -> Result<Self> {
        Ok(Self {})
    }

    fn location(&self) -> crate::DeviceLocation {
        todo!()
    }

    fn same_device(&self, _: &Self) -> bool {
        todo!()
    }

    fn zeros_impl(&self, _shape: &crate::Shape, _dtype: DType) -> Result<Self::Storage> {
        todo!()
    }

    fn ones_impl(&self, _shape: &crate::Shape, _dtype: DType) -> Result<Self::Storage> {
        todo!()
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        todo!()
    }

    fn rand_uniform(&self, _: &crate::Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        todo!()
    }

    fn rand_normal(&self, _: &crate::Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        todo!()
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        todo!()
    }
}

impl std::fmt::Debug for WgpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

#[derive(Debug, Clone)]
pub struct WgpuStorage;

impl BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn dtype(&self) -> DType {
        todo!()
    }

    fn device(&self) -> &Self::Device {
        todo!()
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        todo!()
    }

    fn affine(&self, _: &crate::Layout, _: f64, _: f64) -> crate::Result<Self> {
        todo!()
    }

    fn powf(&self, _: &crate::Layout, _: f64) -> crate::Result<Self> {
        todo!()
    }

    fn elu(&self, _: &crate::Layout, _: f64) -> crate::Result<Self> {
        todo!()
    }

    fn reduce_op(
        &self,
        _: crate::op::ReduceOp,
        _: &crate::Layout,
        _: &[usize],
    ) -> crate::Result<Self> {
        todo!()
    }

    fn cmp(
        &self,
        _: crate::op::CmpOp,
        _: &Self,
        _: &crate::Layout,
        _: &crate::Layout,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn to_dtype(&self, _: &crate::Layout, _: DType) -> crate::Result<Self> {
        todo!()
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, _: &crate::Layout) -> crate::Result<Self> {
        todo!()
    }

    fn binary_impl<B: crate::op::BinaryOpT>(
        &self,
        _: &Self,
        _: &crate::Layout,
        _: &crate::Layout,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn where_cond(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConv1D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv_transpose1d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv2d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConv2D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn conv_transpose2d(
        &self,
        _l: &crate::Layout,
        _kernel: &Self,
        _kernel_l: &crate::Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn avg_pool2d(
        &self,
        _: &crate::Layout,
        _: (usize, usize),
        _: (usize, usize),
    ) -> crate::Result<Self> {
        todo!()
    }

    fn max_pool2d(
        &self,
        _: &crate::Layout,
        _: (usize, usize),
        _: (usize, usize),
    ) -> crate::Result<Self> {
        todo!()
    }

    fn upsample_nearest1d(&self, _: &crate::Layout, _: usize) -> crate::Result<Self> {
        todo!()
    }

    fn upsample_nearest2d(&self, _: &crate::Layout, _: usize, _: usize) -> crate::Result<Self> {
        todo!()
    }

    fn gather(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn scatter_add(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn index_select(
        &self,
        _: &Self,
        _: &crate::Layout,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn index_add(
        &self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: &Self,
        _: &crate::Layout,
        _: usize,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &crate::Layout,
        _: &crate::Layout,
    ) -> crate::Result<Self> {
        todo!()
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &crate::Layout) -> crate::Result<()> {
        todo!()
    }
}
