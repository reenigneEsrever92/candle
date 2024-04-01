use std::sync::Arc;

use candle_wgpu_kernels::{WgpuBackend, WgpuBackendError};
use wgpu::Id;

use crate::{
    backend::{BackendDevice, BackendStorage},
    dtype,
    layout::Layout,
    CpuStorage, DType, Result,
};

#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    #[error("{0}")]
    Message(String),
    #[error("Wgpu backend error: {0}")]
    WgpuBackendError(#[from] WgpuBackendError),
}

impl From<String> for WgpuError {
    fn from(e: String) -> Self {
        WgpuError::Message(e)
    }
}

#[derive(Clone)]
pub struct WgpuDevice {
    backend: WgpuBackend,
}

impl BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;

    fn new(_ordinal: usize) -> Result<Self> {
        let backend = WgpuBackend::new().map_err(WgpuError::from)?;
        Ok(Self { backend })
    }

    fn location(&self) -> crate::DeviceLocation {
        crate::DeviceLocation::Wgpu {
            gpu_id: self.backend.get_gpu_id(),
        }
    }

    fn same_device(&self, rhs: &Self) -> bool {
        self.backend.get_gpu_id() == rhs.backend.get_gpu_id()
    }

    fn zeros_impl(&self, _shape: &crate::Shape, _dtype: DType) -> Result<Self::Storage> {
        todo!()
    }

    fn ones_impl(&self, _shape: &crate::Shape, _dtype: DType) -> Result<Self::Storage> {
        todo!()
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let buffer = match storage {
            CpuStorage::U8(storage) => self.backend.create_buffer_with_data(storage),
            CpuStorage::U32(storage) => self.backend.create_buffer_with_data(storage),
            CpuStorage::I64(storage) => self.backend.create_buffer_with_data(storage),
            CpuStorage::BF16(storage) => self.backend.create_buffer_with_data(storage),
            CpuStorage::F16(storage) => self.backend.create_buffer_with_data(storage),
            CpuStorage::F32(storage) => self.backend.create_buffer_with_data(storage),
            CpuStorage::F64(storage) => self.backend.create_buffer_with_data(storage),
        }
        .map_err(WgpuError::from)?;

        Ok(Self::Storage::new(buffer, self.clone(), storage.dtype()))
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
pub struct WgpuStorage {
    id: Id<wgpu::Buffer>,
    device: WgpuDevice,
    dtype: DType,
}

impl WgpuStorage {
    fn new(id: Id<wgpu::Buffer>, device: WgpuDevice, dtype: DType) -> Self {
        Self { id, device, dtype }
    }
}

impl BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        todo!()
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        let data = self
            .device
            .backend
            .read_buf_as::<f32>(self.id)
            .map_err(|e| WgpuError::WgpuBackendError(e))?;
        Ok(CpuStorage::F32(data))
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
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let params = candle_wgpu_kernels::conv::WgpuConvParams {
            input_h: 1, // since we use 2d conv for 1d operation
            input_w: layout.dims()[2] as u32,
            kernel_h: 1, // same as above
            kernel_w: kernel_l.dims()[2] as u32,
            stride: params.stride as u32,
            padding_x: params.padding as u32,
            batch_size: layout.dims()[0] as u32,
            channels_in: layout.dims()[1] as u32,
            channels_out: kernel_l.dims()[0] as u32,
            ..Default::default()
        };

        let result = self
            .device
            .backend
            .conv2d(self.id, kernel.id, &params)
            .map_err(|e| WgpuError::WgpuBackendError(e))?;

        Ok(WgpuStorage {
            id: result,
            ..self.clone()
        })
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
