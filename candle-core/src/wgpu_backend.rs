use std::sync::Arc;

use candle_wgpu_kernels::{WgpuBackend, WgpuBackendError};
use wgpu::Id;

use crate::{
    backend::{BackendDevice, BackendStorage},
    dtype,
    layout::Layout,
    CpuStorage, DType, Error, Result, Shape,
};

#[derive(Debug, thiserror::Error)]
pub enum WgpuError {
    #[error("{0}")]
    Message(String),
    #[error("Wgpu backend error: {0}")]
    WgpuBackendError(#[from] WgpuBackendError),
    #[error("Unsupported operation: {name}, on type: {dtype}")]
    UnsupportedOperation { name: String, dtype: String },
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

    fn zeros_impl(&self, shape: &crate::Shape, dtype: DType) -> Result<Self::Storage> {
        match dtype {
            DType::F32 => {
                let buffer_size = shape.dims().iter().product::<usize>() * 4;
                let buffer = self
                    .backend
                    .create_buffer(buffer_size as u64)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                self.backend
                    .fill_zeroes(buffer)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                Ok(WgpuStorage::new(buffer, self.clone(), dtype))
            }
            DType::U8 => {
                let buffer_size = shape.dims().iter().product::<usize>();
                let buffer = self
                    .backend
                    .create_buffer(buffer_size as u64)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                self.backend
                    .fill_zeroes_u8(buffer)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                Ok(WgpuStorage::new(buffer, self.clone(), dtype))
            }
            DType::U32 => {
                let buffer_size = shape.dims().iter().product::<usize>() * 4;
                let buffer = self
                    .backend
                    .create_buffer(buffer_size as u64)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                self.backend
                    .fill_zeroes_u32(buffer)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                Ok(WgpuStorage::new(buffer, self.clone(), dtype))
            }
            dtype => Err(Error::Wgpu(WgpuError::UnsupportedOperation {
                name: "zeroes".to_string(),
                dtype: dtype.as_str().to_string(),
            })),
        }
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

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let output_buffer = self
            .device
            .backend
            .create_buffer(layout.dims().iter().product::<usize>() as u64)
            .map_err(|e| WgpuError::WgpuBackendError(e))?;

        self.device
            .backend
            .affine(self.id, output_buffer, mul as f32, add as f32)
            .map_err(|e| WgpuError::WgpuBackendError(e))?;

        Ok(WgpuStorage {
            id: output_buffer,
            ..self.clone()
        })
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

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        match (self.dtype, dtype) {
            (DType::U8, DType::F32) => {
                let output_buffer = self
                    .device
                    .backend
                    .create_buffer(layout.dims().iter().product::<usize>() as u64)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                self.device
                    .backend
                    .convert_u8_to_f32(self.id, output_buffer)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                Ok(WgpuStorage {
                    id: output_buffer,
                    dtype: DType::F32,
                    device: self.device.clone(),
                })
            }
            (DType::U32, DType::F32) => {
                let output_buffer = self
                    .device
                    .backend
                    .create_buffer(layout.dims().iter().product::<usize>() as u64)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                self.device
                    .backend
                    .convert_u8_to_f32(self.id, output_buffer)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                Ok(WgpuStorage {
                    id: output_buffer,
                    dtype: DType::F32,
                    device: self.device.clone(),
                })
            }
            (_, _) => crate::bail!(
                "Wgpu backend does not support conversion from: {:?}, to: {:?}",
                self.dtype,
                dtype
            ),
        }
    }

    fn unary_impl<B: crate::op::UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let op = B::NAME;

        match op {
            "sqrt" => self.unary_sqrt(layout),
            "neg" => self.unary_neg(layout),
            "exp" => self.unary_exp(layout),
            "recip" => self.unary_recip(layout),
            _ => crate::bail!("Wgpu backend does not support unary operation: {op}"),
        }
    }

    fn binary_impl<B: crate::op::BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let op = B::NAME;

        match op {
            "add" => self.binary_add(rhs, lhs_l, rhs_l),
            "sub" => self.binary_sub(rhs, lhs_l, rhs_l),
            "mul" => self.binary_mul(rhs, lhs_l, rhs_l),
            "div" => self.binary_div(rhs, lhs_l, rhs_l),
            "maximum" => self.binary_max(rhs, lhs_l, rhs_l),
            _ => crate::bail!("Wgpu backend does not support binary operation: {op}"),
        }
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
        layout: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let params = candle_wgpu_kernels::conv::WgpuConvParams {
            input_w: layout.dims()[2] as u32,
            input_h: layout.dims()[3] as u32,
            kernel_w: kernel_l.dims()[2] as u32,
            kernel_h: kernel_l.dims()[3] as u32,
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

    fn upsample_nearest2d(&self, layout: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        let output_buffer = self
            .device
            .backend
            .create_buffer((layout.dims().iter().take(2).product::<usize>() * out_w * out_h) as u64)
            .map_err(WgpuError::WgpuBackendError)?;

        self.device
            .backend
            .upsample_nearest(
                layout.dims()[0] as u32,
                layout.dims()[1] as u32,
                layout.dims()[2] as u32,
                layout.dims()[3] as u32,
                out_w as u32,
                out_h as u32,
                self.id,
                output_buffer,
            )
            .map_err(WgpuError::WgpuBackendError)?;

        Ok(Self {
            id: output_buffer,
            ..self.clone()
        })
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
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        todo!()
    }

    fn copy_strided_src(&self, dst: &mut Self, _dst_offset: usize, layout: &Layout) -> Result<()> {
        let input_w = layout.dims()[0] as u32;
        let input_h = if layout.dims().len() > 1 {
            layout.dims()[0] as u32
        } else {
            0
        };

        let stride_x = layout.stride()[0] as u32;
        let stride_y = if layout.stride().len() > 1 {
            layout.stride()[1] as u32
        } else {
            0
        };

        self.device
            .backend
            .copy(self.id, dst.id, input_w, input_h, stride_x, stride_y)
            .map_err(|e| WgpuError::WgpuBackendError(e))?;

        Ok(())
    }

    fn repeat(&self, layout: &Layout, shape: &Shape, new_shape: &Shape) -> Result<Self> {
        todo!()
    }
}

impl WgpuStorage {
    fn new(id: Id<wgpu::Buffer>, device: WgpuDevice, dtype: DType) -> Self {
        Self { id, device, dtype }
    }

    fn binary_add(&self, rhs: &Self, _lhs_l: &Layout, _rhs_l: &Layout) -> Result<Self> {
        match (self.dtype, rhs.dtype) {
            (DType::F32, DType::F32) => {
                let output_buffer = self
                    .device
                    .backend
                    .create_buffer(_lhs_l.dims().iter().product::<usize>() as u64)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                self.device
                    .backend
                    .binary_add(self.id, rhs.id, output_buffer)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                Ok(WgpuStorage {
                    id: output_buffer,
                    ..self.clone()
                })
            }
            (d_type_lhs, d_type_rhs) => crate::bail!(
                "Wgpu backend does not support binary op add for types: {d_type_lhs:?}, {d_type_rhs:?}"
            ),
        }
    }

    fn binary_sub(&self, rhs: &Self, _lhs_l: &Layout, _rhs_l: &Layout) -> Result<Self> {
        match (self.dtype, rhs.dtype) {
            (DType::F32, DType::F32) => {
                let output_buffer = self
                    .device
                    .backend
                    .create_buffer(_lhs_l.dims().iter().product::<usize>() as u64)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                self.device
                    .backend
                    .binary_sub(self.id, rhs.id, output_buffer)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                Ok(WgpuStorage {
                    id: output_buffer,
                    ..self.clone()
                })
            }
            (d_type_lhs, d_type_rhs) => crate::bail!(
                "Wgpu backend does not support binary op sub for types: {d_type_lhs:?}, {d_type_rhs:?}"
            ),
        }
    }

    fn binary_mul(&self, rhs: &Self, _lhs_l: &Layout, _rhs_l: &Layout) -> Result<Self> {
        match (self.dtype, rhs.dtype) {
            (DType::F32, DType::F32) => {
                let output_buffer = self
                    .device
                    .backend
                    .create_buffer(_lhs_l.dims().iter().product::<usize>() as u64)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                self.device
                    .backend
                    .binary_mul(self.id, rhs.id, output_buffer)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                Ok(WgpuStorage {
                    id: output_buffer,
                    ..self.clone()
                })
            }
            (d_type_lhs, d_type_rhs) => crate::bail!(
                "Wgpu backend does not support binary op mul for types: {d_type_lhs:?}, {d_type_rhs:?}"
            ),
        }
    }

    fn binary_div(&self, rhs: &Self, _lhs_l: &Layout, _rhs_l: &Layout) -> Result<Self> {
        match (self.dtype, rhs.dtype) {
            (DType::F32, DType::F32) => {
                let output_buffer = self
                    .device
                    .backend
                    .create_buffer(_lhs_l.dims().iter().product::<usize>() as u64)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                self.device
                    .backend
                    .binary_div(self.id, rhs.id, output_buffer)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                Ok(WgpuStorage {
                    id: output_buffer,
                    ..self.clone()
                })
            }
            (d_type_lhs, d_type_rhs) => crate::bail!(
                "Wgpu backend does not support binary op div for types: {d_type_lhs:?}, {d_type_rhs:?}"
            ),
        }
    }

    fn binary_max(&self, rhs: &Self, _lhs_l: &Layout, _rhs_l: &Layout) -> Result<Self> {
        match (self.dtype, rhs.dtype) {
            (DType::F32, DType::F32) => {
                let output_buffer = self
                    .device
                    .backend
                    .create_buffer(_lhs_l.dims().iter().product::<usize>() as u64)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                self.device
                    .backend
                    .binary_max(self.id, rhs.id, output_buffer)
                    .map_err(|e| WgpuError::WgpuBackendError(e))?;

                Ok(WgpuStorage {
                    id: output_buffer,
                    ..self.clone()
                })
            }
            (d_type_lhs, d_type_rhs) => crate::bail!(
                "Wgpu backend does not support binary op div for types: {d_type_lhs:?}, {d_type_rhs:?}"
            ),
        }
    }

    fn unary_sqrt(&self, layout: &Layout) -> Result<Self> {
        match self.dtype {
            DType::F32 => {
                let output_buffer = self
                    .device
                    .backend
                    .create_buffer(layout.dims().iter().product::<usize>() as u64)
                    .map_err(WgpuError::WgpuBackendError)?;

                self.device
                    .backend
                    .sqrt(self.id, output_buffer)
                    .map_err(WgpuError::WgpuBackendError)?;

                Ok(WgpuStorage {
                    id: output_buffer,
                    ..self.clone()
                })
            }
            d_type => crate::bail!("Wgpu backend does not support op sqrt for type: {d_type:?}"),
        }
    }

    fn unary_neg(&self, layout: &Layout) -> Result<Self> {
        match self.dtype {
            DType::F32 => {
                let output_buffer = self
                    .device
                    .backend
                    .create_buffer(layout.dims().iter().product::<usize>() as u64)
                    .map_err(WgpuError::WgpuBackendError)?;

                self.device
                    .backend
                    .neg(self.id, output_buffer)
                    .map_err(WgpuError::WgpuBackendError)?;

                Ok(WgpuStorage {
                    id: output_buffer,
                    ..self.clone()
                })
            }
            d_type => crate::bail!("Wgpu backend does not support op sqrt for type: {d_type:?}"),
        }
    }
    fn unary_exp(&self, layout: &Layout) -> Result<Self> {
        match self.dtype {
            DType::F32 => {
                let output_buffer = self
                    .device
                    .backend
                    .create_buffer(layout.dims().iter().product::<usize>() as u64)
                    .map_err(WgpuError::WgpuBackendError)?;

                self.device
                    .backend
                    .exp(self.id, output_buffer)
                    .map_err(WgpuError::WgpuBackendError)?;

                Ok(WgpuStorage {
                    id: output_buffer,
                    ..self.clone()
                })
            }
            d_type => crate::bail!("Wgpu backend does not support op exp for type: {d_type:?}"),
        }
    }

    fn unary_recip(&self, layout: &Layout) -> Result<Self> {
        match self.dtype {
            DType::F32 => {
                let output_buffer = self
                    .device
                    .backend
                    .create_buffer(layout.dims().iter().product::<usize>() as u64)
                    .map_err(WgpuError::WgpuBackendError)?;

                self.device
                    .backend
                    .recip(self.id, output_buffer)
                    .map_err(WgpuError::WgpuBackendError)?;

                Ok(WgpuStorage {
                    id: output_buffer,
                    ..self.clone()
                })
            }
            d_type => crate::bail!("Wgpu backend does not support op recip for type: {d_type:?}"),
        }
    }
}
