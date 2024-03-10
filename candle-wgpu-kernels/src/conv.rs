use wgpu::{Buffer, Id};

use crate::{WgpuBackend, WgpuBackendResult};

impl WgpuBackend {
    pub fn conv1d(&self, input: Id<Buffer>) -> WgpuBackendResult<Id<Buffer>> {
        smol::block_on(async {})
    }
}
