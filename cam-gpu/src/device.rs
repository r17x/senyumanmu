use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice};

#[derive(Debug, Clone, thiserror::Error)]
pub enum GpuError {
    #[error("no Metal-capable GPU found")]
    NoDevice,
    #[error("Metal library error: {0}")]
    LibraryError(String),
    #[error("Metal pipeline error: {0}")]
    PipelineError(String),
    #[error("Metal command error: {0}")]
    CommandError(String),
    #[error("Metal texture error: {0}")]
    TextureError(String),
}

pub struct GpuContext {
    pub device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

impl GpuContext {
    pub fn new() -> Result<Self, GpuError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(GpuError::NoDevice)?;
        let queue = device
            .newCommandQueue()
            .ok_or(GpuError::CommandError("failed to create command queue".into()))?;
        Ok(Self { device, queue })
    }
}
