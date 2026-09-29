#![allow(unsafe_code)]

use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureUsage,
};

use crate::device::{GpuContext, GpuError};

pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

pub struct MetalTexture {
    pub(crate) texture: Retained<ProtocolObject<dyn MTLTexture>>,
}

impl MetalTexture {
    pub fn as_mtl(&self) -> &ProtocolObject<dyn MTLTexture> {
        &self.texture
    }
}

pub fn upload_frame(ctx: &GpuContext, frame: &Frame) -> Result<MetalTexture, GpuError> {
    let w = frame.width as usize;
    let h = frame.height as usize;
    let expected = w * h * 4;
    if frame.data.len() != expected {
        return Err(GpuError::TextureError(format!(
            "expected {} bytes for {}x{} BGRA8, got {}",
            expected,
            frame.width,
            frame.height,
            frame.data.len()
        )));
    }

    let desc = MTLTextureDescriptor::new();
    desc.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    unsafe { desc.setWidth(frame.width as usize) };
    unsafe { desc.setHeight(frame.height as usize) };
    desc.setStorageMode(MTLStorageMode::Shared);
    desc.setUsage(MTLTextureUsage::ShaderRead);

    let texture = ctx
        .device
        .newTextureWithDescriptor(&desc)
        .ok_or_else(|| GpuError::TextureError("failed to create texture".into()))?;

    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: frame.width as usize,
            height: frame.height as usize,
            depth: 1,
        },
    };

    let bytes_per_row = (frame.width as usize) * 4;
    let ptr = NonNull::new(frame.data.as_ptr() as *mut std::ffi::c_void)
        .ok_or_else(|| GpuError::TextureError("null pointer for frame data".into()))?;

    unsafe {
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(region, 0, ptr, bytes_per_row);
    }

    Ok(MetalTexture { texture })
}
