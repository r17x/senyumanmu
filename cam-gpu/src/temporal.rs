#![allow(unsafe_code)]

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use cam_signal::ColorSample;

use crate::device::{GpuContext, GpuError};
use crate::frame::MetalTexture;

pub struct RoiRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

const FIXED_POINT_SCALE: f32 = 65536.0;
const ACCUM_BUFFER_SIZE: usize = 3 * std::mem::size_of::<u32>();

fn compile_temporal_pipeline(
    ctx: &GpuContext,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, GpuError> {
    let source = include_str!("../shaders/temporal.metal");
    let ns_source = NSString::from_str(source);

    let library = ctx
        .device
        .newLibraryWithSource_options_error(&ns_source, None)
        .map_err(|e| GpuError::LibraryError(e.localizedDescription().to_string()))?;

    let function_name = NSString::from_str("temporal_sample_kernel");
    let function = library
        .newFunctionWithName(&function_name)
        .ok_or_else(|| {
            GpuError::LibraryError("temporal_sample_kernel not found in library".into())
        })?;

    ctx.device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| GpuError::PipelineError(e.localizedDescription().to_string()))
}

pub fn run_temporal_kernel(
    ctx: &GpuContext,
    texture: &MetalTexture,
    roi: &RoiRect,
    frame_idx: u32,
) -> Result<ColorSample, GpuError> {
    let pipeline = compile_temporal_pipeline(ctx)?;

    let accum_buffer = ctx
        .device
        .newBufferWithLength_options(ACCUM_BUFFER_SIZE, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| GpuError::CommandError("failed to create accumulator buffer".into()))?;

    unsafe {
        let ptr = accum_buffer.contents().as_ptr() as *mut u32;
        ptr.write(0);
        ptr.add(1).write(0);
        ptr.add(2).write(0);
    }

    let command_buffer = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| GpuError::CommandError("failed to create command buffer".into()))?;

    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| GpuError::CommandError("failed to create compute encoder".into()))?;

    encoder.setComputePipelineState(&pipeline);

    unsafe {
        encoder.setTexture_atIndex(Some(texture.as_mtl()), 0);
        encoder.setBuffer_offset_atIndex(Some(&accum_buffer), 0, 0);
    }

    let roi_origin: [u32; 2] = [roi.x, roi.y];
    let roi_size: [u32; 2] = [roi.width, roi.height];

    unsafe {
        let origin_ptr =
            NonNull::new(roi_origin.as_ptr() as *mut c_void).expect("stack pointer is non-null");
        encoder.setBytes_length_atIndex(origin_ptr, std::mem::size_of::<[u32; 2]>(), 1);

        let size_ptr =
            NonNull::new(roi_size.as_ptr() as *mut c_void).expect("stack pointer is non-null");
        encoder.setBytes_length_atIndex(size_ptr, std::mem::size_of::<[u32; 2]>(), 2);
    }

    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let thread_width = pipeline.threadExecutionWidth();
    let thread_height = max_threads / thread_width;

    let threads_per_grid = MTLSize {
        width: roi.width as usize,
        height: roi.height as usize,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: thread_width.min(roi.width as usize),
        height: thread_height.min(roi.height as usize),
        depth: 1,
    };

    encoder.dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup);
    encoder.endEncoding();

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    if let Some(err) = command_buffer.error() {
        return Err(GpuError::CommandError(
            err.localizedDescription().to_string(),
        ));
    }

    let pixel_count = roi.width as f32 * roi.height as f32;
    let (r_sum, g_sum, b_sum) = unsafe {
        let ptr = accum_buffer.contents().as_ptr() as *const u32;
        (*ptr, *ptr.add(1), *ptr.add(2))
    };

    Ok(ColorSample {
        r: (r_sum as f32 / FIXED_POINT_SCALE) / pixel_count,
        g: (g_sum as f32 / FIXED_POINT_SCALE) / pixel_count,
        b: (b_sum as f32 / FIXED_POINT_SCALE) / pixel_count,
        frame_index: frame_idx,
    })
}
