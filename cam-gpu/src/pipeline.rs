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

use crate::device::{GpuContext, GpuError};
use crate::frame::MetalTexture;

const FIXED_POINT_SCALE: f32 = 10000.0;
const ACCUM_BUFFER_LEN: usize = 3;
const MSL_SOURCE: &str = include_str!("../shaders/spatial.metal");

pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub fn run_spatial_kernel(
    ctx: &GpuContext,
    texture: &MetalTexture,
    roi: Rect,
) -> Result<[f32; 3], GpuError> {
    let library = compile_library(&ctx.device)?;
    let function = get_kernel_function(&library)?;
    let pipeline_state = create_pipeline_state(&ctx.device, &function)?;

    let accum_buffer = ctx
        .device
        .newBufferWithLength_options(
            (ACCUM_BUFFER_LEN * std::mem::size_of::<u32>()) as usize,
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or_else(|| GpuError::CommandError("failed to create accumulator buffer".into()))?;

    zero_buffer(&accum_buffer);

    let command_buffer = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| GpuError::CommandError("failed to create command buffer".into()))?;

    encode_dispatch(
        &command_buffer,
        &pipeline_state,
        texture,
        &accum_buffer,
        &roi,
    )?;

    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    if let Some(err) = command_buffer.error() {
        return Err(GpuError::CommandError(
            err.localizedDescription().to_string(),
        ));
    }

    read_results(&accum_buffer, &roi)
}

fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, GpuError> {
    let source = NSString::from_str(MSL_SOURCE);
    device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|e| GpuError::LibraryError(e.localizedDescription().to_string()))
}

fn get_kernel_function(
    library: &ProtocolObject<dyn MTLLibrary>,
) -> Result<Retained<ProtocolObject<dyn objc2_metal::MTLFunction>>, GpuError> {
    let name = NSString::from_str("spatial_mean_kernel");
    library
        .newFunctionWithName(&name)
        .ok_or_else(|| GpuError::LibraryError("function 'spatial_mean_kernel' not found".into()))
}

fn create_pipeline_state(
    device: &ProtocolObject<dyn MTLDevice>,
    function: &ProtocolObject<dyn objc2_metal::MTLFunction>,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, GpuError> {
    device
        .newComputePipelineStateWithFunction_error(function)
        .map_err(|e| GpuError::PipelineError(e.localizedDescription().to_string()))
}

fn zero_buffer(buffer: &ProtocolObject<dyn MTLBuffer>) {
    let ptr = buffer.contents().as_ptr() as *mut u32;
    unsafe {
        std::ptr::write_bytes(ptr, 0, ACCUM_BUFFER_LEN);
    }
}

fn encode_dispatch(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    pipeline_state: &ProtocolObject<dyn MTLComputePipelineState>,
    texture: &MetalTexture,
    accum_buffer: &ProtocolObject<dyn MTLBuffer>,
    roi: &Rect,
) -> Result<(), GpuError> {
    let encoder = command_buffer
        .computeCommandEncoder()
        .ok_or_else(|| GpuError::CommandError("failed to create compute encoder".into()))?;

    encoder.setComputePipelineState(pipeline_state);

    unsafe {
        encoder.setTexture_atIndex(Some(&texture.texture), 0);
    }

    unsafe {
        encoder.setBuffer_offset_atIndex(Some(accum_buffer), 0, 0);
    }

    let origin: [u32; 2] = [roi.x, roi.y];
    let size: [u32; 2] = [roi.width, roi.height];

    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new(origin.as_ptr() as *mut c_void).expect("origin ptr"),
            std::mem::size_of::<[u32; 2]>(),
            1,
        );
    }

    unsafe {
        encoder.setBytes_length_atIndex(
            NonNull::new(size.as_ptr() as *mut c_void).expect("size ptr"),
            std::mem::size_of::<[u32; 2]>(),
            2,
        );
    }

    let max_threads = pipeline_state.maxTotalThreadsPerThreadgroup();
    let thread_width = pipeline_state.threadExecutionWidth();
    let thread_height = max_threads / thread_width;

    let threads_per_grid = MTLSize {
        width: roi.width as usize,
        height: roi.height as usize,
        depth: 1,
    };
    let threads_per_threadgroup = MTLSize {
        width: thread_width,
        height: thread_height,
        depth: 1,
    };

    encoder.dispatchThreads_threadsPerThreadgroup(threads_per_grid, threads_per_threadgroup);
    encoder.endEncoding();

    Ok(())
}

fn read_results(
    accum_buffer: &ProtocolObject<dyn MTLBuffer>,
    roi: &Rect,
) -> Result<[f32; 3], GpuError> {
    let pixel_count = (roi.width as u64) * (roi.height as u64);
    if pixel_count == 0 {
        return Err(GpuError::CommandError("ROI has zero area".into()));
    }

    let ptr = accum_buffer.contents().as_ptr() as *const u32;
    let sums: [u32; 3] = unsafe { [*ptr, *ptr.add(1), *ptr.add(2)] };

    Ok([
        (sums[0] as f64 / pixel_count as f64 / FIXED_POINT_SCALE as f64) as f32,
        (sums[1] as f64 / pixel_count as f64 / FIXED_POINT_SCALE as f64) as f32,
        (sums[2] as f64 / pixel_count as f64 / FIXED_POINT_SCALE as f64) as f32,
    ])
}
