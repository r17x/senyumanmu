#![allow(unsafe_code)]

use std::ffi::c_void;
use std::ptr::{self, NonNull};

use objc2::rc::Retained;
use objc2::AnyThread;
use objc2_core_video::{CVPixelBuffer, CVPixelBufferCreateWithBytes};
use objc2_foundation::{NSDictionary, NSArray};
use objc2_vision::{
    VNDetectFaceLandmarksRequest, VNFaceLandmarkRegion2D, VNFaceObservation,
    VNImageRequestHandler, VNRequest,
};

use cam_gpu::{Frame, Rect};

const KCV_PIXEL_FORMAT_TYPE_32BGRA: u32 = 0x42475241;

pub struct LandmarkMesh {
    pub points: Vec<(f32, f32)>,
}

pub struct FaceObservation {
    pub roi: Rect,
    pub landmarks: LandmarkMesh,
}

fn create_pixel_buffer(frame: &Frame) -> Option<Retained<CVPixelBuffer>> {
    let mut pixel_buffer: *mut CVPixelBuffer = ptr::null_mut();
    let bytes_per_row = frame.width as usize * 4;

    let status = unsafe {
        CVPixelBufferCreateWithBytes(
            None,
            frame.width as usize,
            frame.height as usize,
            KCV_PIXEL_FORMAT_TYPE_32BGRA,
            NonNull::new(frame.data.as_ptr() as *mut c_void)?,
            bytes_per_row,
            None,
            ptr::null_mut(),
            None,
            NonNull::new(&mut pixel_buffer)?,
        )
    };

    if status != 0 {
        return None;
    }

    unsafe { Retained::retain(pixel_buffer) }
}

fn extract_region_points(
    region: &VNFaceLandmarkRegion2D,
    bbox_x: f64,
    bbox_y: f64,
    bbox_w: f64,
    bbox_h: f64,
    img_w: f64,
    img_h: f64,
) -> Vec<(f32, f32)> {
    let count = unsafe { region.pointCount() };
    if count == 0 {
        return Vec::new();
    }

    let raw_ptr = unsafe { region.normalizedPoints() };
    if raw_ptr.is_null() {
        return Vec::new();
    }

    let mut points = Vec::with_capacity(count);
    for i in 0..count {
        let pt = unsafe { *raw_ptr.add(i) };
        let pixel_x = (bbox_x + pt.x * bbox_w) * img_w;
        let pixel_y = (1.0 - (bbox_y + pt.y * bbox_h)) * img_h;
        let norm_x = pixel_x / img_w;
        let norm_y = pixel_y / img_h;
        points.push((norm_x as f32, norm_y as f32));
    }
    points
}

fn collect_all_landmarks(
    face: &VNFaceObservation,
    bbox_x: f64,
    bbox_y: f64,
    bbox_w: f64,
    bbox_h: f64,
    img_w: f64,
    img_h: f64,
) -> Vec<(f32, f32)> {
    let landmarks = unsafe { face.landmarks() };
    let landmarks = match landmarks {
        Some(l) => l,
        None => return Vec::new(),
    };

    let all_points = unsafe { landmarks.allPoints() };
    match all_points {
        Some(region) => {
            extract_region_points(&region, bbox_x, bbox_y, bbox_w, bbox_h, img_w, img_h)
        }
        None => Vec::new(),
    }
}

pub fn detect_face(frame: &Frame) -> Option<FaceObservation> {
    let pixel_buffer = create_pixel_buffer(frame)?;

    let options: Retained<NSDictionary<objc2_vision::VNImageOption, objc2::runtime::AnyObject>> =
        NSDictionary::new();

    let handler = unsafe {
        VNImageRequestHandler::initWithCVPixelBuffer_options(
            VNImageRequestHandler::alloc(),
            &pixel_buffer,
            &options,
        )
    };

    let request = unsafe { VNDetectFaceLandmarksRequest::new() };

    let request_ref: &VNRequest = &request;
    let requests = NSArray::from_slice(&[request_ref]);

    handler.performRequests_error(&requests).ok()?;

    let results = unsafe { request.results() }?;
    if results.is_empty() {
        return None;
    }

    let face = results.objectAtIndex(0);
    let bbox = unsafe { face.boundingBox() };

    let img_w = f64::from(frame.width);
    let img_h = f64::from(frame.height);

    let roi = Rect {
        x: (bbox.origin.x * img_w) as u32,
        y: ((1.0 - bbox.origin.y - bbox.size.height) * img_h) as u32,
        width: (bbox.size.width * img_w) as u32,
        height: (bbox.size.height * img_h) as u32,
    };

    let points = collect_all_landmarks(
        &face,
        bbox.origin.x,
        bbox.origin.y,
        bbox.size.width,
        bbox.size.height,
        img_w,
        img_h,
    );

    Some(FaceObservation {
        roi,
        landmarks: LandmarkMesh { points },
    })
}
