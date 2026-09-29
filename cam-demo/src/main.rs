#![allow(unsafe_code)]

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use cam_affect::{classify_affect, compute_au_activations};
use cam_gpu::{
    run_spatial_kernel, run_temporal_kernel, upload_frame, Frame, GpuContext, GpuError, Rect,
    RoiRect,
};
use cam_vision::detect_face;

fn main() {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    })
    .unwrap_or_else(|e| {
        eprintln!("failed to set Ctrl+C handler: {e}");
        std::process::exit(1);
    });

    eprintln!("[cam-demo] initializing nokhwa...");
    let (init_tx, init_rx) = std::sync::mpsc::channel();
    nokhwa::nokhwa_initialize(move |granted| {
        let _ = init_tx.send(granted);
    });
    match init_rx.recv_timeout(std::time::Duration::from_secs(10)) {
        Ok(true) => eprintln!("[cam-demo] camera permission granted"),
        Ok(false) => {
            println!("SKIP: camera permission denied");
            return;
        }
        Err(_) => {
            eprintln!("[cam-demo] permission request timed out, continuing anyway...");
        }
    }

    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(GpuError::NoDevice) => {
            println!("SKIP: no Metal device");
            return;
        }
        Err(e) => {
            eprintln!("GPU init failed: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("[cam-demo] GPU context ready");

    let camera_index = nokhwa::utils::CameraIndex::Index(0);
    let requested_format = nokhwa::utils::RequestedFormat::new::<nokhwa::pixel_format::RgbFormat>(
        nokhwa::utils::RequestedFormatType::AbsoluteHighestFrameRate,
    );

    eprintln!("[cam-demo] creating camera...");
    let mut camera = match nokhwa::Camera::new(camera_index, requested_format) {
        Ok(c) => c,
        Err(e) => {
            let msg = format!("{e}");
            if msg.contains("permission") || msg.contains("Permission") || msg.contains("denied") {
                println!("SKIP: camera permission denied");
                return;
            }
            eprintln!("camera init failed: {e}");
            std::process::exit(1);
        }
    };

    eprintln!("[cam-demo] camera created, opening stream...");
    if let Err(e) = camera.open_stream() {
        let msg = format!("{e}");
        if msg.contains("permission") || msg.contains("Permission") || msg.contains("denied") {
            println!("SKIP: camera permission denied");
            return;
        }
        eprintln!("camera open failed: {e}");
        std::process::exit(1);
    }

    eprintln!("[cam-demo] stream opened, capturing frames...");

    let mut frame_idx: u32 = 0;
    let stdout = std::io::stdout();

    while running.load(Ordering::SeqCst) {
        let buf = match camera.frame() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[cam-demo] frame error: {e}");
                continue;
            }
        };

        let res = buf.resolution();
        let width = res.width_x;
        let height = res.height_y;

        let rgb_image = match buf.decode_image::<nokhwa::pixel_format::RgbFormat>() {
            Ok(decoded) => decoded,
            Err(e) => {
                eprintln!("[cam-demo] decode error: {e}");
                continue;
            }
        };
        let rgb_bytes = rgb_image.into_raw();

        let pixel_count = (width as usize) * (height as usize);
        let expected_len = pixel_count * 3;
        if rgb_bytes.len() < expected_len {
            eprintln!(
                "[cam-demo] short buffer after decode: {} < {expected_len}",
                rgb_bytes.len()
            );
            continue;
        }

        let mut bgra = Vec::with_capacity(pixel_count * 4);
        for i in 0..pixel_count {
            let base = i * 3;
            bgra.push(rgb_bytes[base + 2]);
            bgra.push(rgb_bytes[base + 1]);
            bgra.push(rgb_bytes[base]);
            bgra.push(255);
        }

        let frame = Frame {
            width,
            height,
            data: bgra,
        };

        let obs = match detect_face(&frame) {
            Some(o) => o,
            None => {
                let mut out = stdout.lock();
                let _ = writeln!(out, "frame={:<6} face=NO  (skipped)", frame_idx);
                frame_idx = frame_idx.wrapping_add(1);
                continue;
            }
        };

        let texture = match upload_frame(&ctx, &frame) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("upload_frame failed: {e}");
                continue;
            }
        };

        let spatial_roi = Rect {
            x: obs.roi.x,
            y: obs.roi.y,
            width: obs.roi.width,
            height: obs.roi.height,
        };
        let spatial_means = match run_spatial_kernel(&ctx, &texture, spatial_roi) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("spatial kernel failed: {e}");
                continue;
            }
        };

        let temporal_roi = RoiRect {
            x: obs.roi.x,
            y: obs.roi.y,
            width: obs.roi.width,
            height: obs.roi.height,
        };
        let color_sample = match run_temporal_kernel(&ctx, &texture, &temporal_roi, frame_idx) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("temporal kernel failed: {e}");
                continue;
            }
        };

        let au = compute_au_activations(&obs.landmarks);
        let affect = classify_affect(&au);

        {
            let mut out = stdout.lock();
            let _ = writeln!(
                out,
                "frame={:<6} face=YES  R={:.3} G={:.3} B={:.3}  rPPG_g={:.3}  affect={}",
                frame_idx, spatial_means[0], spatial_means[1], spatial_means[2], color_sample.g, affect,
            );
        }

        frame_idx = frame_idx.wrapping_add(1);
    }

    let _ = std::io::stdout().flush();
}
