use cam_gpu::{upload_frame, run_spatial_kernel, run_temporal_kernel, Frame, GpuContext, GpuError, Rect, RoiRect};
use cam_signal::{FrameSignals, SignalStream};

const WIDTH: u32 = 128;
const HEIGHT: u32 = 128;
const TOLERANCE: f32 = 0.02;

const EXPECTED_R: f32 = 200.0 / 255.0;
const EXPECTED_G: f32 = 100.0 / 255.0;
const EXPECTED_B: f32 = 50.0 / 255.0;

fn assert_approx(label: &str, actual: f32, expected: f32) {
    let diff = (actual - expected).abs();
    assert!(
        diff <= TOLERANCE,
        "{label}: expected {expected:.4} +/-{TOLERANCE}, got {actual:.4} (diff={diff:.6})"
    );
}

fn main() {
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
    println!("[1/8] GpuContext created");

    let pixel: [u8; 4] = [50, 100, 200, 255];
    let data: Vec<u8> = pixel.repeat((WIDTH * HEIGHT) as usize);
    let frame = Frame {
        width: WIDTH,
        height: HEIGHT,
        data,
    };
    println!("[2/8] Synthetic frame: {WIDTH}x{HEIGHT} BGRA8 [B=50, G=100, R=200, A=255]");

    let texture = upload_frame(&ctx, &frame).unwrap_or_else(|e| {
        eprintln!("upload_frame failed: {e}");
        std::process::exit(1);
    });
    println!("[3/8] Frame uploaded to Metal texture");

    let spatial_roi = Rect {
        x: 0,
        y: 0,
        width: WIDTH,
        height: HEIGHT,
    };
    let spatial_means = run_spatial_kernel(&ctx, &texture, spatial_roi).unwrap_or_else(|e| {
        eprintln!("run_spatial_kernel failed: {e}");
        std::process::exit(1);
    });
    println!(
        "[4/8] Spatial kernel: R={:.4}, G={:.4}, B={:.4}",
        spatial_means[0], spatial_means[1], spatial_means[2]
    );
    assert_approx("spatial R", spatial_means[0], EXPECTED_R);
    assert_approx("spatial G", spatial_means[1], EXPECTED_G);
    assert_approx("spatial B", spatial_means[2], EXPECTED_B);
    println!("[5/8] Spatial assertions passed");

    let temporal_roi = RoiRect {
        x: 0,
        y: 0,
        width: WIDTH,
        height: HEIGHT,
    };
    let color_sample = run_temporal_kernel(&ctx, &texture, &temporal_roi, 0).unwrap_or_else(|e| {
        eprintln!("run_temporal_kernel failed: {e}");
        std::process::exit(1);
    });
    println!(
        "[6/8] Temporal kernel: R={:.4}, G={:.4}, B={:.4}, frame={}",
        color_sample.r, color_sample.g, color_sample.b, color_sample.frame_index
    );
    assert_approx("temporal R", color_sample.r, EXPECTED_R);
    assert_approx("temporal G", color_sample.g, EXPECTED_G);
    assert_approx("temporal B", color_sample.b, EXPECTED_B);
    assert!(
        color_sample.frame_index == 0,
        "expected frame_index=0, got {}",
        color_sample.frame_index
    );
    println!("[7/8] Temporal assertions passed");

    let signals = FrameSignals {
        frame_index: 0,
        spatial_means,
        color_sample,
    };
    let mut stream = SignalStream::new();
    stream.push(signals);
    assert!(stream.len() == 1, "expected stream len=1, got {}", stream.len());
    println!("[8/8] SignalStream: len={}", stream.len());

    println!();
    println!("PASS");
}
