#include <metal_stdlib>
using namespace metal;

kernel void temporal_sample_kernel(
    texture2d<float, access::read>  in_texture  [[texture(0)]],
    device atomic_uint*             out_accum   [[buffer(0)]],
    constant uint2&                 roi_origin  [[buffer(1)]],
    constant uint2&                 roi_size    [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    if (gid.x >= roi_size.x || gid.y >= roi_size.y) {
        return;
    }

    uint2 pixel_pos = roi_origin + gid;
    float4 px = in_texture.read(pixel_pos);

    uint r_fixed = uint(px.r * 65536.0);
    uint g_fixed = uint(px.g * 65536.0);
    uint b_fixed = uint(px.b * 65536.0);

    atomic_fetch_add_explicit(&out_accum[0], r_fixed, memory_order_relaxed);
    atomic_fetch_add_explicit(&out_accum[1], g_fixed, memory_order_relaxed);
    atomic_fetch_add_explicit(&out_accum[2], b_fixed, memory_order_relaxed);
}
