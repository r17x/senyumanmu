#include <metal_stdlib>
using namespace metal;

kernel void spatial_mean_kernel(
    texture2d<float, access::read> in_texture [[texture(0)]],
    device atomic_uint* out_accum             [[buffer(0)]],
    constant uint2& roi_origin                [[buffer(1)]],
    constant uint2& roi_size                  [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
)
{
    if (gid.x >= roi_size.x || gid.y >= roi_size.y) {
        return;
    }

    uint2 pixel_pos = roi_origin + gid;
    float4 px = in_texture.read(pixel_pos);

    uint r = uint(px.r * 10000.0);
    uint g = uint(px.g * 10000.0);
    uint b = uint(px.b * 10000.0);

    atomic_fetch_add_explicit(&out_accum[0], r, memory_order_relaxed);
    atomic_fetch_add_explicit(&out_accum[1], g, memory_order_relaxed);
    atomic_fetch_add_explicit(&out_accum[2], b, memory_order_relaxed);
}
