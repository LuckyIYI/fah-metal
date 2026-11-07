// Image pre-processing for FAH straight from the camera buffer (BGRA or bi-planar 4:2:0 YCbCr), emulating the
// uint8 OpenCV path of FAH's Python pipeline (fah.detector._letterbox, estimator._make_batch_fast) so that the network
// sees the same pixels: the detector letterbox (cv2.resize INTER_LINEAR + gray-114 padding) and the pose crops
// (cv2.warpAffine INTER_LINEAR, zero border). See ImageWarp.swift.
#include <metal_stdlib>
using namespace metal;

struct YUVParams { float4 m0; float4 m1; float4 m2; float4 off; };
struct LetterboxParams {
    int2 outSize; int2 resized; int2 pad; int2 canvas; float2 scale; int2 srcSize; float4 a0; float4 a1; int quarter;
};

static inline float3 round_half_up(float3 v) { return floor(v + 0.5); }

// uint8 RGB (0..255) of an in-bounds buffer pixel. BGRA: t0 (t1 unused); YUV: luma t0, chroma t1 (bilinear, centre sited)
template <bool YUV>
static inline float3 fetch(texture2d<float, access::read> t0, texture2d<float, access::sample> t1, constant YUVParams& Y,
                           int2 size, int2 p) {
    if (!YUV) return rint(t0.read(uint2(p)).rgb * 255.0);
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float y = (t0.read(uint2(p)).r - Y.off.x) * Y.off.y;
    float2 c = (t1.sample(s, (float2(p) + 0.5) / float2(size)).rg - Y.off.z) * Y.off.w;
    float3 v = float3(y, c.x, c.y);
    float3 rgb = float3(dot(Y.m0.xyz, v), dot(Y.m1.xyz, v), dot(Y.m2.xyz, v));
    return round_half_up(clamp(rgb, 0.0, 1.0) * 255.0);
}

struct CropParams { int4 dims; float4 f0; float4 f1; };

// cv2.warpAffine(INTER_LINEAR, BORDER_CONSTANT 0) into (n, 3, S, S) float RGB / 255 (OpenCV 5: float32 source
// coordinates, float bilinear, round half to even). M: per crop the inverse affine (8 floats), dst -> upright canvas;
// then canvas -> buffer (exact for quarter turns up to float rounding).
template <bool YUV>
static inline void crop(texture2d<float, access::read> t0, texture2d<float, access::sample> t1, device float* out,
                        constant float* M, constant CropParams& C, constant YUVParams& Y, uint3 gid) {
    int S = C.dims.x;
    int2 size = int2(C.dims.y, C.dims.z);
    if (gid.x >= uint(S) || gid.y >= uint(S)) return;
    constant float* m = M + gid.z * 8;
    float x = float(gid.x), y = float(gid.y);
    float2 c = float2(m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]);
    float2 s = float2(C.f0.x * c.x + C.f0.y * c.y + C.f0.z, C.f1.x * c.x + C.f1.y * c.y + C.f1.z);
    float2 g = floor(s);
    int2 i0 = int2(g);
    float2 f = s - g;
    float3 v[4];
    for (int k = 0; k < 4; k++) {
        int2 q = i0 + int2(k & 1, k >> 1);
        v[k] = (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y) ? float3(0.0) : fetch<YUV>(t0, t1, Y, size, q);
    }
    float3 r0 = v[0] + f.x * (v[1] - v[0]), r1 = v[2] + f.x * (v[3] - v[2]);
    float3 o = clamp(rint(r0 + f.y * (r1 - r0)), 0.0, 255.0) / 255.0;
    uint plane = uint(S) * uint(S);
    uint base = gid.z * 3 * plane + gid.y * uint(S) + gid.x;
    out[base] = o.r;
    out[base + plane] = o.g;
    out[base + 2 * plane] = o.b;
}

kernel void crop_bgra(texture2d<float, access::read> t0 [[texture(0)]], texture2d<float, access::sample> t1 [[texture(1)]],
                      device float* out [[buffer(0)]], constant float* M [[buffer(1)]], constant CropParams& C [[buffer(2)]],
                      constant YUVParams& Y [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    crop<false>(t0, t1, out, M, C, Y, gid);
}
kernel void crop_yuv(texture2d<float, access::read> t0 [[texture(0)]], texture2d<float, access::sample> t1 [[texture(1)]],
                     device float* out [[buffer(0)]], constant float* M [[buffer(1)]], constant CropParams& C [[buffer(2)]],
                     constant YUVParams& Y [[buffer(3)]], uint3 gid [[thread_position_in_grid]]) {
    crop<true>(t0, t1, out, M, C, Y, gid);
}

// one upright-canvas pixel (integer canvas coords): quarter turns read the buffer pixel it is; other angles sample the
// buffer bilinearly (gray 114 outside it)
template <bool YUV>
static inline float3 canvas(texture2d<float, access::read> t0, texture2d<float, access::sample> t1, constant YUVParams& Y,
                            constant LetterboxParams& L, int2 c) {
    int2 size = L.srcSize;
    float2 b = float2(L.a0.x * c.x + L.a0.y * c.y + L.a0.z, L.a1.x * c.x + L.a1.y * c.y + L.a1.z);
    if (L.quarter) return fetch<YUV>(t0, t1, Y, size, clamp(int2(rint(b)), int2(0), size - 1));
    float2 g = floor(b);
    int2 j = int2(g);
    float2 h = b - g;
    float3 r = 0.0;
    for (int m = 0; m < 4; m++) {
        int2 o = int2(m & 1, m >> 1);
        int2 q = j + o;
        float w = (o.x ? h.x : 1.0 - h.x) * (o.y ? h.y : 1.0 - h.y);
        r += w * ((q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y) ? float3(114.0) : fetch<YUV>(t0, t1, Y, size, q));
    }
    return round_half_up(r);
}

// letterbox: cv2.resize(INTER_LINEAR) of the upright canvas to (nw, nh) placed at pad, gray 114 around it;
// (1, 3, H, W) half RGB / 255
template <bool YUV>
static inline void letterbox(texture2d<float, access::read> t0, texture2d<float, access::sample> t1, device half* out,
                             constant LetterboxParams& L, constant YUVParams& Y, uint2 gid) {
    if (int(gid.x) >= L.outSize.x || int(gid.y) >= L.outSize.y) return;
    int2 c = int2(gid.xy) - L.pad;
    float3 v = float3(114.0);
    if (c.x >= 0 && c.y >= 0 && c.x < L.resized.x && c.y < L.resized.y) {
        float fx = (float(c.x) + 0.5) * L.scale.x - 0.5;
        int sx = int(floor(fx));
        fx -= float(sx);
        if (sx < 0) { fx = 0.0; sx = 0; }
        if (sx >= L.canvas.x - 1) { fx = 0.0; sx = L.canvas.x - 1; }
        float fy = (float(c.y) + 0.5) * L.scale.y - 0.5;
        int sy = int(floor(fy));
        fy -= float(sy);
        if (sy < 0) { fy = 0.0; sy = 0; }
        if (sy >= L.canvas.y - 1) { fy = 0.0; sy = L.canvas.y - 1; }
        float wx = rint(fx * 2048.0) / 2048.0, wy = rint(fy * 2048.0) / 2048.0;  // OpenCV's 11-bit resize weights
        int sx1 = min(sx + 1, L.canvas.x - 1), sy1 = min(sy + 1, L.canvas.y - 1);
        float3 v00 = canvas<YUV>(t0, t1, Y, L, int2(sx, sy));
        float3 v10 = wx > 0.0 ? canvas<YUV>(t0, t1, Y, L, int2(sx1, sy)) : v00;
        float3 v01 = wy > 0.0 ? canvas<YUV>(t0, t1, Y, L, int2(sx, sy1)) : v00;
        float3 v11 = (wx > 0.0 && wy > 0.0) ? canvas<YUV>(t0, t1, Y, L, int2(sx1, sy1)) : (wx > 0.0 ? v10 : v01);
        v = round_half_up((1.0 - wy) * ((1.0 - wx) * v00 + wx * v10) + wy * ((1.0 - wx) * v01 + wx * v11));
    }
    v /= 255.0;
    uint plane = uint(L.outSize.x) * uint(L.outSize.y);
    uint base = gid.y * uint(L.outSize.x) + gid.x;
    out[base] = half(v.r);
    out[base + plane] = half(v.g);
    out[base + 2 * plane] = half(v.b);
}

kernel void letterbox_bgra(texture2d<float, access::read> t0 [[texture(0)]], texture2d<float, access::sample> t1 [[texture(1)]],
                           device half* out [[buffer(0)]], constant LetterboxParams& L [[buffer(1)]],
                           constant YUVParams& Y [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    letterbox<false>(t0, t1, out, L, Y, gid);
}
kernel void letterbox_yuv(texture2d<float, access::read> t0 [[texture(0)]], texture2d<float, access::sample> t1 [[texture(1)]],
                          device half* out [[buffer(0)]], constant LetterboxParams& L [[buffer(1)]],
                          constant YUVParams& Y [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    letterbox<true>(t0, t1, out, L, Y, gid);
}
