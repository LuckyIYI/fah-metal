// FAH-S pose network (DINOv3 ViT-S/16 student + keypoint / depth / verify heads) in plain Metal compute.
//
// Matrix products go through two per-simdgroup primitives (mm_tile, mm_tg) with two backends:
//   FAH_NAX=1  Metal 4 tensor ops (MetalPerformancePrimitives matmul2d at simdgroup scope, whole tiles per call,
//              right operand read by the op from device memory): on M5 / A19 this runs on the neural accelerators.
//   FAH_NAX=0  simdgroup_matrix 8x8 (any Apple GPU family 7+, e.g. A17 Pro / M1..M4).
// Results are staged through threadgroup memory, so every epilogue (bias, GELU, RoPE, LayerNorm, residual, pixel
// shuffle, softmax) is backend independent. The NAX GEMM structure follows MLX's steel thin NAX GEMM, the
// simdgroup_matrix attention (attn_sg) MLX's steel attention / BaseMMAFrag, and the erf approximation is MLX's:
//
// Copyright © 2023-2025 Apple Inc. (MLX, MIT licence: https://github.com/ml-explore/mlx/blob/main/LICENSE)
//
// Everything is fp16 storage / fp32 accumulation; the ViT residual stream, LayerNorm statistics and softmax are fp32,
// and the attention keys carry an fp16 hi/lo split (see attn). FAH_RELAXED sets matmul2d's relaxed_precision (as
// MLX; no measurable accuracy difference here).

#include <metal_stdlib>
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#ifndef FAH_RELAXED
#define FAH_RELAXED true
#endif
#ifndef FAH_NAX
#define FAH_NAX 0
#endif
#if FAH_NAX
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;

// ------------------------------------------------------------------------------------------------------------
// math
// ------------------------------------------------------------------------------------------------------------
inline float fah_erf(float a) {  // MLX erf.h (faithfully rounded erff)
  float r, s, t, u;
  t = metal::abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    r = metal::fma(-1.72853470e-5f, t, 3.83197126e-4f);
    u = metal::fma(-3.88396438e-3f, t, 2.42546219e-2f);
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f);
    r = metal::fma(r, t, -6.34846687e-1f);
    r = metal::fma(r, t, -1.28717512e-1f);
    r = metal::fma(r, t, -t);
    r = 1.0f - metal::precise::exp(r);
    r = metal::copysign(r, a);
  } else {
    r = -5.96761703e-4f;
    r = metal::fma(r, s, 4.99119423e-3f);
    r = metal::fma(r, s, -2.67681349e-2f);
    r = metal::fma(r, s, 1.12819925e-1f);
    r = metal::fma(r, s, -3.76125336e-1f);
    r = metal::fma(r, s, 1.28379166e-1f);
    r = metal::fma(r, a, a);
  }
  return r;
}
inline float gelu(float x) { return 0.5f * x * (1.0f + fah_erf(x * 0.70710678118654752f)); }

// ------------------------------------------------------------------------------------------------------------
// matrix-product primitives (one simdgroup each)
// ------------------------------------------------------------------------------------------------------------
#if FAH_NAX
using ext2 = metal::dextents<int32_t, 2>;
using str2 = metal::array<int32_t, 2>;
using dten = metal::tensor<device half, ext2, metal::tensor_inline>;
using tten = metal::tensor<threadgroup half, ext2, metal::tensor_inline>;
using tften = metal::tensor<threadgroup float, ext2, metal::tensor_inline>;
// a rows x cols row-major view with row stride ld (tensor extents are innermost first)
inline dten view(const device half* p, int cols, int rows, int ld) { return dten((device half*)p, ext2(cols, rows), str2{1, ld}); }
inline tten view(const threadgroup half* p, int cols, int rows, int ld) {
  return tten((threadgroup half*)p, ext2(cols, rows), str2{1, ld});
}
#endif

// T (TSM x TSN, fp32 threadgroup, row stride ldt) = sum_{i < nk} A_i * B_i, A_i = aptr(i): TSM x BK (row stride lda),
// B_i = bptr(i): BK x TSN (row stride ldb), fp16 device memory. NAX: one matmul2d(TSM, TSN, BK) per chunk, the left
// operand through a cooperative tensor and the right one read by the tensor op (as MLX's thin NAX GEMM).
template <int TSM, int TSN, int BK, typename AF, typename BF>
inline void mm_tile(AF aptr, int lda, BF bptr, int ldb, int nk, threadgroup float* T, int ldt) {
#if FAH_NAX
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(TSM, TSN, BK, false, false, FAH_RELAXED,
                                                              mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  dten t0 = view(aptr(0), BK, TSM, lda);
  dten u0 = view(bptr(0), TSN, BK, ldb);
  auto a0 = t0.template slice<BK, TSM>(0, 0);
  auto b0 = u0.template slice<TSN, BK>(0, 0);
  auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();
#pragma unroll
  for (uint16_t i = 0; i < acc.get_capacity(); i++) acc[i] = 0.0f;
  auto areg = op.template get_left_input_cooperative_tensor<half, half, float>();
  for (int i = 0; i < nk; i++) {
    dten ta = view(aptr(i), BK, TSM, lda);
    dten tb = view(bptr(i), TSN, BK, ldb);
    auto a = ta.template slice<BK, TSM>(0, 0);
    auto b = tb.template slice<TSN, BK>(0, 0);
    areg.load(a);
    op.run(areg, b, acc);
  }
  tften t(T, ext2(TSN, TSM), str2{1, ldt});
  acc.store(t);
#else
  simdgroup_float8x8 c[TSM / 8][TSN / 8];
#pragma unroll
  for (short i = 0; i < TSM / 8; i++)
#pragma unroll
    for (short j = 0; j < TSN / 8; j++) c[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
  for (int it = 0; it < nk; it++) {
    const device half* A = aptr(it);
    const device half* B = bptr(it);
    for (short k = 0; k < BK; k += 8) {
      simdgroup_half8x8 a[TSM / 8], b[TSN / 8];
#pragma unroll
      for (short i = 0; i < TSM / 8; i++) simdgroup_load(a[i], A + i * 8 * lda + k, lda);
#pragma unroll
      for (short j = 0; j < TSN / 8; j++) simdgroup_load(b[j], B + k * ldb + j * 8, ldb);
#pragma unroll
      for (short i = 0; i < TSM / 8; i++)
#pragma unroll
        for (short j = 0; j < TSN / 8; j++) simdgroup_multiply_accumulate(c[i][j], a[i], b[j], c[i][j]);
    }
  }
#pragma unroll
  for (short i = 0; i < TSM / 8; i++)
#pragma unroll
    for (short j = 0; j < TSN / 8; j++) simdgroup_store(c[i][j], T + i * 8 * ldt + j * 8, ldt);
#endif
}

// C (TSM x TSN, fp32 threadgroup, ld ldc) (+)= A (TSM x K, ld lda; device or threadgroup fp16) * B (device fp16:
// K x TSN with ld ldb, or with TB its transpose stored TSN x K)
template <int TSM, int TSN, int K, bool TB, bool ACC, typename PA>
inline void mm_tg(PA A, int lda, const device half* B, int ldb, threadgroup float* C, int ldc) {
#if FAH_NAX
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      TSM, TSN, K, false, TB, FAH_RELAXED,
      ACC ? mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate : mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto ta = view(A, K, TSM, lda);
  dten tb = TB ? view(B, K, TSN, ldb) : view(B, TSN, K, ldb);
  tften tc(C, ext2(TSN, TSM), str2{1, ldc});
  op.run(ta, tb, tc);
#else
  simdgroup_float8x8 c[TSM / 8][TSN / 8];
#pragma unroll
  for (short i = 0; i < TSM / 8; i++)
#pragma unroll
    for (short j = 0; j < TSN / 8; j++) {
      if (ACC) simdgroup_load(c[i][j], C + i * 8 * ldc + j * 8, ldc);
      else c[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    }
  for (short k = 0; k < K; k += 8) {
    simdgroup_half8x8 a[TSM / 8], b[TSN / 8];
#pragma unroll
    for (short i = 0; i < TSM / 8; i++) simdgroup_load(a[i], A + i * 8 * lda + k, lda);
#pragma unroll
    for (short j = 0; j < TSN / 8; j++) {
      if (TB) simdgroup_load(b[j], B + j * 8 * ldb + k, ldb, ulong2(0, 0), true);
      else simdgroup_load(b[j], B + k * ldb + j * 8, ldb);
    }
#pragma unroll
    for (short i = 0; i < TSM / 8; i++)
#pragma unroll
      for (short j = 0; j < TSN / 8; j++) simdgroup_multiply_accumulate(c[i][j], a[i], b[j], c[i][j]);
  }
#pragma unroll
  for (short i = 0; i < TSM / 8; i++)
#pragma unroll
    for (short j = 0; j < TSN / 8; j++) simdgroup_store(c[i][j], C + i * 8 * ldc + j * 8, ldc);
#endif
}

struct GP {
  int M, K, lda, ldb, ldo, N;
  float eps, eps2;
  int a0, a1, a2, a3;  // kernel-specific
};

// ------------------------------------------------------------------------------------------------------------
// GEMM + bias (+ GELU). Each simdgroup a TSM x TSN tile (WM x WN per threadgroup), staged through threadgroup memory.
// OUT 0: fp16 dense (ldo). 1: 2x2 stride-2 transposed conv pixel shuffle into a zero-bordered NHWC map
//   (rows (b, i, j) on an a0 x a0 grid, cols (di*2 + dj) * a1 + o -> pixel (2i + di + 1, 2j + dj + 1) of a
//   (2 a0 + 2)^2 map with channel stride ldo). 2: fp32 dense (ldo).
// ------------------------------------------------------------------------------------------------------------
#define MM_ACT_ARGS                                                                                                \
  const device half *A [[buffer(0)]], const device half *Bw [[buffer(1)]], const device float *bias [[buffer(2)]], \
      device half *out [[buffer(3)]], constant GP &p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],      \
      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]
template <int TSM, int TSN, int WM, int WN, int ACT, int OUT>
kernel void mm_act(MM_ACT_ARGS) {
  threadgroup float T[WM * WN][TSM * TSN];
  const int row0 = tg.x * TSM * WM + (sg / WN) * TSM, col0 = tg.y * TSN * WN + (sg % WN) * TSN;
  const device half* Ab = A + row0 * p.lda;
  const device half* Bb = Bw + col0;
  const int lda = p.lda, ldb = p.ldb;
  mm_tile<TSM, TSN, 64>([&](int i) { return Ab + i * 64; }, lda, [&](int i) { return Bb + i * 64 * ldb; }, ldb, p.K / 64,
                        T[sg], TSN);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = lane * 4; e < TSM * TSN; e += 128) {
    const int r = e / TSN, c = e % TSN, row = row0 + r, col = col0 + c;
    if (row >= p.M) continue;
    float4 v = *(threadgroup float4*)(T[sg] + e) + *(const device float4*)(bias + col);
    if (ACT == 1) v = float4(gelu(v.x), gelu(v.y), gelu(v.z), gelu(v.w));
    if (ACT == 2) v = max(v, 0.f);
    if (OUT == 0) {
      *(device half4*)(out + row * p.ldo + col) = half4(v);
    } else if (OUT == 1) {
      const int g = p.a0, O = p.a1, G = 2 * g + 2;
      const int b = row / (g * g), rem = row % (g * g), i = rem / g, j = rem % g;
      const int sub = col / O, oc = col % O, di = sub >> 1, dj = sub & 1;
      *(device half4*)(out + ((b * G + 2 * i + di + 1) * G + 2 * j + dj + 1) * p.ldo + oc) = half4(v);
    } else {
      *(device float4*)((device float*)out + row * p.ldo + col) = v;
    }
  }
}
#define INST_MM_ACT(TSM, TSN, WM, WN, ACT, OUT)                                                                         \
  template [[host_name("mm_act_" #TSM "_" #TSN "_" #WM "_" #WN "_" #ACT "_" #OUT)]] kernel void mm_act<TSM, TSN, WM, WN, \
                                                                                                       ACT, OUT>(     \
      MM_ACT_ARGS);
INST_MM_ACT(32, 32, 1, 2, 1, 0)
INST_MM_ACT(32, 32, 1, 2, 1, 1)
INST_MM_ACT(32, 32, 1, 2, 0, 2)
INST_MM_ACT(32, 32, 1, 2, 0, 0)
INST_MM_ACT(32, 32, 1, 2, 2, 0)

// ------------------------------------------------------------------------------------------------------------
// QKV projection + RoPE (rotate-half on patch tokens) + head split; q pre-scaled by 1/sqrt(64).
// Simdgroup tile = TSM rows x one 64-wide head. out q, k, v: [B, 6, a2 (padded tokens), 64] fp16;
// a0 = tokens per image (405), a1 = prefix tokens (5)
// ------------------------------------------------------------------------------------------------------------
#define QKV_ARGS                                                                                                       \
  const device half *A [[buffer(0)]], const device half *Bw [[buffer(1)]], const device float *cosT [[buffer(2)]],    \
      const device float *sinT [[buffer(3)]], device half *Q [[buffer(4)]], device half *Kt [[buffer(5)]],             \
      device half *V [[buffer(6)]], constant GP &p [[buffer(7)]], uint3 tg [[threadgroup_position_in_grid]],           \
      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]
template <int TSM, int WM, int WN>
kernel void mm_qkv(QKV_ARGS) {
  threadgroup float T[WM * WN][TSM * 64];
  const int row0 = tg.x * TSM * WM + (sg / WN) * TSM, col0 = tg.y * 64 * WN + (sg % WN) * 64;
  const device half* Ab = A + row0 * p.lda;
  const device half* Bb = Bw + col0;
  const int lda = p.lda, ldb = p.ldb;
  mm_tile<TSM, 64, 64>([&](int i) { return Ab + i * 64; }, lda, [&](int i) { return Bb + i * 64 * ldb; }, ldb, p.K / 64,
                       T[sg], 64);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const int which = col0 / 384, head = (col0 % 384) / 64;
  const int NT = p.a0, P = p.a1, NKP = p.a2;
  device half* dst = which == 0 ? Q : (which == 1 ? Kt : V);
  const float sc = which == 0 ? 0.125f : 1.0f;
  const int d0 = (lane & 1) * 16;
  for (int r = lane >> 1; r < TSM; r += 16) {
    const int row = row0 + r;
    if (row >= p.M) continue;
    const int b = row / NT, n = row % NT;
    const int W = which < 2 ? 128 : 64;  // q, k: fp16 hi/lo split (see attn)
    device half* o = dst + ((b * 6 + head) * NKP + n) * W;
    threadgroup const float* t = T[sg] + r * 64;
    const bool rope = which < 2 && n >= P;
    for (int j = 0; j < 16; j += 4) {
      float4 x1 = *(threadgroup const float4*)(t + d0 + j), x2 = *(threadgroup const float4*)(t + d0 + j + 32);
      if (rope) {
        const device float* cs = cosT + (n - P) * 64 + d0 + j;
        const device float* sn = sinT + (n - P) * 64 + d0 + j;
        const float4 c1 = *(const device float4*)cs, s1 = *(const device float4*)sn;
        const float4 c2 = *(const device float4*)(cs + 32), s2 = *(const device float4*)(sn + 32);
        const float4 y1 = x1 * c1 - x2 * s1, y2 = x2 * c2 + x1 * s2;
        x1 = y1;
        x2 = y2;
      }
      x1 *= sc;
      x2 *= sc;
      const half4 h1 = half4(x1), h2 = half4(x2);
      if (which == 1) {  // k, transposed [dim][token]: [kh ; kl] (128 x NKP)
        const half4 l1 = half4(x1 - float4(h1)), l2 = half4(x2 - float4(h2));
        device half* kt = Kt + (b * 6 + head) * 128 * NKP + n;
#pragma unroll
        for (short u = 0; u < 4; u++) {
          kt[(d0 + j + u) * NKP] = h1[u];
          kt[(d0 + j + u + 32) * NKP] = h2[u];
          kt[(64 + d0 + j + u) * NKP] = l1[u];
          kt[(64 + d0 + j + u + 32) * NKP] = l2[u];
        }
        continue;
      }
      *(device half4*)(o + d0 + j) = h1;
      *(device half4*)(o + d0 + j + 32) = h2;
      if (which == 0) {  // q -> [qh | qh]
        *(device half4*)(o + 64 + d0 + j) = h1;
        *(device half4*)(o + 64 + d0 + j + 32) = h2;
      }
    }
  }
}
template [[host_name("mm_qkv_16_2_2")]] kernel void mm_qkv<16, 2, 2>(QKV_ARGS);

// ------------------------------------------------------------------------------------------------------------
// GEMM + row LayerNorm epilogue: the threadgroup owns 16 rows x NC (= TSN WN) columns (grid.y tiles N).
// MODE 0 RESID : X += g * (acc + bias); Y = LN(X)  (X row stride a1, default 384)   [ViT proj / fc2, contact decoder]
// MODE 1 FINAL : X += g * (acc + bias); Y[compact patch row] = LN2(LN(X))            [last fc2: final norm + adapter LN]
// MODE 2 EMBED : X[b*405 + 5 + p] = acc + bias; Y = LN(X)                            [patch embedding]
// MODE 3 PLAIN : Y = gelu(LN(acc + bias))                                             [img_proj]
// MODE 4 SHUF  : as 3, Y scattered as a 2x2 transposed-conv pixel shuffle (a0 grid, NC channels)   [up0]
// ------------------------------------------------------------------------------------------------------------
#define ROWLN_ARGS                                                                                                      \
  const device half *A [[buffer(0)]], const device half *Bw [[buffer(1)]], const device float *bias [[buffer(2)]],     \
      const device float *gam [[buffer(3)]], device float *X [[buffer(4)]], device half *Y [[buffer(5)]],               \
      const device float *lnw [[buffer(6)]], const device float *lnb [[buffer(7)]], const device float *lnw2 [[buffer(8)]], \
      const device float *lnb2 [[buffer(9)]], constant GP &p [[buffer(10)]], uint3 tg [[threadgroup_position_in_grid]], \
      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]
template <int NC>
inline void ln_vals(thread float* v, const device float* w, const device float* b, float eps, ushort lane) {
  float s = 0;
#pragma unroll
  for (short i = 0; i < NC / 32; i++) s += v[i];
  const float mu = simd_sum(s) / NC;
  float q = 0;
#pragma unroll
  for (short i = 0; i < NC / 32; i++) {
    v[i] -= mu;
    q += v[i] * v[i];
  }
  const float rs = rsqrt(simd_sum(q) / NC + eps);
#pragma unroll
  for (short i = 0; i < NC / 32; i++) v[i] = v[i] * rs * w[i * 32 + lane] + b[i * 32 + lane];
}
template <int TSN, int WN, int MODE>
kernel void mm_rowln(ROWLN_ARGS) {
  constexpr int NC = TSN * WN;
  threadgroup float T[16 * NC];
  const int row0 = tg.x * 16, ctile = tg.y * NC, col0 = ctile + sg * TSN;
  const device half* Ab = A + row0 * p.lda;
  const device half* Bb = Bw + col0;
  const int lda = p.lda, ldb = p.ldb;
  mm_tile<16, TSN, 64>([&](int i) { return Ab + i * 64; }, lda, [&](int i) { return Bb + i * 64 * ldb; }, ldb, p.K / 64,
                       T + sg * TSN, NC);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int r = sg; r < 16; r += WN) {
    const int row = row0 + r;
    if (row >= p.M) break;
    float v[NC / 32];
#pragma unroll
    for (short i = 0; i < NC / 32; i++) v[i] = T[r * NC + i * 32 + lane] + bias[ctile + i * 32 + lane];
    if (MODE <= 2) {
      device float* x = X + (MODE == 2 ? (row / 400) * 405 + 5 + row % 400 : row) * (p.a1 ? p.a1 : 384);
#pragma unroll
      for (short i = 0; i < NC / 32; i++) {
        const int c = i * 32 + lane;
        if (MODE <= 1) v[i] = x[c] + gam[c] * v[i];
        x[c] = v[i];
      }
    }
    ln_vals<NC>(v, lnw, lnb, p.eps, lane);
    device half* y;
    if (MODE == 1) {
      const int b = row / 405, n = row % 405;
      if (n < 5) continue;
      ln_vals<NC>(v, lnw2, lnb2, p.eps2, lane);
      y = Y + (b * 400 + n - 5) * 384;
    } else if (MODE == 2) {
      y = Y + ((row / 400) * 405 + 5 + row % 400) * 384;
    } else if (MODE == 4) {
      const int g = p.a0;
      const int b = row / (g * g), rem = row % (g * g), i0 = rem / g, j0 = rem % g;
      const int di = tg.y >> 1, dj = tg.y & 1;
      y = Y + ((b * 2 * g + 2 * i0 + di) * 2 * g + 2 * j0 + dj) * NC;
    } else {
      y = Y + row * p.ldo + ctile;
    }
#pragma unroll
    for (short i = 0; i < NC / 32; i++) y[i * 32 + lane] = half(MODE >= 3 ? gelu(v[i]) : v[i]);
  }
}
#define INST_ROWLN(TSN, WN, MODE)                                                                                  \
  template [[host_name("mm_rowln_" #TSN "_" #WN "_" #MODE)]] kernel void mm_rowln<TSN, WN, MODE>(ROWLN_ARGS);
INST_ROWLN(64, 6, 0)
INST_ROWLN(64, 6, 1)
INST_ROWLN(64, 6, 2)
INST_ROWLN(64, 4, 3)
INST_ROWLN(64, 2, 4)
INST_ROWLN(64, 4, 0)

// token LayerNorm (1280 channels) of the fp32 folded projection + per-image column sums (token mean)
// grid (ceil(400 / 32), B), 8 simdgroups x 4 rows
kernel void ln_tok(const device float* S [[buffer(0)]], const device float* w [[buffer(1)]], const device float* b [[buffer(2)]],
                   device half* tok [[buffer(3)]], device float* colsum [[buffer(4)]], constant GP& p [[buffer(5)]],
                   uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float red[4][1280];
  float cs[40];
#pragma unroll
  for (short i = 0; i < 40; i++) cs[i] = 0;
  for (int k = 0; k < 4; k++) {
    const int r = tg.x * 32 + sg * 4 + k;
    if (r >= 400) break;
    const int row = tg.y * 400 + r;
    float v[40];
#pragma unroll
    for (short i = 0; i < 40; i++) v[i] = S[row * 1280 + i * 32 + lane];
    ln_vals<1280>(v, w, b, p.eps, lane);
#pragma unroll
    for (short i = 0; i < 40; i++) {
      const half h = half(v[i]);
      tok[row * 1280 + i * 32 + lane] = h;
      cs[i] += float(h);
    }
  }
  if (sg >= 4) {
#pragma unroll
    for (short i = 0; i < 40; i++) red[sg - 4][i * 32 + lane] = cs[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < 4) {
#pragma unroll
    for (short i = 0; i < 40; i++) red[sg][i * 32 + lane] += cs[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int c = sg * 32 + lane; c < 1280; c += 256)
    atomic_fetch_add_explicit((device atomic_float*)(colsum + tg.y * 1280 + c), red[0][c] + red[1][c] + red[2][c] + red[3][c],
                              memory_order_relaxed);
}

// ------------------------------------------------------------------------------------------------------------
// 3x3 convolutions, pad 1, as implicit GEMMs over zero-bordered NHWC inputs ((H + 2)^2 x CIN): a simdgroup computes
// 16 consecutive output pixels of one output row (80-wide outputs) x TSN channels; per tap and 32-channel chunk the
// 16 input pixels are a strided view (row stride STRIDE * CIN).
// MODE 0: bias + gelu -> out (zero-bordered (Hout + 2)^2 map, channel stride ldo, channel offset a3)       [stem]
// MODE 1: bias + LayerNorm + gelu -> out (zero-bordered, ldo)                                             [fuse0]
// MODE 2: bias + LayerNorm + gelu, then the 1x1 conv (W2, b2) -> out (dense rows, ldo)                     [fuse1+2]
// p.a0 = Hin (unpadded), p.a2 = Hout
// ------------------------------------------------------------------------------------------------------------
#define CONV_ARGS                                                                                                       \
  const device half *Xin [[buffer(0)]], const device half *Bw [[buffer(1)]], const device float *bias [[buffer(2)]],   \
      const device float *lnw [[buffer(3)]], const device float *lnb [[buffer(4)]], const device half *W2 [[buffer(5)]], \
      const device float *b2 [[buffer(6)]], device half *out [[buffer(7)]], constant GP &p [[buffer(8)]],               \
      uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],                          \
      ushort lane [[thread_index_in_simdgroup]]
template <int TSN, int CIN, int STRIDE, int WM, int MODE>
kernel void conv3(CONV_ARGS) {
  threadgroup float T[WM][16 * TSN];
  threadgroup half H[MODE == 2 ? WM : 1][16 * TSN];
  const int Hin = p.a0 + 2, Ho = p.a2, tiles = Ho / 16;
  const int t = tg.x * WM + sg;
  const int b = t / (Ho * tiles), rem = t % (Ho * tiles), oy = rem / tiles, ox0 = (rem % tiles) * 16;
  const bool live = b < p.a1;  // p.a1 = batch
  const device half* base = Xin + ((min(b, p.a1 - 1) * Hin + oy * STRIDE) * Hin + ox0 * STRIDE) * CIN;
  constexpr int CH = CIN / 32;
  mm_tile<16, TSN, 32>(
      [&](int i) {
        const int tap = i / CH, c = (i % CH) * 32;
        return base + ((tap / 3) * Hin + tap % 3) * CIN + c;
      },
      STRIDE * CIN, [&](int i) { return Bw + i * 32 * TSN; }, TSN, 9 * CH, T[sg], TSN);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (!live) return;
  if (MODE == 0) {
    const int G = Ho + 2;
    for (int e = lane * 4; e < 16 * TSN; e += 128) {
      const int r = e / TSN, c = e % TSN;
      float4 v = *(threadgroup float4*)(T[sg] + e) + *(const device float4*)(bias + c);
      v = float4(gelu(v.x), gelu(v.y), gelu(v.z), gelu(v.w));
      *(device half4*)(out + ((b * G + oy + 1) * G + ox0 + r + 1) * p.ldo + p.a3 + c) = half4(v);
    }
    return;
  }
  // LayerNorm over TSN (= 64) channels: 2 lanes per pixel row, 32 channels each
  const int r = lane >> 1, c0 = (lane & 1) * (TSN / 2);
  float v[TSN / 2];
  float s = 0;
#pragma unroll
  for (short j = 0; j < TSN / 2; j++) {
    v[j] = T[sg][r * TSN + c0 + j] + bias[c0 + j];
    s += v[j];
  }
  const float mu = (s + simd_shuffle_xor(s, ushort(1))) / TSN;
  float q = 0;
#pragma unroll
  for (short j = 0; j < TSN / 2; j++) {
    v[j] -= mu;
    q += v[j] * v[j];
  }
  const float rs = rsqrt((q + simd_shuffle_xor(q, ushort(1))) / TSN + p.eps);
#pragma unroll
  for (short j = 0; j < TSN / 2; j++) v[j] = gelu(v[j] * rs * lnw[c0 + j] + lnb[c0 + j]);
  if (MODE == 1) {
    const int G = Ho + 2;
    device half* o = out + ((b * G + oy + 1) * G + ox0 + r + 1) * p.ldo + c0;
#pragma unroll
    for (short j = 0; j < TSN / 2; j += 4) *(device half4*)(o + j) = half4(v[j], v[j + 1], v[j + 2], v[j + 3]);
    return;
  }
  // MODE 2: fused 1x1 conv on the normalised tile
#pragma unroll
  for (short j = 0; j < TSN / 2; j++) H[sg][r * TSN + c0 + j] = half(v[j]);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  mm_tg<16, TSN, TSN, false, false>((const threadgroup half*)H[sg], TSN, W2, TSN, T[sg], TSN);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  device half* o = out + ((b * Ho + oy) * Ho + ox0 + r) * p.ldo + c0;
#pragma unroll
  for (short j = 0; j < TSN / 2; j += 4) {
    const float4 x = *(threadgroup float4*)(T[sg] + r * TSN + c0 + j) + *(const device float4*)(b2 + c0 + j);
    *(device half4*)(o + j) = half4(x);
  }
}
#define INST_CONV(TSN, CIN, STRIDE, WM, MODE)                                                                     \
  template [[host_name("conv3_" #TSN "_" #CIN "_" #STRIDE "_" #WM "_" #MODE)]] kernel void                       \
  conv3<TSN, CIN, STRIDE, WM, MODE>(CONV_ARGS);
INST_CONV(32, 32, 2, 4, 0)
INST_CONV(64, 96, 1, 4, 1)
INST_CONV(64, 64, 1, 4, 2)

// first stem conv: 3 -> 32 channels, 3x3 stride 2 pad 1 on the normalised image, gelu; direct (K = 27);
// output zero-bordered (162 x 162 x 32)
struct PrepP {
  int B;
};
kernel void stem0(const device float* img [[buffer(0)]], constant float* mean [[buffer(1)]], constant float* stdv [[buffer(2)]],
                  constant float* w [[buffer(3)]], constant float* bias [[buffer(4)]], device half* out [[buffer(5)]],
                  constant PrepP& p [[buffer(6)]], uint t [[thread_position_in_grid]]) {
  const int S = 320, H = 160;
  if (t >= uint(p.B * H * H)) return;
  const int b = t / (H * H), rem = t % (H * H), oy = rem / H, ox = rem % H;
  float in[27];
#pragma unroll
  for (short ky = 0; ky < 3; ky++)
#pragma unroll
    for (short kx = 0; kx < 3; kx++) {
      const int y = oy * 2 + ky - 1, x = ox * 2 + kx - 1;
      const bool ok = y >= 0 && y < S && x >= 0 && x < S;
#pragma unroll
      for (short c = 0; c < 3; c++)
        in[(ky * 3 + kx) * 3 + c] = ok ? (img[((b * 3 + c) * S + y) * S + x] - mean[c]) / stdv[c] : 0.0f;
    }
  device half* o = out + ((b * (H + 2) + oy + 1) * (H + 2) + ox + 1) * 32;
#pragma unroll
  for (short oc = 0; oc < 32; oc += 4) {
    float4 acc = float4(bias[oc], bias[oc + 1], bias[oc + 2], bias[oc + 3]);
#pragma unroll
    for (short k = 0; k < 27; k++)  // weights (O, kh, kw, I): uniform across the simdgroup -> constant-cache broadcast
      acc += in[k] * float4(w[oc * 27 + k], w[(oc + 1) * 27 + k], w[(oc + 2) * 27 + k], w[(oc + 3) * 27 + k]);
    *(device half4*)(o + oc) = half4(gelu(acc.x), gelu(acc.y), gelu(acc.z), gelu(acc.w));
  }
}

// ------------------------------------------------------------------------------------------------------------
// flash attention: a simdgroup owns 16 query rows; kv chunks of 64; S = Q K^T and O += P V are tensor ops /
// simdgroup matrices with S, P, O staged in threadgroup memory; online softmax in fp32 (exp2 domain), 2 lanes per row.
// ViT: q [B, heads, NKP, 128], k transposed [B, heads, 128, NKP], v [B, heads, NKP, 64] (tokens >= NK zero), out
//   [B*NQ, heads*64] fp16. k carries an fp16 hi/lo split, q = [qh | qh], k = [kh ; kl], so that the K = 128 product is
//   qh.k with k to ~fp32 accuracy
//   (fp16 rounding of k is the largest single error source of the network otherwise: p99 uv error 0.61 -> 0.38 px).
// Heatmap (SPLIT): q [B, NQP, 64]; k = v = ff [B, NK, DV] (cols 0..63 features, 64/65 pixel x/y + 0.5, 66 ones);
//   per kv split: o [B, NS, NQP, DV], (m, l) [B, NS, NQP, 2] fp32.
// ------------------------------------------------------------------------------------------------------------
struct AP {
  int NQ, NK, ldq, ldk, ldv, ldo;
  int qbs, kbs, vbs;  // per (b, head) strides (elements)
  int heads, kv_per_split, NQP;
  float scale;  // softmax scale * log2(e)
};
#define ATTN_ARGS                                                                                                     \
  const device half *Q [[buffer(0)]], const device half *K [[buffer(1)]], const device half *V [[buffer(2)]],         \
      device half *O [[buffer(3)]], device float *OP [[buffer(4)]], device float *ML [[buffer(5)]],                    \
      constant AP &p [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],                                         \
      ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]
template <int DK, int DV, int WM, int KS, bool SPLIT, bool KT, int QR>
kernel void attn(ATTN_ARGS) {
  constexpr int LPR = 32 / QR, CPL = 64 / LPR, DVL = DV / LPR;  // lanes per query row, S / O columns per lane
  // simdgroup sg: query tile qt = sg / KS, key split ks = sg % KS (chunks ks, ks + KS, ...), merged at the end
  threadgroup float SPs[WM * KS][QR * 64];  // S (fp32), then P (fp16) in place
  threadgroup float Os[WM * KS][QR * DV];
  threadgroup float MLs[WM * KS][QR * 2];
  const int qt = sg / KS, ks = sg % KS;
  const int b = tg.z, bh = SPLIT ? int(tg.z) : int(tg.y + tg.z * p.heads);
  const int q0 = tg.x * QR * WM + qt * QR;
  const device half* Qp = Q + bh * p.qbs + q0 * p.ldq;
  K += bh * p.kbs;
  V += bh * p.vbs;
  int kbeg = 0, kend = p.NK;
  if (SPLIT) {
    kbeg = tg.y * p.kv_per_split;
    kend = min(kend, kbeg + p.kv_per_split);
  }
  threadgroup float* S = SPs[sg];
  threadgroup half* P = (threadgroup half*)SPs[sg];
  threadgroup float* Ob = Os[sg];
  const int r = lane / LPR, h = lane % LPR;
#pragma unroll
  for (short c = 0; c < DVL; c += 4) *(threadgroup float4*)(Ob + r * DV + h * DVL + c) = 0;
  float m = -INFINITY, l = 0;
  for (int kb = kbeg + ks * 64; kb < kend; kb += 64 * KS) {
    if (KT) mm_tg<QR, 64, DK, false, false>(Qp, p.ldq, K + kb, p.ldk, S, 64);
    else mm_tg<QR, 64, DK, true, false>(Qp, p.ldq, K + kb * p.ldk, p.ldk, S, 64);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    float sv[CPL];
    float mx = -INFINITY;
#pragma unroll
    for (short j = 0; j < CPL; j += 4) {
      const float4 x = *(threadgroup float4*)(S + r * 64 + h * CPL + j) * p.scale;
#pragma unroll
      for (short u = 0; u < 4; u++) {
        sv[j + u] = (kb + h * CPL + j + u < kend) ? x[u] : -INFINITY;
        mx = max(mx, sv[j + u]);
      }
    }
    if (LPR == 2) mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
    const float mn = max(m, mx), fac = exp2(m - mn);
    simdgroup_barrier(mem_flags::mem_threadgroup);  // every lane has read its S before P overwrites it
    float sum = 0;
#pragma unroll
    for (short j = 0; j < CPL; j += 4) {
      half4 hv;
#pragma unroll
      for (short u = 0; u < 4; u++) {
        const float e = exp2(sv[j + u] - mn);
        sum += e;
        hv[u] = half(e);
      }
      *(threadgroup half4*)(P + r * 64 + h * CPL + j) = hv;
    }
    if (LPR == 2) sum += simd_shuffle_xor(sum, ushort(1));
    l = l * fac + sum;
    m = mn;
#pragma unroll
    for (short c = 0; c < DVL; c += 4) *(threadgroup float4*)(Ob + r * DV + h * DVL + c) *= fac;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    mm_tg<QR, DV, 64, false, true>((const threadgroup half*)P, 64, V + kb * p.ldv, p.ldv, Ob, DV);
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }
  float4 acc[DVL / 4];
#pragma unroll
  for (short c = 0; c < DVL / 4; c++) acc[c] = *(threadgroup float4*)(Ob + r * DV + h * DVL + c * 4);
  if (KS > 1) {  // merge the key splits of this query tile into split 0
    if (h == 0) {
      MLs[sg][r * 2] = m;
      MLs[sg][r * 2 + 1] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ks != 0) return;
    float M = m;
    for (short t = 1; t < KS; t++) M = max(M, MLs[sg + t][r * 2]);
    const float w0 = exp2(m - M);
    l *= w0;
#pragma unroll
    for (short c = 0; c < DVL / 4; c++) acc[c] *= w0;
    for (short t = 1; t < KS; t++) {
      const float w = exp2(MLs[sg + t][r * 2] - M);
      l += w * MLs[sg + t][r * 2 + 1];
#pragma unroll
      for (short c = 0; c < DVL / 4; c++) acc[c] += w * *(threadgroup float4*)(Os[sg + t] + r * DV + h * DVL + c * 4);
    }
    m = M;
  }
  const int q = q0 + r;
  if (q >= p.NQ) return;
  if (!SPLIT) {
    const float il = 1.0f / l;
    device half* dst = O + (b * p.NQ + q) * p.ldo + tg.y * DV + h * DVL;
#pragma unroll
    for (short c = 0; c < DVL / 4; c++) *(device half4*)(dst + c * 4) = half4(acc[c] * il);
  } else {
    const int NS = (p.NK + p.kv_per_split - 1) / p.kv_per_split;
    const int idx = (b * NS + tg.y) * p.NQP + q;
    device float* dst = OP + idx * DV + h * DVL;
#pragma unroll
    for (short c = 0; c < DVL / 4; c++) *(device float4*)(dst + c * 4) = acc[c];
    if (h == 0) {
      ML[idx * 2] = m;
      ML[idx * 2 + 1] = l;
    }
  }
}

// Register-resident flash attention on simdgroup_matrix 8x8 (the structure of MLX's steel_attention), used by the
// simdgroup_matrix backend: a simdgroup owns 16 query rows (S, P and O stay in registers as per-lane 2-element
// fragments, turned into simdgroup matrices only for the multiply, as MLX's BaseMMAFrag; element-wise access to
// simdgroup_matrix::thread_elements() is pathologically slow), the threadgroup stages Q, the K chunk (transposed)
// and the V chunk (32 keys) in threadgroup memory. 8x8 layout: lane -> row fm = (qid & 4) + ((lane >> 1) & 3), cols
// fn, fn + 1 with fn = (qid & 2) * 2 + (lane & 1) * 2 (qid = lane >> 2); lanes sharing a row differ in bits 0 and 3.
typedef vec<float, 2> f2;
typedef vec<half, 2> h2;
inline void mma8(thread f2& d, thread const h2& a, thread const h2& b) {  // d += a b (8x8 fragments)
  simdgroup_half8x8 A, B;
  simdgroup_float8x8 C;
  reinterpret_cast<thread h2&>(A.thread_elements()) = a;
  reinterpret_cast<thread h2&>(B.thread_elements()) = b;
  reinterpret_cast<thread f2&>(C.thread_elements()) = d;
  simdgroup_multiply_accumulate(C, A, B, C);
  d = reinterpret_cast<thread f2&>(C.thread_elements());
}
template <int DK, int DV, int WM, bool SPLIT, bool KT>
kernel void attn_sg(ATTN_ARGS) {
  constexpr int LK = 32 + 8, LV = DV + 8, LQ = DK + 8;  // padded threadgroup row strides
  threadgroup half Ks[DK * LK];                        // K chunk transposed: [dim][key]
  threadgroup half Vs[32 * LV];
  threadgroup half Qs[WM][16 * LQ];
  const int b = tg.z, bh = SPLIT ? int(tg.z) : int(tg.y + tg.z * p.heads);
  const int q0 = tg.x * 16 * WM + sg * 16;
  const device half* Qp = Q + bh * p.qbs + q0 * p.ldq;
  K += bh * p.kbs;
  V += bh * p.vbs;
  int kbeg = 0, kend = p.NK;
  if (SPLIT) {
    kbeg = tg.y * p.kv_per_split;
    kend = min(kend, kbeg + p.kv_per_split);
  }
  const short qid = lane >> 2;
  const short fm = (qid & 4) + ((lane >> 1) & 3), fn = (qid & 2) * 2 + (lane & 1) * 2;
  for (int e = lane; e < 16 * (DK / 4); e += 32) {
    const int r = e / (DK / 4), c = (e % (DK / 4)) * 4;
    *(threadgroup half4*)(Qs[sg] + r * LQ + c) = *(const device half4*)(Qp + r * p.ldq + c);
  }
  f2 o[2][DV / 8];
#pragma unroll
  for (short i = 0; i < 2; i++)
#pragma unroll
    for (short n = 0; n < DV / 8; n++) o[i][n] = 0;
  float m[2] = {-INFINITY, -INFINITY}, l[2] = {0, 0};
  const ushort tid = sg * 32 + lane;
  threadgroup const half* qrow0 = Qs[sg] + fm * LQ + fn;
  threadgroup const half* qrow1 = Qs[sg] + (fm + 8) * LQ + fn;
  for (int kb = kbeg; kb < kend; kb += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (KT) {  // K given as [dim][token]: copy rows
      for (int e = tid; e < DK * 8; e += WM * 32) {
        const int d = e >> 3, c = (e & 7) * 4;
        *(threadgroup half4*)(Ks + d * LK + c) = *(const device half4*)(K + d * p.ldk + kb + c);
      }
    } else {
      for (int e = tid; e < 32 * (DK / 4); e += WM * 32) {
        const int r = e & 31, c = (e >> 5) * 4;
        const half4 x = *(const device half4*)(K + (kb + r) * p.ldk + c);
        Ks[(c + 0) * LK + r] = x[0];
        Ks[(c + 1) * LK + r] = x[1];
        Ks[(c + 2) * LK + r] = x[2];
        Ks[(c + 3) * LK + r] = x[3];
      }
    }
    for (int e = tid; e < 32 * (DV / 4); e += WM * 32) {
      const int r = e / (DV / 4), c = (e % (DV / 4)) * 4;
      *(threadgroup half4*)(Vs + r * LV + c) = *(const device half4*)(V + (kb + r) * p.ldv + c);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    f2 sc[2][4];
#pragma unroll
    for (short j = 0; j < 4; j++) {
      sc[0][j] = 0;
      sc[1][j] = 0;
    }
#pragma unroll
    for (short k = 0; k < DK / 8; k++) {
      const h2 a0 = *(threadgroup const h2*)(qrow0 + k * 8);
      const h2 a1 = *(threadgroup const h2*)(qrow1 + k * 8);
#pragma unroll
      for (short j = 0; j < 4; j++) {
        const h2 kt = *(threadgroup const h2*)(Ks + (k * 8 + fm) * LK + j * 8 + fn);
        mma8(sc[0][j], a0, kt);
        mma8(sc[1][j], a1, kt);
      }
    }
    h2 pm[2][4];
#pragma unroll
    for (short i = 0; i < 2; i++) {
      float mx = -INFINITY;
#pragma unroll
      for (short j = 0; j < 4; j++) {
        sc[i][j] *= p.scale;
        if (kb + j * 8 + fn >= kend) sc[i][j][0] = -INFINITY;
        if (kb + j * 8 + fn + 1 >= kend) sc[i][j][1] = -INFINITY;
        mx = max(mx, max(sc[i][j][0], sc[i][j][1]));
      }
      mx = max(mx, simd_shuffle_xor(mx, ushort(1)));
      mx = max(mx, simd_shuffle_xor(mx, ushort(8)));
      const float mn = max(m[i], mx), fac = exp2(m[i] - mn);
      float sum = 0;
#pragma unroll
      for (short j = 0; j < 4; j++) {
        const f2 e = exp2(sc[i][j] - mn);
        sum += e[0] + e[1];
        pm[i][j] = h2(e);
      }
      sum += simd_shuffle_xor(sum, ushort(1));
      sum += simd_shuffle_xor(sum, ushort(8));
      l[i] = l[i] * fac + sum;
      m[i] = mn;
#pragma unroll
      for (short n = 0; n < DV / 8; n++) o[i][n] *= fac;
    }
#pragma unroll
    for (short j = 0; j < 4; j++)
#pragma unroll
      for (short n = 0; n < DV / 8; n++) {
        const h2 vm = *(threadgroup const h2*)(Vs + (j * 8 + fm) * LV + n * 8 + fn);
        mma8(o[0][n], pm[0][j], vm);
        mma8(o[1][n], pm[1][j], vm);
      }
  }
#pragma unroll
  for (short i = 0; i < 2; i++) {
    const int q = q0 + i * 8 + fm;
    if (q >= p.NQ) continue;
    if (!SPLIT) {
      const float il = 1.0f / l[i];
      device half* dst = O + (b * p.NQ + q) * p.ldo + tg.y * DV + fn;
#pragma unroll
      for (short n = 0; n < DV / 8; n++) *(device h2*)(dst + n * 8) = h2(o[i][n] * il);
    } else {
      const int NS = (p.NK + p.kv_per_split - 1) / p.kv_per_split;
      const int idx = (b * NS + tg.y) * p.NQP + q;
      device float* dst = OP + idx * DV + fn;
#pragma unroll
      for (short n = 0; n < DV / 8; n++) *(device f2*)(dst + n * 8) = o[i][n];
      if (fn == 0) {
        ML[idx * 2] = m[i];
        ML[idx * 2 + 1] = l[i];
      }
    }
  }
}
template [[host_name("attn_sg_vit")]] kernel void attn_sg<128, 64, 2, false, true>(ATTN_ARGS);
template [[host_name("attn_sg_heat")]] kernel void attn_sg<64, 96, 2, true, false>(ATTN_ARGS);

#define INST_ATTN(NAME, DK, DV, WM, KS, SPLIT, KT, QR) \
  template [[host_name(NAME)]] kernel void attn<DK, DV, WM, KS, SPLIT, KT, QR>(ATTN_ARGS);
INST_ATTN("attn_vit", 128, 64, 2, 1, false, true, 16)
INST_ATTN("attn_heat", 64, 96, 2, 1, true, false, 16)

// ------------------------------------------------------------------------------------------------------------
// prep: im2col patches, ray embedding, prefix rows of the residual stream, zero the token column sums
// ------------------------------------------------------------------------------------------------------------
kernel void prep(const device float* img [[buffer(0)]], const device float* ray [[buffer(1)]],
                 const device float* mean [[buffer(2)]], const device float* stdv [[buffer(3)]],
                 const device float* uv_ds [[buffer(4)]], const device float* freq [[buffer(5)]],
                 const device float* prefix_x [[buffer(6)]], const device half* prefix_y [[buffer(7)]],
                 device half* patches [[buffer(8)]], device half* hh [[buffer(9)]], device float* X [[buffer(10)]],
                 device half* Y [[buffer(11)]], device float* colsum [[buffer(12)]], constant PrepP& p [[buffer(13)]],
                 uint t [[thread_position_in_grid]]) {
  const int S = 320, G = 20, NP = 400;
  if (t < uint(p.B * NP * 256)) {  // one thread per input pixel: 3 channels -> patch column (ky*16+kx)*3 + c
    const int b = t / (NP * 256), rem = t % (NP * 256);
    const int pi = rem / 256, pix = rem % 256;
    const int py = pi / G, px = pi % G, ky = pix / 16, kx = pix % 16;
    const int y = py * 16 + ky, x = px * 16 + kx;
    const device float* src = img + b * 3 * S * S + y * S + x;
    device half* dst = patches + (b * NP + pi) * 768 + pix * 3;
#pragma unroll
    for (short c = 0; c < 3; c++) dst[c] = half((src[c * S * S] - mean[c]) / stdv[c]);
  }
  if (t < uint(p.B * NP * 128)) {  // ray embedding -> hh[:, 1280:1408]
    const int b = t / (NP * 128), rem = t % (NP * 128);
    const int tok = rem / 128, j = rem % 128;
    float v = 0;
    if (j < 99) {
      const device float* rr = ray + b * 4;
      float r3[3] = {uv_ds[tok * 2] * rr[0] + rr[1], uv_ds[tok * 2 + 1] * rr[2] + rr[3], 1.0f};
      if (j < 3) {
        v = r3[j];
      } else {
        const int f = (j - 3) % 48;
        const float ph = r3[f / 16] * freq[f] * M_PI_F;
        v = j < 51 ? precise::sin(ph) : precise::cos(ph);
      }
    }
    hh[(b * NP + tok) * 1408 + 1280 + j] = half(v);
  }
  if (t < uint(p.B * 5 * 384)) {  // prefix tokens (and block 0's LN1 of them, constant)
    const int b = t / (5 * 384), rem = t % (5 * 384);
    X[b * 405 * 384 + rem] = prefix_x[rem];
    Y[b * 405 * 384 + rem] = prefix_y[rem];
  }
  if (t < uint(p.B * 1280)) colsum[t] = 0;
}


// ------------------------------------------------------------------------------------------------------------
// pose token: pose = gelu(LN([mean_tok, cond]) @ W + b), W transposed (1024, KP); one simdgroup per output column
// ------------------------------------------------------------------------------------------------------------
struct PoseP {
  int B, KIN, KP, NOUT;
  float eps;
  float inv_tokens;
};
kernel void pose_gemv(const device float* colsum [[buffer(0)]], const device float* cond [[buffer(1)]],
                      const device float* lnw [[buffer(2)]], const device float* lnb [[buffer(3)]],
                      const device half* Wt [[buffer(4)]], const device float* bias [[buffer(5)]],
                      device float* pose [[buffer(6)]], constant PoseP& p [[buffer(7)]],
                      uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float xin[4][1312];
  threadgroup float stat[4][2];
  // input vectors (B <= 4) + LayerNorm (KIN = 1283)
  for (int b = 0; b < p.B; b++)
    for (int k = tid; k < p.KP; k += 256)
      xin[b][k] = k < 1280 ? colsum[b * 1280 + k] * p.inv_tokens : (k < p.KIN ? cond[b * 3 + k - 1280] : 0.0f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < p.B) {
    float s = 0, q = 0;
    for (int k = lane; k < p.KIN; k += 32) s += xin[sg][k];
    const float mu = simd_sum(s) / p.KIN;
    for (int k = lane; k < p.KIN; k += 32) {
      const float d = xin[sg][k] - mu;
      q += d * d;
    }
    const float var = simd_sum(q) / p.KIN;
    if (lane == 0) {
      stat[sg][0] = mu;
      stat[sg][1] = rsqrt(var + p.eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int b = 0; b < p.B; b++)
    for (int k = tid; k < p.KIN; k += 256) xin[b][k] = (xin[b][k] - stat[b][0]) * stat[b][1] * lnw[k] + lnb[k];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int n = tg * 8 + sg;
  if (n >= p.NOUT) return;
  const device half* w = Wt + n * p.KP;
  float acc[4] = {0, 0, 0, 0};
  for (int k = lane * 4; k < p.KP; k += 128) {
    const float4 wv = float4(*(const device half4*)(w + k));
    for (int b = 0; b < p.B; b++)
      acc[b] += dot(wv, float4(xin[b][k], xin[b][k + 1], xin[b][k + 2], xin[b][k + 3]));
  }
  for (int b = 0; b < p.B; b++) {
    const float s = simd_sum(acc[b]);
    if (lane == 0) pose[b * p.NOUT + n] = gelu(s + bias[n]);
  }
}

// pose MLP first layers: [query0 (512, gelu) | depth0 (256, gelu of LN(pose)) | verify0 (256, gelu) | xyz pose part (256)]
kernel void pose_heads(const device float* pose [[buffer(0)]], const device float* dlnw [[buffer(1)]],
                       const device float* dlnb [[buffer(2)]], const device half* Wt [[buffer(3)]],
                       const device float* bias [[buffer(4)]], device float* ph [[buffer(5)]],
                       constant PoseP& p [[buffer(6)]], uint tg [[threadgroup_position_in_grid]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                       ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float xin[4][1024];
  threadgroup float stat[4][2];
  for (int b = 0; b < p.B; b++)
    for (int k = tid; k < 1024; k += 256) xin[b][k] = pose[b * 1024 + k];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg < p.B) {
    float s = 0, q = 0;
    for (int k = lane; k < 1024; k += 32) s += xin[sg][k];
    const float mu = simd_sum(s) / 1024;
    for (int k = lane; k < 1024; k += 32) {
      const float d = xin[sg][k] - mu;
      q += d * d;
    }
    const float var = simd_sum(q) / 1024;
    if (lane == 0) {
      stat[sg][0] = mu;
      stat[sg][1] = rsqrt(var + p.eps);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int n = tg * 8 + sg;
  if (n >= 1280) return;
  const bool use_ln = n >= 512 && n < 768;  // depth head reads LN(pose)
  const device half* w = Wt + n * 1024;
  float acc[4] = {0, 0, 0, 0};
  for (int k = lane * 4; k < 1024; k += 128) {
    const float4 wv = float4(*(const device half4*)(w + k));
    float4 lw = 1.0f, lb = 0.0f;
    if (use_ln) {
      lw = float4(dlnw[k], dlnw[k + 1], dlnw[k + 2], dlnw[k + 3]);
      lb = float4(dlnb[k], dlnb[k + 1], dlnb[k + 2], dlnb[k + 3]);
    }
    for (int b = 0; b < p.B; b++) {
      float4 x = float4(xin[b][k], xin[b][k + 1], xin[b][k + 2], xin[b][k + 3]);
      if (use_ln) x = (x - stat[b][0]) * stat[b][1] * lw + lb;
      acc[b] += dot(wv, x);
    }
  }
  for (int b = 0; b < p.B; b++) {
    const float s = simd_sum(acc[b]) + bias[n];
    if (lane == 0) ph[b * 1280 + n] = n < 1024 ? gelu(s) : s;
  }
}

// ------------------------------------------------------------------------------------------------------------
// point queries: qq = h_q @ q1 + b (64); q[p] = gelu(LN(qq + point[p]) @ m0 + b0) @ m1 + b1; also depth / verify
// grid (ceil(NP / 32), B), 256 threads
// ------------------------------------------------------------------------------------------------------------
struct PQP {
  int NP, NPP;
  float eps;
};
kernel void point_query(const device float* ph [[buffer(0)]], const device half* q1t [[buffer(1)]],
                        const device float* q1b [[buffer(2)]], const device float* point [[buffer(3)]],
                        const device float* lnw [[buffer(4)]], const device float* lnb [[buffer(5)]],
                        const device half* m0 [[buffer(6)]], const device float* m0b [[buffer(7)]],
                        const device half* m1 [[buffer(8)]], const device float* m1b [[buffer(9)]],
                        const device float* d1w [[buffer(10)]], const device float* d1b [[buffer(11)]],
                        const device float* a1t [[buffer(12)]], const device float* a1b [[buffer(13)]],
                        device half* Qout [[buffer(14)]], device float* out_depth [[buffer(15)]],
                        device float* out_ver [[buffer(16)]], constant PQP& p [[buffer(17)]],
                        uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float qq[64];
  threadgroup float z[32][65];
  threadgroup float hdn[32][65];
  const int b = tg.y;
  const device float* h = ph + b * 1280;
  for (int o = sg; o < 64; o += 8) {
    float s = 0;
    for (int k = lane; k < 512; k += 32) s += float(q1t[o * 512 + k]) * h[k];
    s = simd_sum(s);
    if (lane == 0) qq[o] = s + q1b[o];
  }
  if (tg.x == 0) {
    if (sg == 0) {
      float s = 0;
      for (int k = lane; k < 256; k += 32) s += d1w[k] * h[512 + k];
      s = simd_sum(s);
      if (lane == 0) out_depth[b] = s + d1b[0];
    } else if (sg <= 2) {
      const int o = sg - 1;
      float s = 0;
      for (int k = lane; k < 256; k += 32) s += a1t[o * 256 + k] * h[768 + k];
      s = simd_sum(s);
      if (lane == 0) out_ver[b * 2 + o] = s + a1b[o];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int p0 = tg.x * 32;
  for (int i = sg; i < 32; i += 8) {  // LN(qq + point)
    const int pt = min(p0 + i, p.NP - 1);
    float v0 = qq[lane] + point[pt * 64 + lane], v1 = qq[lane + 32] + point[pt * 64 + lane + 32];
    const float mu = simd_sum(v0 + v1) / 64;
    v0 -= mu;
    v1 -= mu;
    const float rs = rsqrt(simd_sum(v0 * v0 + v1 * v1) / 64 + p.eps);
    z[i][lane] = v0 * rs * lnw[lane] + lnb[lane];
    z[i][lane + 32] = v1 * rs * lnw[lane + 32] + lnb[lane + 32];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = tid; e < 32 * 64; e += 256) {
    const int i = e / 64, o = e % 64;
    float s = m0b[o];
    for (int k = 0; k < 64; k++) s += z[i][k] * float(m0[k * 64 + o]);
    hdn[i][o] = gelu(s);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = tid; e < 32 * 64; e += 256) {
    const int i = e / 64, o = e % 64;
    if (p0 + i >= p.NP) continue;
    float s = m1b[o];
    for (int k = 0; k < 64; k++) s += hdn[i][k] * float(m1[k * 64 + o]);
    Qout[(b * p.NPP + p0 + i) * 64 + o] = half(s);
  }
}

// ------------------------------------------------------------------------------------------------------------
// point heads: combine the heatmap splits -> pooled features, uv, conf; then xyz / vis MLPs on [pooled, q]
// grid (ceil(NP / 8), B), 256 threads
// ------------------------------------------------------------------------------------------------------------
struct PHP {
  int NP, NPP, NS;
  float eps;
  float inv_w;  // 1 / heatmap width (pixel-centre coordinates are stored as j + 0.5)
};
kernel void point_heads(const device float* OP [[buffer(0)]], const device float* ML [[buffer(1)]],
                        const device half* Qin [[buffer(2)]], const device float* ph [[buffer(3)]],
                        const device half* xa [[buffer(4)]], const device float* x2t [[buffer(5)]],
                        const device float* x2b [[buffer(6)]], const device float* vlnw [[buffer(7)]],
                        const device float* vlnb [[buffer(8)]], const device half* v0 [[buffer(9)]],
                        const device float* v0b [[buffer(10)]], const device float* v1 [[buffer(11)]],
                        const device float* v1b [[buffer(12)]], device float* out_uv [[buffer(13)]],
                        device float* out_xyz [[buffer(14)]], device float* out_conf [[buffer(15)]],
                        device float* out_vis [[buffer(16)]], device float* out_geo [[buffer(17)]],
                        constant PHP& p [[buffer(18)]],
                        uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float pq[8][129];
  threadgroup float pl[8][129];
  threadgroup float hx[8][257];
  threadgroup float hv[8][65];
  const int b = tg.y, p0 = tg.x * 8;
  // combine splits: one simdgroup per point; lanes over the 96 columns (3 per lane)
  for (int i = sg; i < 8; i += 8) {
    const int pt = min(p0 + i, p.NP - 1);
    float mmax = -INFINITY;
    for (int s = 0; s < p.NS; s++) mmax = max(mmax, ML[((b * p.NS + s) * p.NPP + pt) * 2]);
    float l = 0, acc[3] = {0, 0, 0};
    for (int s = 0; s < p.NS; s++) {
      const int idx = (b * p.NS + s) * p.NPP + pt;
      const float w = exp2(ML[idx * 2] - mmax);
      l += w * ML[idx * 2 + 1];
      for (int j = 0; j < 3; j++) acc[j] += w * OP[idx * 96 + lane + 32 * j];
    }
    // column 66 = sum of the (fp16) probabilities: normalise by it
    const float ones = simd_shuffle(acc[2], ushort(2));
    const float inv = 1.0f / ones;
    pq[i][lane] = acc[0] * inv;
    pq[i][lane + 32] = acc[1] * inv;
    pq[i][64 + lane] = float(Qin[(b * p.NPP + pt) * 64 + lane]);
    pq[i][96 + lane] = float(Qin[(b * p.NPP + pt) * 64 + 32 + lane]);
    if (lane == 0 && p0 + i < p.NP) {
      out_uv[(b * p.NP + pt) * 2] = acc[2] * inv * p.inv_w;  // col 64 (lane 0 of group 2)
      out_conf[b * p.NP + pt] = 1.0f / l;
    }
    if (lane == 1 && p0 + i < p.NP) out_uv[(b * p.NP + pt) * 2 + 1] = acc[2] * inv * p.inv_w;  // col 65
    if (p0 + i < p.NP)
      for (int c = lane; c < 128; c += 32) out_geo[(b * p.NP + pt) * 132 + c] = pq[i][c];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // vis LayerNorm of pq
  for (int i = sg; i < 8; i += 8) {
    float v[4], s = 0;
    for (int j = 0; j < 4; j++) {
      v[j] = pq[i][lane + 32 * j];
      s += v[j];
    }
    const float mu = simd_sum(s) / 128;
    float q = 0;
    for (int j = 0; j < 4; j++) {
      v[j] -= mu;
      q += v[j] * v[j];
    }
    const float rs = rsqrt(simd_sum(q) / 128 + p.eps);
    for (int j = 0; j < 4; j++) pl[i][lane + 32 * j] = v[j] * rs * vlnw[lane + 32 * j] + vlnb[lane + 32 * j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // hidden layers: xyz 256 (pq @ xa + pose part), vis 64 (pl @ v0)
  const device float* xpb = ph + b * 1280 + 1024;
  for (int o = tid; o < 256 + 64; o += 256) {
    if (o < 256) {
      float s[8];
      for (int i = 0; i < 8; i++) s[i] = xpb[o];
      for (int k = 0; k < 128; k++) {
        const float w = float(xa[k * 256 + o]);
        for (int i = 0; i < 8; i++) s[i] += pq[i][k] * w;
      }
      for (int i = 0; i < 8; i++) hx[i][o] = gelu(s[i]);
    } else {
      const int oo = o - 256;
      float s[8];
      for (int i = 0; i < 8; i++) s[i] = v0b[oo];
      for (int k = 0; k < 128; k++) {
        const float w = float(v0[k * 64 + oo]);
        for (int i = 0; i < 8; i++) s[i] += pl[i][k] * w;
      }
      for (int i = 0; i < 8; i++) hv[i][oo] = gelu(s[i]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // outputs: 8 points x (3 xyz + 1 vis) dots; one simdgroup per point
  for (int i = sg; i < 8; i += 8) {
    const int pt = p0 + i;
    float s[3] = {0, 0, 0};
    for (int k = lane; k < 256; k += 32)
      for (int j = 0; j < 3; j++) s[j] += hx[i][k] * x2t[j * 256 + k];
    float sv = 0;
    for (int k = lane; k < 64; k += 32) sv += hv[i][k] * v1[k];
    for (int j = 0; j < 3; j++) s[j] = simd_sum(s[j]);
    sv = simd_sum(sv);
    if (lane == 0 && pt < p.NP) {
      for (int j = 0; j < 3; j++) {
        out_xyz[(b * p.NP + pt) * 3 + j] = s[j] + x2b[j];
        out_geo[(b * p.NP + pt) * 132 + 128 + j] = 10.0f * (s[j] + x2b[j]);
      }
      out_vis[b * p.NP + pt] = sv + v1b[0];
      out_geo[(b * p.NP + pt) * 132 + 131] = sv + v1b[0];
    }
  }
}

// ------------------------------------------------------------------------------------------------------------
// hand-object contact head (FAH keypoint_head.contact_from): a branch of the encoder's last blocks (own weights, on
// its hidden state before them) and the heatmap features (4x4-pooled) form an 800-token memory; 400 dense-point
// tokens (point geometry + pose token + learned positions) go through 4 pre-norm transformer decoder layers
// (self-attention, cross-attention to the memory, ReLU MLP); outputs: a logit per point and per hand at K tolerances.
// ------------------------------------------------------------------------------------------------------------
struct CP {
  int B, rows, a0, a1;
  float eps, eps2;
};

// branch input: copy the residual stream (rows of 384, fp32) and LayerNorm it (the branch's first LN1)
// grid (ceil(rows / 8)), 8 simdgroups x 1 row
kernel void ln_copy(const device float* X [[buffer(0)]], const device float* w [[buffer(1)]], const device float* b [[buffer(2)]],
                    device float* Xo [[buffer(3)]], device half* Y [[buffer(4)]], constant CP& p [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
  const int row = tg * 8 + sg;
  if (row >= p.rows) return;
  float v[12];
  for (short i = 0; i < 12; i++) {
    v[i] = X[row * 384 + i * 32 + lane];
    Xo[row * 384 + i * 32 + lane] = v[i];
  }
  ln_vals<384>(v, w, b, p.eps, lane);
  for (short i = 0; i < 12; i++) Y[row * 384 + i * 32 + lane] = half(v[i]);
}

// memory rows of image b: 0..399 the heatmap features average-pooled 4x4 (80 -> 20 grid) -> 1x1 conv 64 -> 256 ->
// LayerNorm2d -> + sin-cos positions; 400..799 the branch tokens + the same positions. One simdgroup per row.
// ff: (B, 6400, 96) fp16 (features in cols 0..63), bm: (B, 400, 256) fp16; grid (800 / 8, B)
kernel void contact_mem(const device half* ff [[buffer(0)]], const device half* bm [[buffer(1)]],
                        const device half* W [[buffer(2)]], const device float* bias [[buffer(3)]],
                        const device float* lnw [[buffer(4)]], const device float* lnb [[buffer(5)]],
                        const device float* pos [[buffer(6)]], device half* mem [[buffer(7)]], constant CP& p [[buffer(8)]],
                        uint2 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float pooled[8][64];
  const int b = tg.y, t = tg.x * 8 + sg;
  const int n = t % 400;
  float v[8];
  if (t < 400) {
    const int y = n / 20, x = n % 20;
    for (int c = lane; c < 64; c += 32) {
      float s = 0;
      for (int dy = 0; dy < 4; dy++)
        for (int dx = 0; dx < 4; dx++) s += float(ff[((b * 80 + 4 * y + dy) * 80 + 4 * x + dx) * 96 + c]);
      pooled[sg][c] = s * (1.0f / 16);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (short i = 0; i < 8; i++) {
      const int o = i * 32 + lane;
      float s = bias[o];
      for (int k = 0; k < 64; k++) s += pooled[sg][k] * float(W[k * 256 + o]);
      v[i] = s;
    }
    ln_vals<256>(v, lnw, lnb, p.eps, lane);
  } else {
    for (short i = 0; i < 8; i++) v[i] = float(bm[(b * 400 + n) * 256 + i * 32 + lane]);
  }
  for (short i = 0; i < 8; i++) {
    const int o = i * 32 + lane;
    mem[(b * 800 + t) * 256 + o] = half(v[i] + pos[n * 256 + o]);
  }
}

// pose token -> 256 (once per image); grid (1, B), 256 threads
kernel void contact_tok(const device float* pose [[buffer(0)]], const device half* W [[buffer(1)]],
                        const device float* bias [[buffer(2)]], device float* out [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
  const int b = tg.y;
  float s = bias[tid];
  for (int k = 0; k < 1024; k++) s += pose[b * 1024 + k] * float(W[k * 256 + tid]);
  out[b * 256 + tid] = s;
}

// point tokens: geo (the last 400 of the 461 points, 132 values) -> LN -> 132x256 -> GELU -> 256x256, + pose token +
// learned positions -> residual X (fp32) and the first decoder layer's LN1 -> Y (fp16). One simdgroup per point;
// grid (400 / 8, B). a0 = first dense point (461 - 400), a1 = points per image in geo (461)
kernel void contact_points(const device float* geo [[buffer(0)]], const device float* inlnw [[buffer(1)]],
                           const device float* inlnb [[buffer(2)]], const device half* W1 [[buffer(3)]],
                           const device float* b1 [[buffer(4)]], const device half* W2 [[buffer(5)]],
                           const device float* b2 [[buffer(6)]], const device float* tok [[buffer(7)]],
                           const device float* pos [[buffer(8)]], const device float* lnw [[buffer(9)]],
                           const device float* lnb [[buffer(10)]], device float* X [[buffer(11)]], device half* Y [[buffer(12)]],
                           constant CP& p [[buffer(13)]], uint2 tg [[threadgroup_position_in_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float g[8][132];
  threadgroup float h[8][256];
  const int b = tg.y, n = tg.x * 8 + sg;
  const device float* gr = geo + (b * p.a1 + p.a0 + n) * 132;
  float s = 0;
  for (int c = lane; c < 132; c += 32) s += gr[c];
  const float mu = simd_sum(s) / 132;
  float q = 0;
  for (int c = lane; c < 132; c += 32) q += (gr[c] - mu) * (gr[c] - mu);
  const float rs = rsqrt(simd_sum(q) / 132 + p.eps);
  for (int c = lane; c < 132; c += 32) g[sg][c] = (gr[c] - mu) * rs * inlnw[c] + inlnb[c];
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (short i = 0; i < 8; i++) {
    const int o = i * 32 + lane;
    float a = b1[o];
    for (int k = 0; k < 132; k++) a += g[sg][k] * float(W1[k * 256 + o]);
    h[sg][o] = gelu(a);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  float v[8];
  for (short i = 0; i < 8; i++) {
    const int o = i * 32 + lane;
    float a = b2[o] + tok[b * 256 + o] + pos[n * 256 + o];
    for (int k = 0; k < 256; k++) a += h[sg][k] * float(W2[k * 256 + o]);
    v[i] = a;
    X[(b * 400 + n) * 256 + o] = a;
  }
  ln_vals<256>(v, lnw, lnb, p.eps2, lane);
  for (short i = 0; i < 8; i++) Y[(b * 400 + n) * 256 + i * 32 + lane] = half(v[i]);
}

// multi-head attention with 32-wide heads. Four threads per query (a quad of lanes), each holding 8 of the 32 dims of
// q and of the output: partial q.k products are summed across the quad with two shuffles. Keys / values are staged
// in tiles of 64 (fp32) through threadgroup memory; scores 8 keys at a time (unrolled, in registers) with an
// online-softmax rescale per 8. 64 queries per threadgroup (256 threads). Q rows b * NQ + i (head h at column h * 32), K / V rows
// b * NK + j; out rows b * NQ + i. grid (ceil(NQ / 64), heads, B)
struct A32 {
  int NQ, NK, ldq, ldk, ldv, ldo;
  float scale;  // 1 / sqrt(32) * log2(e)
};
kernel void attn32(const device half* Q [[buffer(0)]], const device half* K [[buffer(1)]], const device half* V [[buffer(2)]],
                   device half* O [[buffer(3)]], constant A32& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                   ushort tid [[thread_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 Ks[64][8];
  threadgroup float4 Vs[64][8];
  const int h = tg.y, b = tg.z, part = tid & 3, i = tg.x * 64 + (tid >> 2);
  const int qi = min(i, p.NQ - 1);
  const device half4* qr = (const device half4*)(Q + (b * p.NQ + qi) * p.ldq + h * 32 + part * 8);
  const float4 q0 = float4(qr[0]) * p.scale, q1 = float4(qr[1]) * p.scale;
  float4 o0 = 0, o1 = 0;
  float m = -INFINITY, l = 0;
  for (int k0 = 0; k0 < p.NK; k0 += 64) {
    for (int e = tid; e < 64 * 8; e += 256) {
      const int r = e >> 3, c = e & 7, key = min(k0 + r, p.NK - 1);
      Ks[r][c] = float4(*(const device half4*)(K + (b * p.NK + key) * p.ldk + h * 32 + c * 4));
      Vs[r][c] = float4(*(const device half4*)(V + (b * p.NK + key) * p.ldv + h * 32 + c * 4));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int n = min(64, p.NK - k0);
    for (int j0 = 0; j0 < n; j0 += 8) {  // 8 keys at a time, fully unrolled: the scores stay in registers
      float sc[8], cm = -INFINITY;
#pragma unroll
      for (short k = 0; k < 8; k++) {
        const int j = j0 + k;
        float a = dot(q0, Ks[j][part * 2]) + dot(q1, Ks[j][part * 2 + 1]);
        a += simd_shuffle_xor(a, ushort(1));
        a += simd_shuffle_xor(a, ushort(2));
        sc[k] = j < n ? a : -INFINITY;
        cm = max(cm, sc[k]);
      }
      const float mn = max(m, cm), corr = exp2(m - mn);
      l *= corr;
      o0 *= corr;
      o1 *= corr;
#pragma unroll
      for (short k = 0; k < 8; k++) {
        const float w = exp2(sc[k] - mn);
        l += w;
        o0 += w * Vs[j0 + k][part * 2];
        o1 += w * Vs[j0 + k][part * 2 + 1];
      }
      m = mn;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (i >= p.NQ) return;
  device half4* out = (device half4*)(O + (b * p.NQ + i) * p.ldo + h * 32 + part * 8);
  const float inv = 1.0f / l;
  out[0] = half4(o0 * inv);
  out[1] = half4(o1 * inv);
}

// The same attention on the simdgroup matrix units: each simdgroup takes 8 queries of one head through key tiles of 32,
// staged once per threadgroup in threadgroup memory (the 4 simdgroups share them: device reads of K / V are the cost).
// S (8 x 32) = Q K^T as 4 x 4 8x8 products, the online softmax on S in threadgroup memory (4 lanes per row), then
// O (8 x 32) += P V as 4 x 4 products, rescaled per row in threadgroup memory. 4 simdgroups (32 queries) per
// threadgroup; grid (ceil(NQ / 32), heads, B).
kernel void attn32_mma(const device half* Q [[buffer(0)]], const device half* K [[buffer(1)]], const device half* V [[buffer(2)]],
                       device half* O [[buffer(3)]], constant A32& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]],
                       ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup half Kt[32 * 32];
  threadgroup half Vt[32 * 32];
  threadgroup float Sb[4][8 * 32];
  threadgroup half Pb[4][8 * 32];
  threadgroup float Ob[4][8 * 32];
  threadgroup half Qb[4][8 * 32];
  const int h = tg.y, b = tg.z, q0 = tg.x * 32 + sg * 8;
  // the 8 query rows (clamped at the end), scaled; O = 0
  for (int e = lane; e < 8 * 32; e += 32) {
    const int r = e / 32, d = e % 32, qi = min(q0 + r, p.NQ - 1);
    Qb[sg][e] = half(float(Q[(b * p.NQ + qi) * p.ldq + h * 32 + d]) * p.scale);
    Ob[sg][e] = 0;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 qf[4];
  for (short d = 0; d < 4; d++) simdgroup_load(qf[d], Qb[sg] + d * 8, 32);
  const short row = lane / 4, part = lane % 4;  // softmax: 4 lanes per row, 8 columns each
  float m = -INFINITY, l = 0;
  const device half* Kh = K + (b * p.NK) * p.ldk + h * 32;
  const device half* Vh = V + (b * p.NK) * p.ldv + h * 32;
  for (int k0 = 0; k0 < p.NK; k0 += 32) {
    // this tile's keys and values (rows past the end: the last row; their weights are zeroed below)
    for (int e = tid; e < 32 * 8; e += 128) {
      const int r = e / 8, c = (e % 8) * 4, key = min(k0 + r, p.NK - 1);
      *(threadgroup half4*)(Kt + r * 32 + c) = *(const device half4*)(Kh + key * p.ldk + c);
      *(threadgroup half4*)(Vt + r * 32 + c) = *(const device half4*)(Vh + key * p.ldv + c);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // S = Q K^T
    for (short j = 0; j < 4; j++) {
      simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
      for (short d = 0; d < 4; d++) {
        simdgroup_half8x8 kt;
        simdgroup_load(kt, Kt + (j * 8) * 32 + d * 8, 32, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(acc, qf[d], kt, acc);
      }
      simdgroup_store(acc, Sb[sg] + j * 8, 32);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // online softmax over this tile's 32 keys
    float s[8], tm = -INFINITY;
    for (short c = 0; c < 8; c++) {
      const int key = k0 + part * 8 + c;
      s[c] = key < p.NK ? Sb[sg][row * 32 + part * 8 + c] : -INFINITY;
      tm = max(tm, s[c]);
    }
    tm = max(tm, simd_shuffle_xor(tm, ushort(1)));
    tm = max(tm, simd_shuffle_xor(tm, ushort(2)));
    const float mn = max(m, tm), corr = exp2(m - mn);
    float ls = 0;
    for (short c = 0; c < 8; c++) {
      const float w = exp2(s[c] - mn);
      ls += w;
      Pb[sg][row * 32 + part * 8 + c] = half(w);
      Ob[sg][row * 32 + part * 8 + c] *= corr;
    }
    ls += simd_shuffle_xor(ls, ushort(1));
    ls += simd_shuffle_xor(ls, ushort(2));
    l = l * corr + ls;
    m = mn;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    // O += P V (rows past the end have zero weight)
    simdgroup_half8x8 pf[4];
    for (short kk = 0; kk < 4; kk++) simdgroup_load(pf[kk], Pb[sg] + kk * 8, 32);
    for (short j = 0; j < 4; j++) {
      simdgroup_float8x8 acc;
      simdgroup_load(acc, Ob[sg] + j * 8, 32);
      for (short kk = 0; kk < 4; kk++) {
        simdgroup_half8x8 vf;
        simdgroup_load(vf, Vt + (kk * 8) * 32 + j * 8, 32);
        simdgroup_multiply_accumulate(acc, pf[kk], vf, acc);
      }
      simdgroup_store(acc, Ob[sg] + j * 8, 32);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // the next tile overwrites Kt / Vt
  }
  // O / l (each row's sum lives in its 4 lanes)
  const float inv = 1.0f / l;
  const int qi = q0 + row;
  if (qi < p.NQ)
    for (short c = 0; c < 8; c++) O[(b * p.NQ + qi) * p.ldo + h * 32 + part * 8 + c] = half(Ob[sg][row * 32 + part * 8 + c] * inv);
}

// outputs: per point, the (already LayerNormed) last tokens Y -> K logits (calibration offset folded into the bias);
// per hand, [mean, max] of the residual X over the points -> LN -> 64 (GELU) -> K. One threadgroup (256) per image;
// a0 = K
kernel void contact_out(const device float* X [[buffer(0)]], const device half* Y [[buffer(1)]],
                        const device float* ow [[buffer(2)]], const device float* ob [[buffer(3)]],
                        const device float* hlnw [[buffer(4)]], const device float* hlnb [[buffer(5)]],
                        const device float* hw1 [[buffer(6)]], const device float* hb1 [[buffer(7)]],
                        const device float* hw2 [[buffer(8)]], const device float* hb2 [[buffer(9)]],
                        device float* out [[buffer(10)]], device float* out_hand [[buffer(11)]], constant CP& p [[buffer(12)]],
                        uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float mm[512];
  threadgroup float hid[64];
  threadgroup float red[2][8];
  const int b = tg.y, K = p.a0;
  for (int n = tid; n < 400; n += 256)
    for (int k = 0; k < K; k++) {
      float s = ob[k];
      for (int c = 0; c < 256; c++) s += float(Y[(b * 400 + n) * 256 + c]) * ow[c * K + k];
      out[(b * 400 + n) * K + k] = s;
    }
  // mean and max of each channel over the points
  float sum = 0, mx = -INFINITY;
  for (int n = 0; n < 400; n++) {
    const float x = X[(b * 400 + n) * 256 + tid];
    sum += x;
    mx = max(mx, x);
  }
  mm[tid] = sum / 400;
  mm[256 + tid] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // LayerNorm over the 512 values
  float a = mm[tid] + mm[256 + tid];
  a = simd_sum(a);
  if (lane == 0) red[0][sg] = a;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mu = 0;
  for (int j = 0; j < 8; j++) mu += red[0][j];
  mu /= 512;
  float d0 = mm[tid] - mu, d1 = mm[256 + tid] - mu;
  float v = simd_sum(d0 * d0 + d1 * d1);
  if (lane == 0) red[1][sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0;
  for (int j = 0; j < 8; j++) var += red[1][j];
  const float rs = rsqrt(var / 512 + p.eps);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mm[tid] = d0 * rs * hlnw[tid] + hlnb[tid];
  mm[256 + tid] = d1 * rs * hlnw[256 + tid] + hlnb[256 + tid];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 64) {
    float s = hb1[tid];
    for (int c = 0; c < 512; c++) s += mm[c] * hw1[c * 64 + tid];
    hid[tid] = gelu(s);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < K) {
    float s = hb2[tid];
    for (int c = 0; c < 64; c++) s += hid[c] * hw2[c * K + tid];
    out_hand[b * K + tid] = s;
  }
}
