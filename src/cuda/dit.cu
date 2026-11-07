// dit.cu — Qwen-Image-2.1's DiT steps on an NVIDIA GPU (docs/design.md),
// compiled with illatio's kernels (its products, v_rms, d4_block_sum).  A
// sequence is N rows of WIDTH floats; the prompt's P rows come first, and a
// row below P takes the modulation computed at t = 0 (row 1 of MOD), the
// image's rows the one at t (row 0).  MOD is 2 x MSTRIDE floats.
//
//   x_ln_mod(width, n, n0, p, mstride, x, mod, out) OUT = LayerNorm(X) * (1 + MOD[row]), X's rows N0..
//   x_gate_add(count, width, n0, p, mstride, x, h, gate)
//                                                   X += tanh(GATE[row]) * H
//   x_qk_norm_rope(n, w, cs, x)                     X's 32 heads of 128: RMSNorm * W, then RoPE
//   x_attend(q, k, v, out, nh, hd, n, q0, p, l, scale)
//                                                   prompt queries causal, image queries all L keys
//   x_swiglu2(count, f, gu, out)                    OUT = silu(GU[:, :F]) * GU[:, F:]
//   x_gelu_tanh(count, x)                           X = gelu(X), the tanh form
//   x_silu(count, x)                                X = silu(X)

extern "C" __global__ void x_ln_mod(int width, int n, int n0, int p, int mstride, const float* __restrict__ x,
                                    const float* __restrict__ mod, float* __restrict__ out) {
  __shared__ float red[32];
  const int t = blockIdx.x;
  if (t >= n) return;
  const float* r = x + (long long)t * width;
  float* o = out + (long long)t * width;
  const float* m = mod + (n0 + t < p ? mstride : 0);
  float s = 0.f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) s += r[i];
  const float mean = d4_block_sum(s, red) / (float)width;
  float ss = 0.f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) { const float d = r[i] - mean; ss += d * d; }
  const float inv = rsqrtf(d4_block_sum(ss, red) / (float)width + 1e-6f);
  for (int i = threadIdx.x; i < width; i += blockDim.x) o[i] = (r[i] - mean) * inv * (1.f + m[i]);
}

// rows N0.. of X (COUNT values from there): the row's absolute index decides its gate
extern "C" __global__ void x_gate_add(long long count, int width, int n0, int p, int mstride, float* __restrict__ x,
                                      const float* __restrict__ h, const float* __restrict__ gate) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int t = n0 + (int)(i / width), c = (int)(i % width);
  x[i] += tanhf(gate[(t < p ? mstride : 0) + c]) * h[i];
}

// A head's 128 values: RMSNorm (eps 1e-6) times W, then each pair (2j, 2j+1)
// turned by the token's angle j: CS holds cos, sin for 64 pairs a token.
// grid (n, 32), block 64.
extern "C" __global__ void x_qk_norm_rope(int n, const float* __restrict__ w, const float* __restrict__ cs,
                                          float* __restrict__ x) {
  __shared__ float red[32];
  const int t = blockIdx.x, h = blockIdx.y, j = threadIdx.x;
  if (t >= n) return;
  float* r = x + (long long)t * 4096 + h * 128;
  const float a = r[2 * j], b = r[2 * j + 1];
  const float inv = rsqrtf(d4_block_sum(a * a + b * b, red) / 128.f + 1e-6f);
  const float an = a * inv * w[2 * j], bn = b * inv * w[2 * j + 1];
  const float c = cs[(long long)t * 128 + 2 * j], s = cs[(long long)t * 128 + 2 * j + 1];
  r[2 * j] = an * c - bn * s;
  r[2 * j + 1] = an * s + bn * c;
}

// illatio's attn_prefill2 (dense.cu), its mask the DiT's: query t is row
// Q0 + t of the sequence; a row below P (the prompt) sees keys 0 .. itself,
// any other row all L keys.  K, V: L rows of NH x HD.  76 KB of shared
// memory at HD 128.  grid (nh, ceil(n/64)), block 256.
#define XA_Q 64
#define XA_K 32
extern "C" __global__ void __launch_bounds__(256)
x_attend(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
         float* __restrict__ out, int nh, int hd, int n, int q0, int p, int l, float scale) {
  extern __shared__ __align__(16) float fsm[];
  int hp = hd + 4;
  float* sq = fsm;                            // [64][hd+4]
  float* sk = sq + XA_Q * hp;                 // [32][hd+4]
  float* sv = sk + XA_K * hp;                 // [32][hd]
  float* sp = sv + XA_K * hd;                 // [32][68]
  int head = blockIdx.x, t0 = blockIdx.y * XA_Q;
  int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  int qd = nh * hd;
  int h4 = hd >> 2, nw = hd >> 6;
  for (int e = tid; e < XA_Q * h4; e += 256) {
    int tq = e / h4, c = e % h4, t = t0 + tq;
    float4 v = t < n ? __ldg((const float4*)(q + (long long)t * qd + head * hd) + c) : make_float4(0, 0, 0, 0);
    v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
    *(float4*)(sq + tq * hp + 4 * c) = v;
  }
  float o[4][16], m[4], lsum[4];
#pragma unroll
  for (int i = 0; i < 4; i++) { m[i] = -INFINITY; lsum[i] = 0.0f;
#pragma unroll
    for (int e = 0; e < 16; e++) o[i][e] = 0.0f; }
  const int tmax = q0 + min(n, t0 + XA_Q) - 1;
  const int last = tmax >= p ? l - 1 : tmax;
  for (int p0 = 0; p0 <= last; p0 += XA_K) {
    __syncthreads();
    {
      float4 rk[8], rv[8];
#pragma unroll
      for (int u = 0; u < 8; u++) {
        int e = tid + u * 256, kk = e / h4, c = e % h4, kp = p0 + kk;
        bool ok = kk < XA_K && kp <= last;
        long long off = (long long)kp * qd + head * hd;
        rk[u] = ok ? __ldg((const float4*)(kc + off) + c) : make_float4(0, 0, 0, 0);
        rv[u] = ok ? __ldg((const float4*)(vc + off) + c) : make_float4(0, 0, 0, 0);
      }
#pragma unroll
      for (int u = 0; u < 8; u++) {
        int e = tid + u * 256, kk = e / h4, c = e % h4;
        if (kk < XA_K) { *(float4*)(sk + kk * hp + 4 * c) = rk[u]; *(float4*)(sv + kk * hd + 4 * c) = rv[u]; }
      }
    }
    __syncthreads();
    float s[4][2];
#pragma unroll
    for (int i = 0; i < 4; i++) { s[i][0] = 0.0f; s[i][1] = 0.0f; }
    for (int j = 0; j < hd; j += 4) {
      float4 k0 = *(const float4*)(sk + tx * hp + j), k1 = *(const float4*)(sk + (tx + 16) * hp + j);
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float4 qv = *(const float4*)(sq + (4 * ty + i) * hp + j);
        s[i][0] += qv.x * k0.x + qv.y * k0.y + qv.z * k0.z + qv.w * k0.w;
        s[i][1] += qv.x * k1.x + qv.y * k1.y + qv.z * k1.z + qv.w * k1.w;
      }
    }
    float pw[4][2], f[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int g = q0 + t0 + 4 * ty + i;
      int lim = g < p ? g : l - 1;
      bool live = t0 + 4 * ty + i < n;
      float a = (live && p0 + tx <= lim) ? s[i][0] : -INFINITY;
      float b = (live && p0 + tx + 16 <= lim) ? s[i][1] : -INFINITY;
      float mx = fmaxf(a, b);
      for (int x = 8; x > 0; x >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, x));
      float mn = fmaxf(m[i], mx);
      float ea = mn == -INFINITY ? 0.0f : expf(a - mn), eb = mn == -INFINITY ? 0.0f : expf(b - mn);
      float ls = ea + eb;
      for (int x = 8; x > 0; x >>= 1) ls += __shfl_xor_sync(0xffffffff, ls, x);
      f[i] = mn == -INFINITY ? 1.0f : expf(m[i] - mn);
      lsum[i] = lsum[i] * f[i] + ls;
      m[i] = mn;
      pw[i][0] = ea; pw[i][1] = eb;
    }
    *(float4*)(sp + tx * 68 + 4 * ty) = make_float4(pw[0][0], pw[1][0], pw[2][0], pw[3][0]);
    *(float4*)(sp + (tx + 16) * 68 + 4 * ty) = make_float4(pw[0][1], pw[1][1], pw[2][1], pw[3][1]);
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
      for (int e = 0; e < 16; e++) o[i][e] *= f[i];
    __syncthreads();
    for (int kk = 0; kk < XA_K; kk++) {
      float4 pv = *(const float4*)(sp + kk * 68 + 4 * ty);
      float pa[4] = {pv.x, pv.y, pv.z, pv.w};
#pragma unroll
      for (int w = 0; w < 4; w++) {
        if (w < nw) {
          float4 v = *(const float4*)(sv + kk * hd + 4 * tx + 64 * w);
#pragma unroll
          for (int i = 0; i < 4; i++) {
            o[i][4 * w + 0] += pa[i] * v.x; o[i][4 * w + 1] += pa[i] * v.y;
            o[i][4 * w + 2] += pa[i] * v.z; o[i][4 * w + 3] += pa[i] * v.w;
          }
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int t = t0 + 4 * ty + i;
    if (t < n) {
      float inv = 1.0f / lsum[i];
#pragma unroll
      for (int w = 0; w < 4; w++)
        if (w < nw)
          *(float4*)(out + (long long)t * qd + head * hd + 4 * tx + 64 * w) =
            make_float4(o[i][4 * w] * inv, o[i][4 * w + 1] * inv, o[i][4 * w + 2] * inv, o[i][4 * w + 3] * inv);
    }
  }
}

extern "C" __global__ void x_swiglu2(long long count, int f, const float* __restrict__ gu, float* __restrict__ out) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const long long t = i / f, j = i % f;
  const float g = gu[t * 2 * f + j];
  out[i] = g / (1.f + expf(-g)) * gu[t * 2 * f + f + j];
}

extern "C" __global__ void x_gelu_tanh(long long count, float* x) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float v = x[i];
  x[i] = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

extern "C" __global__ void x_silu(long long count, float* x) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float v = x[i];
  x[i] = v / (1.f + expf(-v));
}

// x_attend's attention on the tensor cores (sm_80 and up): FlashAttention-2's
// shape.  A block takes 128 queries of one head, its 8 warps 16 each; the
// keys go 64 a step through shared memory as halves (K as rows, V
// transposed), Q held in registers as m16n8k16 A fragments.  S = Q K^T and
// O += P V are half products summed in floats; P is rounded to halves.
// Lane l of a warp holds rows l/4 and l/4 + 8 of its 16, columns 2(l%4)
// and the next of each 8-wide tile.  The mask is x_attend's.
// grid (32 heads, ceil(n/128)), block 256.
#define XM_Q 128
#define XM_K 64
#define XM_KS (128 + 8)                 // halves a K row in shared (+8: no bank conflicts)
#define XM_VS (XM_K + 8)                // halves a row of V^T

__device__ __forceinline__ unsigned xm_h2(float a, float b) { return (unsigned)f2h(a) | ((unsigned)f2h(b) << 16); }

__device__ __forceinline__ void xm_mma(float* c, const unsigned* a, unsigned b0, unsigned b1) {
#if __CUDA_ARCH__ >= 800
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

extern "C" __global__ void __launch_bounds__(256)
x_attend_mma(const float* __restrict__ q, const float* __restrict__ kc, const float* __restrict__ vc,
             float* __restrict__ out, int n, int q0, int p, int l, float scale) {
  __shared__ __align__(16) u16 sk[XM_K * XM_KS];
  __shared__ __align__(16) u16 sv[128 * XM_VS];
  const int head = blockIdx.x, t0 = blockIdx.y * XM_Q;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, tg = lane & 3;
  const int r0 = t0 + warp * 16 + g, r1 = r0 + 8;
  unsigned qa[8][4];
  {
    const bool ok0 = r0 < n, ok1 = r1 < n;
    const float* a = q + (long long)r0 * 4096 + head * 128;
    const float* b = q + (long long)r1 * 4096 + head * 128;
    const float2 z = make_float2(0.f, 0.f);
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      const int c = kk * 16 + 2 * tg;
      const float2 x0 = ok0 ? *(const float2*)(a + c) : z, x1 = ok1 ? *(const float2*)(b + c) : z;
      const float2 x2 = ok0 ? *(const float2*)(a + c + 8) : z, x3 = ok1 ? *(const float2*)(b + c + 8) : z;
      qa[kk][0] = xm_h2(x0.x * scale, x0.y * scale);
      qa[kk][1] = xm_h2(x1.x * scale, x1.y * scale);
      qa[kk][2] = xm_h2(x2.x * scale, x2.y * scale);
      qa[kk][3] = xm_h2(x3.x * scale, x3.y * scale);
    }
  }
  float o[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const int lim0 = q0 + r0 < p ? q0 + r0 : l - 1, lim1 = q0 + r1 < p ? q0 + r1 : l - 1;
  const int tmax = q0 + min(n, t0 + XM_Q) - 1;
  const int last = tmax >= p ? l - 1 : tmax;
  for (int k0 = 0; k0 <= last; k0 += XM_K) {
    __syncthreads();
    for (int e = tid; e < XM_K * 32; e += 256) {
      const int key = e >> 5, c = (e & 31) * 4, kp = k0 + key;
      const bool ok = kp <= last;
      const long long off = (long long)kp * 4096 + head * 128 + c;
      const float4 kv = ok ? __ldg((const float4*)(kc + off)) : make_float4(0.f, 0.f, 0.f, 0.f);
      const float4 vv = ok ? __ldg((const float4*)(vc + off)) : make_float4(0.f, 0.f, 0.f, 0.f);
      *(uint2*)(sk + key * XM_KS + c) = make_uint2(xm_h2(kv.x, kv.y), xm_h2(kv.z, kv.w));
      sv[(c + 0) * XM_VS + key] = f2h(vv.x);
      sv[(c + 1) * XM_VS + key] = f2h(vv.y);
      sv[(c + 2) * XM_VS + key] = f2h(vv.z);
      sv[(c + 3) * XM_VS + key] = f2h(vv.w);
    }
    __syncthreads();
    float s[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        const u16* kp = sk + (j * 8 + g) * XM_KS + kk * 16 + 2 * tg;
        xm_mma(s[j], qa[kk], *(const unsigned*)kp, *(const unsigned*)(kp + 8));
      }
    }
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int a = k0 + j * 8 + 2 * tg;
      if (a > lim0) s[j][0] = -INFINITY;
      if (a + 1 > lim0) s[j][1] = -INFINITY;
      if (a > lim1) s[j][2] = -INFINITY;
      if (a + 1 > lim1) s[j][3] = -INFINITY;
      mx0 = fmaxf(mx0, fmaxf(s[j][0], s[j][1]));
      mx1 = fmaxf(mx1, fmaxf(s[j][2], s[j][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1)); mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1)); mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    const float f0 = mx0 == -INFINITY ? 1.f : expf(m0 - mx0), f1 = mx1 == -INFINITY ? 1.f : expf(m1 - mx1);
    float ls0 = 0.f, ls1 = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      s[j][0] = mx0 == -INFINITY ? 0.f : expf(s[j][0] - mx0);
      s[j][1] = mx0 == -INFINITY ? 0.f : expf(s[j][1] - mx0);
      s[j][2] = mx1 == -INFINITY ? 0.f : expf(s[j][2] - mx1);
      s[j][3] = mx1 == -INFINITY ? 0.f : expf(s[j][3] - mx1);
      ls0 += s[j][0] + s[j][1];
      ls1 += s[j][2] + s[j][3];
    }
    ls0 += __shfl_xor_sync(0xffffffff, ls0, 1); ls0 += __shfl_xor_sync(0xffffffff, ls0, 2);
    ls1 += __shfl_xor_sync(0xffffffff, ls1, 1); ls1 += __shfl_xor_sync(0xffffffff, ls1, 2);
    l0 = l0 * f0 + ls0; l1 = l1 * f1 + ls1;
    m0 = mx0; m1 = mx1;
#pragma unroll
    for (int j = 0; j < 16; ++j) { o[j][0] *= f0; o[j][1] *= f0; o[j][2] *= f1; o[j][3] *= f1; }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      const unsigned pa[4] = {xm_h2(s[2 * kk][0], s[2 * kk][1]), xm_h2(s[2 * kk][2], s[2 * kk][3]),
                              xm_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]), xm_h2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const u16* vp = sv + (j * 8 + g) * XM_VS + kk * 16 + 2 * tg;
        xm_mma(o[j], pa, *(const unsigned*)vp, *(const unsigned*)(vp + 8));
      }
    }
  }
  const float i0 = 1.f / l0, i1 = 1.f / l1;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    const int c = head * 128 + j * 8 + 2 * tg;
    if (r0 < n) *(float2*)(out + (long long)r0 * 4096 + c) = make_float2(o[j][0] * i0, o[j][1] * i0);
    if (r1 < n) *(float2*)(out + (long long)r1 * 4096 + c) = make_float2(o[j][2] * i1, o[j][3] * i1);
  }
}
