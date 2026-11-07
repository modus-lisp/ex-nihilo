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
