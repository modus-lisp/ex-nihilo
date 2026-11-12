// vae.cu — Qwen-Image-2.1's decoder (Wan 2.2's, one frame) on an NVIDIA
// GPU.  A picture is H x W pixels, each a row of C floats (channels last),
// so a 1 x 1 convolution is a product (d41_mm_f32) and the RMS norm is
// v_rms over a row.
//
//   x_conv3(cin, cout, h, w, wt, bias, x, y)   Y = 3 x 3 convolution of X, zero-padded, + BIAS
//                                              WT: COUT rows of (ky, kx, ci); CIN a multiple of 16
//   x_up2(c, h, w, x, y)                       Y (2H x 2W) = X, each pixel four times
//   x_dupup_add(cin, cout, ft, h, w, x, y)     Y (2H x 2W x COUT) += DupUp3D(X) (H x W x CIN), frame FT-1 of FT
//   x_conv3s2(cin, cout, h, w, wt, bias, x, y) Y (H/2 x W/2) = 3 x 3 convolution of X, stride 2, X padded
//                                              by one pixel right and below only (the encoder's downsample)
//   x_avgdown_add(cin, cout, ft, h, w, x, y)   Y (H/2 x W/2 x COUT) += AvgDown3D(X) (H x W x CIN), one frame
//   x_split3(n, c, qkv, q, k, v)               QKV's rows (3C) apart into three N x C
//   x_transpose(rows, cols, x, y)              Y (COLS x ROWS) = X^T
//   x_softmax_rows(n, cols, scale, s)          each row of S: softmax(S * SCALE)
//   x_clamp1(count, x)                         X into -1 .. 1

// The product's tiles (illatio's d41_mm_f32): a block 64 pixels x 64
// output channels, a thread 4 x 4 of them, K = 9 CIN taken 16 at a time --
// within one tap, since CIN is a multiple of 16.  grid (ceil(h*w/64),
// ceil(cout/64)), block 256.
extern "C" __global__ void __launch_bounds__(256)
x_conv3(int cin, int cout, int h, int w, const float* __restrict__ wt, const float* __restrict__ bias,
        const float* __restrict__ x, float* __restrict__ y) {
  __shared__ float xs[16][64 + 4], ws[16][64 + 4];
  const int n = h * w, ncols = 9 * cin;
  const int t0 = blockIdx.x * 64, r0 = blockIdx.y * 64, tid = threadIdx.x;
  const int tx = tid & 15, ty = tid >> 4;
  const int lr = tid >> 2, lk = (tid & 3) * 4;
  const int pix = t0 + lr, py = pix / w, px = pix - py * w;
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
  for (int k0 = 0; k0 < ncols; k0 += 16) {
    const int tap = k0 / cin, ci = k0 - tap * cin + lk;
    const int sy = py + tap / 3 - 1, sx = px + tap % 3 - 1;
    float4 xv = make_float4(0.f, 0.f, 0.f, 0.f), wv = make_float4(0.f, 0.f, 0.f, 0.f);
    if (pix < n && sy >= 0 && sy < h && sx >= 0 && sx < w)
      xv = *(const float4*)(x + ((long long)sy * w + sx) * cin + ci);
    if (r0 + lr < cout) wv = __ldg((const float4*)(wt + (long long)(r0 + lr) * ncols + k0 + lk));
    xs[lk][lr] = xv.x; xs[lk + 1][lr] = xv.y; xs[lk + 2][lr] = xv.z; xs[lk + 3][lr] = xv.w;
    ws[lk][lr] = wv.x; ws[lk + 1][lr] = wv.y; ws[lk + 2][lr] = wv.z; ws[lk + 3][lr] = wv.w;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const float4 a = *(const float4*)&xs[k][4 * ty], b = *(const float4*)&ws[k][4 * tx];
      const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int t = t0 + 4 * ty + i;
    if (t >= n) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int r = r0 + 4 * tx + j;
      if (r < cout) y[(long long)t * cout + r] = acc[i][j] + bias[r];
    }
  }
}

// x_conv3's tiles; H and W the input's (even), the output H/2 x W/2.
extern "C" __global__ void __launch_bounds__(256)
x_conv3s2(int cin, int cout, int h, int w, const float* __restrict__ wt, const float* __restrict__ bias,
          const float* __restrict__ x, float* __restrict__ y) {
  __shared__ float xs[16][64 + 4], ws[16][64 + 4];
  const int ow = w / 2, n = (h / 2) * ow, ncols = 9 * cin;
  const int t0 = blockIdx.x * 64, r0 = blockIdx.y * 64, tid = threadIdx.x;
  const int tx = tid & 15, ty = tid >> 4;
  const int lr = tid >> 2, lk = (tid & 3) * 4;
  const int pix = t0 + lr, py = pix / ow, px = pix - py * ow;
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
  for (int k0 = 0; k0 < ncols; k0 += 16) {
    const int tap = k0 / cin, ci = k0 - tap * cin + lk;
    const int sy = 2 * py + tap / 3, sx = 2 * px + tap % 3;
    float4 xv = make_float4(0.f, 0.f, 0.f, 0.f), wv = make_float4(0.f, 0.f, 0.f, 0.f);
    if (pix < n && sy < h && sx < w)
      xv = *(const float4*)(x + ((long long)sy * w + sx) * cin + ci);
    if (r0 + lr < cout) wv = __ldg((const float4*)(wt + (long long)(r0 + lr) * ncols + k0 + lk));
    xs[lk][lr] = xv.x; xs[lk + 1][lr] = xv.y; xs[lk + 2][lr] = xv.z; xs[lk + 3][lr] = xv.w;
    ws[lk][lr] = wv.x; ws[lk + 1][lr] = wv.y; ws[lk + 2][lr] = wv.z; ws[lk + 3][lr] = wv.w;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const float4 a = *(const float4*)&xs[k][4 * ty], b = *(const float4*)&ws[k][4 * tx];
      const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int t = t0 + 4 * ty + i;
    if (t >= n) continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int r = r0 + 4 * tx + j;
      if (r < cout) y[(long long)t * cout + r] = acc[i][j] + bias[r];
    }
  }
}

// AvgDown3D on one frame, padded in front to FT frames with zeros: output
// channel O of pixel (Y, X) is the mean of the G = CIN FT 4 / COUT values
// J = O G .. O G + G - 1 of the view (CIN, FT, 2, 2): channel J / (4 FT),
// frame (J / 4) % FT (the padding's zero but the last), row Y 2 + (J / 2) % 2,
// column X 2 + J % 2.
extern "C" __global__ void x_avgdown_add(int cin, int cout, int ft, int h, int w, const float* __restrict__ x,
                                         float* __restrict__ y) {
  const int ow = w / 2;
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long count = (long long)(h / 2) * ow * cout;
  if (i >= count) return;
  const int o = (int)(i % cout);
  const long long p = i / cout;
  const int yy = (int)(p / ow), xx = (int)(p % ow);
  const int g = cin * ft * 4 / cout;
  float s = 0.f;
  for (int k = 0; k < g; ++k) {
    const int j = o * g + k, f = (j >> 2) % ft;
    if (f != ft - 1) continue;
    const int c = j / (4 * ft), sy = 2 * yy + ((j >> 1) & 1), sx = 2 * xx + (j & 1);
    s += x[((long long)sy * w + sx) * cin + c];
  }
  y[i] += s / g;
}

extern "C" __global__ void x_up2(int c, int h, int w, const float* __restrict__ x, float* __restrict__ y) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long count = 4LL * h * w * c;
  if (i >= count) return;
  const int ch = (int)(i % c);
  const long long p = i / c;
  const int yy = (int)(p / (2 * w)), xx = (int)(p % (2 * w));
  y[i] = x[((long long)(yy >> 1) * w + (xx >> 1)) * c + ch];
}

// Output channel C of pixel (Y, X) takes input channel J / REP of pixel
// (Y/2, X/2), J = ((C FT + FT - 1) 2 + Y%2) 2 + X%2, REP = COUT FT 4 / CIN:
// repeat_interleave, the view as (COUT, FT, 2, 2), the last frame kept.
extern "C" __global__ void x_dupup_add(int cin, int cout, int ft, int h, int w, const float* __restrict__ x,
                                       float* __restrict__ y) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const long long count = 4LL * h * w * cout;
  if (i >= count) return;
  const int c = (int)(i % cout);
  const long long p = i / cout;
  const int yy = (int)(p / (2 * w)), xx = (int)(p % (2 * w));
  const int rep = cout * ft * 4 / cin;
  const int j = ((c * ft + ft - 1) * 2 + (yy & 1)) * 2 + (xx & 1);
  y[i] += x[((long long)(yy >> 1) * w + (xx >> 1)) * cin + j / rep];
}

extern "C" __global__ void x_split3(long long count, int c, const float* __restrict__ qkv, float* __restrict__ q,
                                    float* __restrict__ k, float* __restrict__ v) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const long long t = i / c, j = i % c;
  const float* r = qkv + t * 3 * c;
  q[i] = r[j]; k[i] = r[c + j]; v[i] = r[2 * c + j];
}

// grid (ceil(cols/32), ceil(rows/32)), block (32, 8)
extern "C" __global__ void x_transpose(int rows, int cols, const float* __restrict__ x, float* __restrict__ y) {
  __shared__ float tile[32][33];
  const int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
  for (int j = threadIdx.y; j < 32; j += 8) {
    const int r = r0 + j, c = c0 + threadIdx.x;
    if (r < rows && c < cols) tile[j][threadIdx.x] = x[(long long)r * cols + c];
  }
  __syncthreads();
  for (int j = threadIdx.y; j < 32; j += 8) {
    const int c = c0 + j, r = r0 + threadIdx.x;
    if (r < rows && c < cols) y[(long long)c * rows + r] = tile[threadIdx.x][j];
  }
}

// grid n rows, block 256
extern "C" __global__ void x_softmax_rows(int n, int cols, float scale, float* s) {
  __shared__ float red[32];
  float* r = s + (long long)blockIdx.x * cols;
  float mx = -INFINITY;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) mx = fmaxf(mx, r[i]);
  // a block's maximum: d4_block_sum's shape, with fmaxf
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    float m = threadIdx.x < (blockDim.x >> 5) ? red[threadIdx.x] : -INFINITY;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (threadIdx.x == 0) red[0] = m;
  }
  __syncthreads();
  mx = red[0];
  __syncthreads();
  float sum = 0.f;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) { const float e = expf((r[i] - mx) * scale); r[i] = e; sum += e; }
  const float inv = 1.f / d4_block_sum(sum, red);
  for (int i = threadIdx.x; i < cols; i += blockDim.x) r[i] *= inv;
}

extern "C" __global__ void x_clamp1(long long count, float* x) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) x[i] = fminf(1.f, fmaxf(-1.f, x[i]));
}
