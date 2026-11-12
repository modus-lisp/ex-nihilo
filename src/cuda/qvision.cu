// qvision.cu — Qwen3-VL's vision tower (src/cuda-vision.lisp): what illatio's
// kernels and the VAE's don't already do.
//
//   q3v_layernorm(width, eps, x, w, b, y)      Y = LayerNorm(X) * W + B, a row a block (256)
//   q3v_heads(n, np, nh, hd, hdp, qkv, cs, q, k, vt)
//                                              QKV's rows (3 x NH x HD) apart into heads, Q and K
//                                              turned by the 2-D rotary (CS: a row's HD cosines,
//                                              then HD sines), each head's rows HDP long (zeros
//                                              past HD); V transposed, a head's HD rows NP long
//                                              (the caller zeroes the columns past N).  grid (n,
//                                              nh), block 128.

extern "C" __global__ void q3v_layernorm(int width, float eps, const float* __restrict__ x,
                                         const float* __restrict__ w, const float* __restrict__ b,
                                         float* __restrict__ y) {
  __shared__ float red[32];
  const float* r = x + (long long)blockIdx.x * width;
  float* o = y + (long long)blockIdx.x * width;
  float s = 0.f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) s += r[i];
  const float mean = d4_block_sum(s, red) / width;
  __syncthreads();
  float v = 0.f;
  for (int i = threadIdx.x; i < width; i += blockDim.x) { const float e = r[i] - mean; v += e * e; }
  const float inv = rsqrtf(d4_block_sum(v, red) / width + eps);
  for (int i = threadIdx.x; i < width; i += blockDim.x) o[i] = (r[i] - mean) * inv * w[i] + b[i];
}

extern "C" __global__ void q3v_heads(int n, int np, int nh, int hd, int hdp, const float* __restrict__ qkv,
                                     const float* __restrict__ cs, float* __restrict__ q, float* __restrict__ k,
                                     float* __restrict__ vt) {
  const int t = blockIdx.x, h = blockIdx.y, half = hd / 2, d = nh * hd;
  const float* row = qkv + (long long)t * 3 * d;
  const float* c = cs + (long long)t * 2 * hd;
  for (int j = threadIdx.x; j < hdp; j += blockDim.x) {
    float qv = 0.f, kv = 0.f;
    if (j < hd) {
      const float* qr = row + h * hd;
      const float* kr = row + d + h * hd;
      const float rq = j < half ? -qr[j + half] : qr[j - half];   // rotate_half
      const float rk = j < half ? -kr[j + half] : kr[j - half];
      qv = qr[j] * c[j] + rq * c[hd + j];
      kv = kr[j] * c[j] + rk * c[hd + j];
      vt[((long long)h * hd + j) * np + t] = row[2 * d + h * hd + j];
    }
    q[((long long)h * n + t) * hdp + j] = qv;
    k[((long long)h * n + t) * hdp + j] = kv;
  }
}
