// exn.metal — Qwen-Image-2.1's DiT and decoder on Apple silicon: Metal
// ports of src/cuda/dit.cu and src/cuda/vae.cu, compiled after illatio's
// kernels (whose d4_mm_q8_0 and d4_mm_f16 take the DiT's products).  A
// sequence or a picture is rows of floats; the attention's Q, K and V are
// halves (its products run on the half simdgroup matrices, summed in floats).
//
// DiT (see dit.cu for the shapes):
//   x_ln_mod, x_gate_add, x_swiglu2, x_gelu_tanh, x_silu  as dit.cu
//   x_rms(width, eps, x, w, y)           Y = RMSNorm(X) * W, a row a threadgroup
//   x_qk_norm_rope_h(n, w, cs, x, xh)    X's 32 heads of 128: RMSNorm * W, RoPE, as halves into XH
//   x_to_half(count, x, xh)              XH = X, as halves
//   x_attend_h(args, q, k, v, out)       dit.cu's x_attend, its mask, halves in
// decoder (see vae.cu):
//   x_conv(args, wt, bias, x, y)         a 3 x 3 convolution (stride 1 or 2) or a 1 x 1 (a product)
//   x_up2, x_dupup_add, x_avgdown_add, x_split3, x_transpose, x_softmax_rows, x_clamp1  as vae.cu
//   x_bias(count, cout, y, bias, res)    Y += BIAS (per channel) [+ RES]
//   x_add(count, x, y)                   X += Y

#include <metal_stdlib>
using namespace metal;

// a threadgroup's sum (or max), any size up to 1024, in every thread: each
// simdgroup's partial into RED (32 floats), then every simdgroup sums them
// all -- each lane reads the partial of its own index
static inline float x_tg_sum(float v, threadgroup float * red, ushort tid, ushort ntg) {
    v = simd_sum(v);
    if ((tid & 31) == 0) red[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ushort lane = tid & 31;
    float s = (lane < (ntg + 31) / 32) ? red[lane] : 0.f;
    s = simd_sum(s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

static inline float x_tg_max(float v, threadgroup float * red, ushort tid, ushort ntg) {
    v = simd_max(v);
    if ((tid & 31) == 0) red[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ushort lane = tid & 31;
    float s = (lane < (ntg + 31) / 32) ? red[lane] : -INFINITY;
    s = simd_max(s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// --- the DiT --------------------------------------------------------------------

struct x_lnmod_args { uint width; uint n; uint n0; uint p; uint mstride; };

// OUT = LayerNorm(X) * (1 + MOD[row]); a row below P (absolute: N0 + t) takes MOD's second row
kernel void x_ln_mod(constant x_lnmod_args & a [[buffer(0)]], device const float * x [[buffer(1)]],
                     device const float * mod [[buffer(2)]], device float * out [[buffer(3)]],
                     uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort ntg [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    if (t >= a.n) return;
    device const float * r = x + (ulong)t * a.width;
    device float * o = out + (ulong)t * a.width;
    device const float * m = mod + (a.n0 + t < a.p ? a.mstride : 0);
    float s = 0.f;
    for (uint i = tid; i < a.width; i += ntg) s += r[i];
    const float mean = x_tg_sum(s, red, tid, ntg) / (float)a.width;
    float ss = 0.f;
    for (uint i = tid; i < a.width; i += ntg) { const float d = r[i] - mean; ss += d * d; }
    const float inv = rsqrt(x_tg_sum(ss, red, tid, ntg) / (float)a.width + 1e-6f);
    for (uint i = tid; i < a.width; i += ntg) o[i] = (r[i] - mean) * inv * (1.f + m[i]);
}

struct x_gate_args { uint count; uint width; uint n0; uint p; uint mstride; };

kernel void x_gate_add(constant x_gate_args & a [[buffer(0)]], device float * x [[buffer(1)]],
                       device const float * h [[buffer(2)]], device const float * gate [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= a.count) return;
    const uint t = a.n0 + i / a.width, c = i % a.width;
    x[i] += precise::tanh(gate[(t < a.p ? a.mstride : 0) + c]) * h[i];
}

struct x_rms_args { uint width; float eps; };

kernel void x_rms(constant x_rms_args & a [[buffer(0)]], device const float * x [[buffer(1)]],
                  device const float * w [[buffer(2)]], device float * y [[buffer(3)]],
                  uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                  ushort ntg [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    device const float * r = x + (ulong)t * a.width;
    device float * o = y + (ulong)t * a.width;
    float ss = 0.f;
    for (uint i = tid; i < a.width; i += ntg) ss += r[i] * r[i];
    const float inv = rsqrt(x_tg_sum(ss, red, tid, ntg) / (float)a.width + a.eps);
    for (uint i = tid; i < a.width; i += ntg) o[i] = r[i] * inv * w[i];
}

struct x_qk_args { uint n; };

// a head's 128 values a simdgroup (lane l: pairs l and l + 32... as dit.cu's
// thread j: the pair (2j, 2j+1)); grid (n, 32), threads 64
kernel void x_qk_norm_rope_h(constant x_qk_args & a [[buffer(0)]], device const float * w [[buffer(1)]],
                             device const float * cs [[buffer(2)]], device const float * x [[buffer(3)]],
                             device half * xh [[buffer(4)]],
                             uint2 tg [[threadgroup_position_in_grid]], ushort j [[thread_index_in_threadgroup]]) {
    threadgroup float red[32];
    const uint t = tg.x, h = tg.y;
    if (t >= a.n) return;
    device const float * r = x + (ulong)t * 4096 + h * 128;
    const float u = r[2 * j], v = r[2 * j + 1];
    const float inv = rsqrt(x_tg_sum(u * u + v * v, red, j, 64) / 128.f + 1e-6f);
    const float un = u * inv * w[2 * j], vn = v * inv * w[2 * j + 1];
    const float c = cs[(ulong)t * 128 + 2 * j], s = cs[(ulong)t * 128 + 2 * j + 1];
    device half * o = xh + (ulong)t * 4096 + h * 128;
    o[2 * j] = (half)(un * c - vn * s);
    o[2 * j + 1] = (half)(un * s + vn * c);
}

struct x_n_args { uint count; };

kernel void x_to_half(constant x_n_args & a [[buffer(0)]], device const float * x [[buffer(1)]],
                      device half * xh [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < a.count) xh[i] = (half)x[i];
}

struct x_swiglu_args { uint count; uint f; };

kernel void x_swiglu2(constant x_swiglu_args & a [[buffer(0)]], device const float * gu [[buffer(1)]],
                      device float * out [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.count) return;
    const ulong t = i / a.f, j = i % a.f;
    const float g = gu[t * 2 * a.f + j];
    out[i] = g / (1.f + precise::exp(-g)) * gu[t * 2 * a.f + a.f + j];
}

kernel void x_gelu_tanh(constant x_n_args & a [[buffer(0)]], device float * x [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= a.count) return;
    const float v = x[i];
    x[i] = 0.5f * v * (1.f + precise::tanh(0.7978845608028654f * (v + 0.044715f * v * v * v)));
}

kernel void x_silu(constant x_n_args & a [[buffer(0)]], device float * x [[buffer(1)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= a.count) return;
    const float v = x[i];
    x[i] = v / (1.f + precise::exp(-v));
}

// The DiT's mask (dit.cu's xa_lim): row G sees keys 0 .. X_LIM(G).
static inline uint x_lim(uint g, uint p, uint l, uint b0, uint b1) {
    return g >= p ? l - 1 : (g >= b0 && g < b1 ? b1 - 1 : g);
}

struct x_attn_args { uint n; uint q0; uint p; uint l; uint b0; uint b1; float scale; };

// Flash attention on the half simdgroup matrices.  A threadgroup takes 64
// query rows of one head, each of its eight simdgroups 8 of them; the keys
// go 32 a step through threadgroup memory (K and V as halves), shared by
// the eight (so K and V are read from memory once for 64 queries).  S = Q
// K^T and O += P V are half products summed in floats, P rounded to halves,
// the softmax kept running per row (max M, sum L): four lanes a row, eight
// keys each.  O's rescaling by a row's factor is a product with that
// diagonal, skipped when every factor is 1.  Q, K, V: rows of 4096 halves
// (32 heads of 128); OUT rows of 4096 floats.  grid (32, ceil(n/64)),
// threads 256, 32768 bytes shared: K and V 16 KB, 2 KB a simdgroup.
#define XA_D 128
#define XA_KB 32
#define XA_SG 8
kernel void x_attend_h(constant x_attn_args & a [[buffer(0)]],
                       device const half * q [[buffer(1)]], device const half * kc [[buffer(2)]],
                       device const half * vc [[buffer(3)]], device float * out [[buffer(4)]],
                       threadgroup char * shmem [[threadgroup(0)]],
                       uint2 tgp [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint h = tgp.x, tq0 = tgp.y * 8 * XA_SG, t0 = tq0 + sg * 8;
    threadgroup half * ks = (threadgroup half *)shmem;                  // 32 x 128
    threadgroup half * vs = ks + XA_KB * XA_D;                          // 32 x 128
    threadgroup char * mine = shmem + 2 * XA_KB * XA_D * 2 + sg * 2048; // this simdgroup's 2 KB
    threadgroup float * ss = (threadgroup float *)mine;                 // 8 x 32: S
    threadgroup float * dg = ss + 8 * XA_KB;                            // 8 x 8: a diagonal
    threadgroup half * ps = (threadgroup half *)(dg + 64);              // 8 x 32: P

    // Q's 8 rows as sixteen 8 x 8 blocks (a row past N repeats N-1's: computed, never stored)
    simdgroup_half8x8 qm[XA_D / 8];
    {
        threadgroup half * qs = (threadgroup half *)mine;              // 8 x 128 halves: 2 KB
        for (uint i = lane; i < 8 * XA_D; i += 32) {
            const uint rr = min(t0 + i / XA_D, a.n - 1);
            qs[i] = q[(ulong)rr * 4096 + h * XA_D + (i % XA_D)];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (short c = 0; c < XA_D / 8; ++c) simdgroup_load(qm[c], qs + c * 8, XA_D);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = lane; i < 64; i += 32) dg[i] = 0.f;

    simdgroup_float8x8 o[XA_D / 8];
    for (short c = 0; c < XA_D / 8; ++c) o[c] = make_filled_simdgroup_matrix<float, 8>(0.f);

    const uint r = lane / 4, part = lane % 4;                           // this lane's row, its 8 keys of a step
    const bool live = t0 + r < a.n;
    const uint lim = x_lim(a.q0 + min(t0 + r, a.n - 1), a.p, a.l, a.b0, a.b1);
    const uint last = x_lim(a.q0 + min(tq0 + 8 * XA_SG - 1, a.n - 1), a.p, a.l, a.b0, a.b1);
    float m = -INFINITY, lsum = 0.f;

    for (uint kb = 0; kb <= last; kb += XA_KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // the step's keys and values: 32 x 128 halves each, 8 halves at a time (two a thread)
        for (uint i = tid; i < XA_KB * XA_D / 8; i += 8 * 32 * XA_SG / 8) {
            const uint key = i / (XA_D / 8), c = (i % (XA_D / 8)) * 8, kp = kb + key;
            half4 k0 = half4(0.h), k1 = half4(0.h), v0 = half4(0.h), v1 = half4(0.h);
            if (kp < a.l) {
                device const half4 * kq = (device const half4 *)(kc + (ulong)kp * 4096 + h * XA_D + c);
                device const half4 * vq = (device const half4 *)(vc + (ulong)kp * 4096 + h * XA_D + c);
                k0 = kq[0]; k1 = kq[1]; v0 = vq[0]; v1 = vq[1];
            }
            *(threadgroup half4 *)(ks + key * XA_D + c) = k0;
            *(threadgroup half4 *)(ks + key * XA_D + c + 4) = k1;
            *(threadgroup half4 *)(vs + key * XA_D + c) = v0;
            *(threadgroup half4 *)(vs + key * XA_D + c + 4) = v1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q K^T, four 8 x 8 blocks of keys
        simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8>(0.f), s1 = s0, s2 = s0, s3 = s0;
        for (short c = 0; c < XA_D / 8; ++c) {
            simdgroup_half8x8 k0, k1, k2, k3;
            simdgroup_load(k0, ks + c * 8,              XA_D, 0, true);
            simdgroup_load(k1, ks + 8 * XA_D + c * 8,   XA_D, 0, true);
            simdgroup_load(k2, ks + 16 * XA_D + c * 8,  XA_D, 0, true);
            simdgroup_load(k3, ks + 24 * XA_D + c * 8,  XA_D, 0, true);
            simdgroup_multiply_accumulate(s0, qm[c], k0, s0);
            simdgroup_multiply_accumulate(s1, qm[c], k1, s1);
            simdgroup_multiply_accumulate(s2, qm[c], k2, s2);
            simdgroup_multiply_accumulate(s3, qm[c], k3, s3);
        }
        simdgroup_store(s0, ss,      XA_KB);
        simdgroup_store(s1, ss + 8,  XA_KB);
        simdgroup_store(s2, ss + 16, XA_KB);
        simdgroup_store(s3, ss + 24, XA_KB);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // the running softmax
        float v[8], mx = -INFINITY;
        for (short i = 0; i < 8; ++i) {
            const uint key = kb + part * 8 + i;
            v[i] = (live && key <= lim) ? ss[r * XA_KB + part * 8 + i] * a.scale : -INFINITY;
            mx = max(mx, v[i]);
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        const float mn = max(m, mx);
        float ls = 0.f;
        for (short i = 0; i < 8; ++i) {
            const float e = mn == -INFINITY ? 0.f : precise::exp(v[i] - mn);
            ps[r * XA_KB + part * 8 + i] = (half)e;
            ls += e;
        }
        ls += simd_shuffle_xor(ls, 1);
        ls += simd_shuffle_xor(ls, 2);
        const float f = mn == -INFINITY ? 1.f : precise::exp(m - mn);
        lsum = lsum * f + ls;
        m = mn;
        if (part == 0) dg[r * 8 + r] = f;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(f) O + P V; the rescaling (float products, slow) only when a row's maximum moved
        if (simd_any(f != 1.f)) {
            simdgroup_float8x8 dm;
            simdgroup_load(dm, dg, 8);
            for (short c = 0; c < XA_D / 8; ++c) simdgroup_multiply(o[c], dm, o[c]);
        }
        for (short kk = 0; kk < XA_KB / 8; ++kk) {
            simdgroup_half8x8 pm;
            simdgroup_load(pm, ps + kk * 8, XA_KB);
            for (short c = 0; c < XA_D / 8; ++c) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, vs + kk * 8 * XA_D + c * 8, XA_D);
                simdgroup_multiply_accumulate(o[c], pm, vm, o[c]);
            }
        }
    }

    // O / L: straight to memory when all 8 rows exist, else 64 columns at a time through this simdgroup's 2 KB
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0) dg[r * 8 + r] = lsum > 0.f ? 1.f / lsum : 0.f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dm;
    simdgroup_load(dm, dg, 8);
    for (short c = 0; c < XA_D / 8; ++c) simdgroup_multiply(o[c], dm, o[c]);
    if (t0 + 8 <= a.n) {
        for (short c = 0; c < XA_D / 8; ++c) simdgroup_store(o[c], out + (ulong)t0 * 4096 + h * XA_D + c * 8, 4096);
    } else if (t0 < a.n) {
        threadgroup float * stage = (threadgroup float *)mine;          // 8 x 64 floats
        for (short half_ = 0; half_ < 2; ++half_) {
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (short c = 0; c < 8; ++c) simdgroup_store(o[half_ * 8 + c], stage + c * 8, 64);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = lane; i < 8 * 64; i += 32) {
                const uint rr = t0 + i / 64;
                if (rr < a.n) out[(ulong)rr * 4096 + h * XA_D + half_ * 64 + (i % 64)] = stage[i];
            }
        }
    }
}

// --- the decoder ------------------------------------------------------------------

struct x_conv_args { uint cin; uint cout; uint h; uint w; uint taps; uint stride; uint ldx; uint ldy; uint has_bias; };

// Y (pixels x COUT) = convolution of X (H x W pixels x CIN, rows LDX apart)
// + BIAS.  TAPS 9: 3 x 3, zero-padded (STRIDE 1), or stride 2 over the
// picture padded right and below only (the output H/2 x W/2); TAPS 1: a 1 x
// 1, a product (H x W rows).  WT: COUT rows of (tap, ci).  A threadgroup 64
// pixels by 64 channels, a thread 4 x 4 of them, K 16 at a time.
// grid (ceil(pixels/64), ceil(cout/64)), threads 256.
kernel void x_conv(constant x_conv_args & a [[buffer(0)]], device const float * wt [[buffer(1)]],
                   device const float * bias [[buffer(2)]], device const float * x [[buffer(3)]],
                   device float * y [[buffer(4)]],
                   uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float xs[16][68], ws[16][68];
    const uint ow = a.stride == 2 ? a.w / 2 : a.w, oh = a.stride == 2 ? a.h / 2 : a.h;
    const uint n = oh * ow, ncols = a.taps * a.cin;
    const uint t0 = tg.x * 64, r0 = tg.y * 64;
    const ushort tx = tid & 15, ty = tid >> 4, lr = tid >> 2, lk = (tid & 3) * 4;
    const uint pix = t0 + lr, py = pix / ow, px = pix - py * ow;
    float acc[4][4];
    for (short i = 0; i < 4; ++i) for (short j = 0; j < 4; ++j) acc[i][j] = 0.f;
    for (uint k0 = 0; k0 < ncols; k0 += 16) {
        // four consecutive K values from k0 + lk, each its own tap and channel
        for (short e = 0; e < 4; ++e) {
            const uint k = k0 + lk + e;
            float xv = 0.f, wv = 0.f;
            if (k < ncols) {
                const uint tap = k / a.cin, ci = k - tap * a.cin;
                int sy, sx;
                if (a.taps == 1) { sy = py; sx = px; }
                else if (a.stride == 2) { sy = 2 * py + tap / 3; sx = 2 * px + tap % 3; }
                else { sy = (int)py + (int)(tap / 3) - 1; sx = (int)px + (int)(tap % 3) - 1; }
                if (pix < n && sy >= 0 && sy < (int)a.h && sx >= 0 && sx < (int)a.w)
                    xv = x[((ulong)sy * a.w + sx) * a.ldx + ci];
                if (r0 + lr < a.cout) wv = wt[(ulong)(r0 + lr) * ncols + k];
            }
            xs[lk + e][lr] = xv;
            ws[lk + e][lr] = wv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short k = 0; k < 16; ++k) {
            const float4 av = *(threadgroup const float4 *)&xs[k][4 * ty];
            const float4 bv = *(threadgroup const float4 *)&ws[k][4 * tx];
            for (short i = 0; i < 4; ++i)
                for (short j = 0; j < 4; ++j) acc[i][j] = fma(av[i], bv[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (short i = 0; i < 4; ++i) {
        const uint t = t0 + 4 * ty + i;
        if (t >= n) continue;
        for (short j = 0; j < 4; ++j) {
            const uint r = r0 + 4 * tx + j;
            if (r < a.cout) y[(ulong)t * a.ldy + r] = acc[i][j] + (a.has_bias ? bias[r] : 0.f);
        }
    }
}

struct x_bias_args { uint count; uint cout; uint has_res; };

kernel void x_bias(constant x_bias_args & a [[buffer(0)]], device float * y [[buffer(1)]],
                   device const float * bias [[buffer(2)]], device const float * res [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= a.count) return;
    y[i] += bias[i % a.cout] + (a.has_res ? res[i] : 0.f);
}

kernel void x_add(constant x_n_args & a [[buffer(0)]], device float * x [[buffer(1)]],
                  device const float * y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < a.count) x[i] += y[i];
}

struct x_resample_args { uint cin; uint cout; uint ft; uint h; uint w; };

// vae.cu's x_avgdown_add: Y (H/2 x W/2 x COUT) += AvgDown3D(X) (H x W x CIN)
kernel void x_avgdown_add(constant x_resample_args & a [[buffer(0)]], device const float * x [[buffer(1)]],
                          device float * y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    const uint ow = a.w / 2;
    if (i >= (a.h / 2) * ow * a.cout) return;
    const uint o = i % a.cout, p = i / a.cout, yy = p / ow, xx = p % ow;
    const uint g = a.cin * a.ft * 4 / a.cout;
    float s = 0.f;
    for (uint k = 0; k < g; ++k) {
        const uint j = o * g + k, f = (j >> 2) % a.ft;
        if (f != a.ft - 1) continue;
        const uint c = j / (4 * a.ft), sy = 2 * yy + ((j >> 1) & 1), sx = 2 * xx + (j & 1);
        s += x[((ulong)sy * a.w + sx) * a.cin + c];
    }
    y[i] += s / g;
}

struct x_up_args { uint c; uint h; uint w; };

kernel void x_up2(constant x_up_args & a [[buffer(0)]], device const float * x [[buffer(1)]],
                  device float * y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= 4 * a.h * a.w * a.c) return;
    const uint ch = i % a.c, p = i / a.c, yy = p / (2 * a.w), xx = p % (2 * a.w);
    y[i] = x[((ulong)(yy >> 1) * a.w + (xx >> 1)) * a.c + ch];
}

// vae.cu's x_dupup_add: Y (2H x 2W x COUT) += DupUp3D(X) (H x W x CIN)
kernel void x_dupup_add(constant x_resample_args & a [[buffer(0)]], device const float * x [[buffer(1)]],
                        device float * y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= 4 * a.h * a.w * a.cout) return;
    const uint c = i % a.cout, p = i / a.cout, yy = p / (2 * a.w), xx = p % (2 * a.w);
    const uint rep = a.cout * a.ft * 4 / a.cin;
    const uint j = ((c * a.ft + a.ft - 1) * 2 + (yy & 1)) * 2 + (xx & 1);
    y[i] += x[((ulong)(yy >> 1) * a.w + (xx >> 1)) * a.cin + j / rep];
}

struct x_split_args { uint count; uint c; };

kernel void x_split3(constant x_split_args & a [[buffer(0)]], device const float * qkv [[buffer(1)]],
                     device float * q [[buffer(2)]], device float * k [[buffer(3)]], device float * v [[buffer(4)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= a.count) return;
    const ulong t = i / a.c, j = i % a.c;
    device const float * r = qkv + t * 3 * a.c;
    q[i] = r[j]; k[i] = r[a.c + j]; v[i] = r[2 * a.c + j];
}

struct x_tr_args { uint rows; uint cols; };

// Y (COLS x ROWS) = X^T; grid (ceil(cols/32), ceil(rows/32)), threads (32, 8)
kernel void x_transpose(constant x_tr_args & a [[buffer(0)]], device const float * x [[buffer(1)]],
                        device float * y [[buffer(2)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint2 lt [[thread_position_in_threadgroup]]) {
    threadgroup float tile[32][33];
    const uint c0 = tg.x * 32, r0 = tg.y * 32;
    for (uint j = lt.y; j < 32; j += 8) {
        const uint r = r0 + j, c = c0 + lt.x;
        if (r < a.rows && c < a.cols) tile[j][lt.x] = x[(ulong)r * a.cols + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = lt.y; j < 32; j += 8) {
        const uint c = c0 + j, r = r0 + lt.x;
        if (r < a.rows && c < a.cols) y[(ulong)c * a.rows + r] = tile[lt.x][j];
    }
}

struct x_smax_args { uint n; uint cols; uint stride; float scale; };

// each row of S (STRIDE apart): softmax of its first COLS times SCALE, the rest zeroed; a row a threadgroup
kernel void x_softmax_rows(constant x_smax_args & a [[buffer(0)]], device float * s [[buffer(1)]],
                           uint t [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                           ushort ntg [[threads_per_threadgroup]]) {
    threadgroup float red[32];
    device float * r = s + (ulong)t * a.stride;
    for (uint i = a.cols + tid; i < a.stride; i += ntg) r[i] = 0.f;
    float mx = -INFINITY;
    for (uint i = tid; i < a.cols; i += ntg) mx = max(mx, r[i]);
    mx = x_tg_max(mx, red, tid, ntg);
    float sum = 0.f;
    for (uint i = tid; i < a.cols; i += ntg) { const float e = precise::exp((r[i] - mx) * a.scale); r[i] = e; sum += e; }
    const float inv = 1.f / x_tg_sum(sum, red, tid, ntg);
    for (uint i = tid; i < a.cols; i += ntg) r[i] *= inv;
}

kernel void x_clamp1(constant x_n_args & a [[buffer(0)]], device float * x [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < a.count) x[i] = min(1.f, max(-1.f, x[i]));
}

// --- copies ---------------------------------------------------------------------

// DST = SRC, COUNT 32-bit words (the prompt's cached keys and values, as halves)
kernel void x_copy(constant x_n_args & a [[buffer(0)]], device const uint * src [[buffer(1)]],
                   device uint * dst [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < a.count) dst[i] = src[i];
}

// x_conv on the half simdgroup matrices (an implicit product): a threadgroup
// 64 pixels by 64 output channels, its four simdgroups 32 x 32 each, K taken
// NK at a time -- within one tap, since CIN is a multiple of NK (and K, for
// a product, too) -- the inputs and weights gathered as halves into
// threadgroup memory, sums in floats.  Same arguments as x_conv.
// grid (ceil(pixels/64), ceil(cout/64)), threads 128, 16384 bytes shared.
template <short NK>
kernel void x_conv_h(constant x_conv_args & a [[buffer(0)]], device const float * wt [[buffer(1)]],
                     device const float * bias [[buffer(2)]], device const float * x [[buffer(3)]],
                     device float * y [[buffer(4)]], threadgroup char * shmem [[threadgroup(0)]],
                     uint2 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)shmem;          // 64 pixels x NK
    threadgroup half * sb = sa + 64 * NK;                       // 64 channels x NK
    threadgroup float * stage = (threadgroup float *)shmem;     // 64 x 64 at the end
    const uint ow = a.stride == 2 ? a.w / 2 : a.w, oh = a.stride == 2 ? a.h / 2 : a.h;
    const uint n = oh * ow, ncols = a.taps * a.cin;
    const uint t0 = tg.x * 64, r0 = tg.y * 64;
    constexpr short PER = NK / 2;                               // halves a thread loads of a row: 2 threads a row
    const ushort lr = tid / 2, lk = (tid % 2) * PER;
    const uint pix = t0 + lr, py = pix / ow, px = pix - py * ow;
    const ushort pb = (sg % 2) * 32, cb = (sg / 2) * 32;
    simdgroup_float8x8 mc[16];
    for (short i = 0; i < 16; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
    for (uint k0 = 0; k0 < ncols; k0 += NK) {
        const uint tap = k0 / a.cin, ci = k0 - tap * a.cin + lk;
        int sy, sx;
        if (a.taps == 1) { sy = py; sx = px; }
        else if (a.stride == 2) { sy = 2 * py + tap / 3; sx = 2 * px + tap % 3; }
        else { sy = (int)py + (int)(tap / 3) - 1; sx = (int)px + (int)(tap % 3) - 1; }
        const bool ok = pix < n && sy >= 0 && sy < (int)a.h && sx >= 0 && sx < (int)a.w;
        // (no address formed for a pixel of the padding or a channel past COUT)
        if (ok) {
            device const float4 * xp = (device const float4 *)(x + ((ulong)sy * a.w + sx) * a.ldx + ci);
            for (short j = 0; j < PER / 4; ++j) *(threadgroup half4 *)(sa + lr * NK + lk + 4 * j) = half4(xp[j]);
        } else {
            for (short j = 0; j < PER / 4; ++j) *(threadgroup half4 *)(sa + lr * NK + lk + 4 * j) = half4(0.h);
        }
        if (r0 + lr < a.cout) {
            device const float4 * wp = (device const float4 *)(wt + (ulong)(r0 + lr) * ncols + k0 + lk);
            for (short j = 0; j < PER / 4; ++j) *(threadgroup half4 *)(sb + lr * NK + lk + 4 * j) = half4(wp[j]);
        } else {
            for (short j = 0; j < PER / 4; ++j) *(threadgroup half4 *)(sb + lr * NK + lk + 4 * j) = half4(0.h);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short kk = 0; kk < NK / 8; ++kk) {
            simdgroup_half8x8 ma[4], mb[4];
            for (short i = 0; i < 4; ++i) simdgroup_load(ma[i], sa + (pb + 8 * i) * NK + kk * 8, NK);
            for (short j = 0; j < 4; ++j) simdgroup_load(mb[j], sb + (cb + 8 * j) * NK + kk * 8, NK, 0, true);
            for (short i = 0; i < 4; ++i)
                for (short j = 0; j < 4; ++j) simdgroup_multiply_accumulate(mc[4 * i + j], ma[i], mb[j], mc[4 * i + j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (short i = 0; i < 4; ++i)
        for (short j = 0; j < 4; ++j) simdgroup_store(mc[4 * i + j], stage + (pb + 8 * i) * 64 + cb + 8 * j, 64);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < 64 * 64; e += 128) {
        const uint t = t0 + e / 64, r = r0 + e % 64;
        if (t < n && r < a.cout) y[(ulong)t * a.ldy + r] = stage[e] + (a.has_bias ? bias[r] : 0.f);
    }
}

typedef decltype(x_conv_h<16>) x_conv_h_t;
template [[host_name("x_conv_h16")]] kernel x_conv_h_t x_conv_h<16>;
template [[host_name("x_conv_h32")]] kernel x_conv_h_t x_conv_h<32>;
