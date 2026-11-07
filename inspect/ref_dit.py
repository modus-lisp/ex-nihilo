"""ref_dit.py -- Qwen-Image-2.1's DiT in numpy, float32, written from
docs/design.md (stable-diffusion.cpp's qwen_image_2_1.hpp): the reference
the Lisp/CUDA forward is held against.

    PYTHONPATH=/mnt/ollama/llama.cpp/gguf-py python3 -I ref_dit.py \
        DIT.gguf CONTEXT.f32 P LATENT.f32 H W SIGMA OUT_DIR

CONTEXT is P x 4096 floats (the encoder's states), LATENT 64 x H x W (the
noisy latent, channel-major), both raw little-endian float32.  Writes to
OUT_DIR: txt.f32 (txt_in's P x 4096), blk0.f32 (the joint sequence after
block 0), v.f32 (the velocity, 64 x H x W).
"""
import sys, os, math
import numpy as np
import gguf
from gguf import quants

PREFIX = "model.diffusion_model."
W, HEADS, HD, FFN, LAYERS = 4096, 32, 128, 12288, 32
AXES = (16, 56, 56)


class Weights:
    def __init__(self, path):
        self.r = gguf.GGUFReader(path)
        self.t = {x.name: x for x in self.r.tensors}

    def __call__(self, name):
        x = self.t[PREFIX + name]
        return np.ascontiguousarray(quants.dequantize(x.data, x.tensor_type), dtype=np.float32)


def linear(x, w):            # w: (out, in), as gguf-py gives it
    return x @ w.T


def layernorm(x, eps=1e-6):  # no affine
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps)


def rmsnorm(x, w, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def silu(x):
    return x / (1 + np.exp(-x))


def timestep_embedding(t, dim=256, period=10000.0):
    half = dim // 2
    f = np.exp(-math.log(period) * np.arange(half, dtype=np.float32) / half)
    a = t * f
    return np.concatenate([np.cos(a), np.sin(a)]).astype(np.float32)


def positions(p, h, w):
    """(P + H*W) x 3: text i at (i, i, i); the image after it at (P, row, col),
    centred: row - (H - H//2), col - (W - W//2)."""
    pos = [(i, i, i) for i in range(p)]
    for r in range(h):
        for c in range(w):
            pos.append((p, r - (h - h // 2), c - (w - w // 2)))
    return np.array(pos, dtype=np.float32)


def rope_tables(pos, theta=10000.0):
    """cos, sin: tokens x 64 (a pair each), the three axes' pairs in turn."""
    angs = []
    for a, d in enumerate(AXES):
        omega = 1.0 / theta ** (np.arange(0, d, 2, dtype=np.float64) / d)
        angs.append(pos[:, a:a + 1].astype(np.float64) * omega[None, :])
    ang = np.concatenate(angs, axis=1)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope(x, cos, sin):       # x: tokens x heads x 128, pairs (2i, 2i+1)
    x0, x1 = x[..., 0::2], x[..., 1::2]
    c, s = cos[:, None, :], sin[:, None, :]
    out = np.empty_like(x)
    out[..., 0::2] = x0 * c - x1 * s
    out[..., 1::2] = x0 * s + x1 * c
    return out


def attention(q, k, v, p):
    """Prompt queries see the prompt, causally; image queries see everything."""
    n = q.shape[0]
    out = np.empty_like(q)
    scale = 1.0 / math.sqrt(HD)
    for h in range(HEADS):
        s = (q[:, h, :] @ k[:, h, :].T) * scale
        mask = np.zeros((n, n), dtype=bool)
        mask[:p, :p] = np.triu(np.ones((p, p), dtype=bool), 1)
        mask[:p, p:] = True
        s = np.where(mask, -np.inf, s)
        s = s - s.max(-1, keepdims=True)
        e = np.exp(s)
        out[:, h, :] = (e / e.sum(-1, keepdims=True)) @ v[:, h, :]
    return out


def forward(wt, ctx, latent, sigma, dump=None):
    p = ctx.shape[0]
    c, h, w = latent.shape
    # timestep: t and 0 through the embedder, then SiLU
    t = sigma * 1000.0
    emb = np.stack([timestep_embedding(t), timestep_embedding(0.0)])
    emb = linear(silu(linear(emb, wt("time_text_embed.timestep_embedder.linear_1.weight"))),
                 wt("time_text_embed.timestep_embedder.linear_2.weight"))
    emb = silu(emb)
    mod = linear(emb, wt("modulation.1.weight"))            # 2 x 16384
    scale1, gate1, scale2, gate2 = (mod[:, i * W:(i + 1) * W] for i in range(4))
    # the prompt through txt_in, the image through img_in
    txt = rmsnorm(ctx, 1.0 + wt("txt_in.text_norm.weight"))
    txt = linear(gelu_tanh(linear(txt, wt("txt_in.in_layer.weight"))), wt("txt_in.out_layer.weight"))
    img = linear(latent.reshape(c, h * w).T, wt("img_in.weight"))
    if dump: txt.astype(np.float32).tofile(os.path.join(dump, "txt.f32"))
    x = np.concatenate([txt, img]).astype(np.float32)
    n = x.shape[0]
    row = np.where(np.arange(n) < p, 1, 0)                  # prompt: modulation row 1 (t = 0)
    cos, sin = rope_tables(positions(p, h, w))

    def mod_of(m):
        return m[row]

    for il in range(LAYERS):
        b = "transformer_blocks.%d." % il
        y = layernorm(x) * (1.0 + mod_of(scale1))
        q = linear(y, wt(b + "attn.to_q.weight")).reshape(n, HEADS, HD)
        k = linear(y, wt(b + "attn.to_k.weight")).reshape(n, HEADS, HD)
        v = linear(y, wt(b + "attn.to_v.weight")).reshape(n, HEADS, HD)
        q = rope(rmsnorm(q, wt(b + "attn.norm_q.weight")), cos, sin)
        k = rope(rmsnorm(k, wt(b + "attn.norm_k.weight")), cos, sin)
        a = attention(q, k, v, p).reshape(n, W)
        x = x + np.tanh(mod_of(gate1)) * linear(a, wt(b + "attn.to_out.0.weight"))
        y = layernorm(x) * (1.0 + mod_of(scale2))
        gu = linear(y, wt(b + "img_mlp.gate_up.weight"))
        y = silu(gu[:, :FFN]) * gu[:, FFN:]
        x = x + np.tanh(mod_of(gate2)) * linear(y, wt(b + "img_mlp.out.weight"))
        x = x.astype(np.float32)
        if dump and il == 0: x.tofile(os.path.join(dump, "blk0.f32"))
        print("block %d: |x| %.4g" % (il, float(np.sqrt((x * x).mean()))), flush=True)
    x = x[p:]
    sc = linear(emb[0:1], wt("norm_out.linear.weight"))
    x = layernorm(x) * (1.0 + sc)
    v = linear(x, wt("proj_out.weight"))                    # (H*W) x 64
    return v.T.reshape(c, h, w).astype(np.float32)


if __name__ == "__main__":
    dit, ctxf, p, latf, h, w, sigma, out = sys.argv[1:9]
    p, h, w, sigma = int(p), int(h), int(w), float(sigma)
    ctx = np.fromfile(ctxf, dtype=np.float32).reshape(p, W)
    lat = np.fromfile(latf, dtype=np.float32).reshape(64, h, w)
    os.makedirs(out, exist_ok=True)
    v = forward(Weights(dit), ctx, lat, sigma, dump=out)
    v.tofile(os.path.join(out, "v.f32"))
    print("v: mean %.5g, rms %.5g" % (float(v.mean()), float(np.sqrt((v * v).mean()))))
