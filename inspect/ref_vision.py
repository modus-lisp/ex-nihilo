"""ref_vision.py -- Qwen3-VL-8B's vision tower in numpy, float32, written from
transformers' modeling_qwen3_vl.py (Qwen3VLVisionModel) and vision_utils.py:
the reference the Lisp/CUDA tower (src/cuda-vision.lisp) is held against.

    python3 -I ref_vision.py MMPROJ.gguf PIXELS.f32 H W OUT_DIR [GGUF_PY_DIR]

PIXELS is H x W x 3 floats, RGB already normalized ((x/255 - 0.5) / 0.5),
raw little-endian float32, H and W multiples of 32.  Writes to OUT_DIR:
embed.f32 (the patches embedded, positions added: N x 1152, merge-block
order), blk0.f32 (after block 0), merged.f32 (N/4 x 4096: the tokens the
language model reads), ds0.f32 ds1.f32 ds2.f32 (deepstack, N/4 x 4096 each).
"""
import sys, os, math
import numpy as np

sys.path.insert(0, sys.argv[6] if len(sys.argv) > 6 else "/mnt/ollama/llama.cpp/gguf-py")
import gguf
from gguf import quants

D, HEADS, FF, DEPTH, PATCH, MERGE, OUT = 1152, 16, 4304, 27, 16, 2, 4096
HD = D // HEADS
DEEPSTACK = (8, 16, 24)
THETA = 10000.0


class Weights:
    def __init__(self, path):
        self.r = gguf.GGUFReader(path)
        self.t = {x.name: x for x in self.r.tensors}

    def __call__(self, name):
        x = self.t[name]
        a = quants.dequantize(x.data, x.tensor_type)
        return np.ascontiguousarray(a, dtype=np.float32)


def layernorm(x, w, b, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    from math import erf
    return 0.5 * x * (1 + np.vectorize(erf)(x / math.sqrt(2))).astype(np.float32)


def merge_order(gh, gw):
    """(row, col) of each patch in merge-block order: block row, block col, in-row, in-col."""
    rows, cols = [], []
    for br in range(gh // MERGE):
        for bc in range(gw // MERGE):
            for ir in range(MERGE):
                for ic in range(MERGE):
                    rows.append(br * MERGE + ir)
                    cols.append(bc * MERGE + ic)
    return np.array(rows), np.array(cols)


def interp_taps(index, size, side):
    """Bilinear, align_corners=True: 2 taps and weights per index (vision_utils)."""
    src = index.astype(np.float64) * (side - 1) / max(size - 1, 1)
    fl = np.floor(src)
    taps = np.stack([fl, fl + 1], 1).astype(np.int64)
    w = np.clip(1 - np.abs(src[:, None] - taps), 0, None)
    return np.clip(taps, 0, side - 1), w


def main():
    mm, pix_path, H, W, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    os.makedirs(out, exist_ok=True)
    w = Weights(mm)
    pix = np.fromfile(pix_path, dtype=np.float32).reshape(H, W, 3)
    gh, gw = H // PATCH, W // PATCH
    rows, cols = merge_order(gh, gw)
    n = gh * gw

    # patch embedding: Conv3d over two identical frames == the two kernels' sum as one Conv2d
    pw = w("v.patch_embd.weight") + w("v.patch_embd.weight.1")       # (1152, 3, 16, 16)
    patches = np.stack([pix[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH, :].transpose(2, 0, 1).reshape(-1)
                        for r, c in zip(rows, cols)])                  # (n, 3*16*16) in (c, ky, kx) order
    x = patches @ pw.reshape(D, -1).T + w("v.patch_embd.bias")

    # learned positions: a 48 x 48 table resampled bilinearly to the grid
    pos = w("v.position_embd.weight").reshape(-1, D)                   # (2304, 1152)
    side = int(round(math.sqrt(pos.shape[0])))
    ht, hw_ = interp_taps(rows, gh, side)
    wt, ww = interp_taps(cols, gw, side)
    pe = np.zeros((n, D), np.float32)
    for a in range(2):
        for b in range(2):
            pe += (hw_[:, a] * ww[:, b])[:, None].astype(np.float32) * pos[ht[:, a] * side + wt[:, b]]
    x = (x + pe).astype(np.float32)
    x.tofile(os.path.join(out, "embed.f32"))

    # 2-D rotary: 18 frequencies over HD/2 = 36 dims; h's then w's, the 36 twice
    inv = 1.0 / (THETA ** (np.arange(0, HD // 2, 2, dtype=np.float32) / (HD // 2)))
    fh, fw = rows[:, None] * inv, cols[:, None] * inv
    f = np.concatenate([fh, fw], 1)
    f = np.concatenate([f, f], 1).astype(np.float32)                    # (n, 72)
    cos, sin = np.cos(f), np.sin(f)

    def rope(t):                                                        # t: (n, heads, 72)
        half = HD // 2
        rot = np.concatenate([-t[..., half:], t[..., :half]], -1)
        return t * cos[:, None, :] + rot * sin[:, None, :]

    def merger(x, prefix, post):
        if post:
            y = layernorm(x.reshape(-1, 4 * D), w(prefix + ".norm.weight"), w(prefix + ".norm.bias"))
            fc1, b1, fc2, b2 = prefix + ".fc1.weight", prefix + ".fc1.bias", prefix + ".fc2.weight", prefix + ".fc2.bias"
        else:
            y = layernorm(x, w("v.post_ln.weight"), w("v.post_ln.bias")).reshape(-1, 4 * D)
            fc1, b1, fc2, b2 = "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias"
        y = gelu_erf(y @ w(fc1).T + w(b1))
        return (y @ w(fc2).T + w(b2)).astype(np.float32)

    ds = []
    for i in range(DEPTH):
        p = f"v.blk.{i}."
        h = layernorm(x, w(p + "ln1.weight"), w(p + "ln1.bias"))
        qkv = (h @ w(p + "attn_qkv.weight").T + w(p + "attn_qkv.bias")).reshape(n, 3, HEADS, HD)
        q, k, v = rope(qkv[:, 0]), rope(qkv[:, 1]), qkv[:, 2]
        att = np.empty((n, HEADS, HD), np.float32)
        for hh in range(HEADS):
            s = (q[:, hh] @ k[:, hh].T) / math.sqrt(HD)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            att[:, hh] = s @ v[:, hh]
        x = x + att.reshape(n, D) @ w(p + "attn_out.weight").T + w(p + "attn_out.bias")
        h = layernorm(x, w(p + "ln2.weight"), w(p + "ln2.bias"))
        x = (x + gelu_tanh(h @ w(p + "ffn_up.weight").T + w(p + "ffn_up.bias")) @ w(p + "ffn_down.weight").T
             + w(p + "ffn_down.bias")).astype(np.float32)
        if i == 0:
            x.tofile(os.path.join(out, "blk0.f32"))
        if i in DEEPSTACK:
            ds.append(merger(x, f"v.deepstack.{i}", True))
        print(f"block {i}", file=sys.stderr, flush=True)
    merger(x, None, False).tofile(os.path.join(out, "merged.f32"))
    for j, d in enumerate(ds):
        d.tofile(os.path.join(out, f"ds{j}.f32"))


if __name__ == "__main__":
    main()
