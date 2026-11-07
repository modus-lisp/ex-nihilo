"""ref_vae.py -- Qwen-Image-2.1's VAE decoder in numpy, float32, for one
frame, written from diffusers' autoencoder_kl_qwenimage21.py: the reference
the CUDA decoder is held against.

    python3 -I ref_vae.py VAE.safetensors LATENT.f32 H W OUT.f32

LATENT is 64 x H x W floats, already denormalized (x std + mean);
OUT is 4 x 16H x 16W (RGBA, -1..1, clamped).
"""
import sys, json, struct
import numpy as np

STAGES = [(1152, 1152, True, True), (1152, 1152, True, True), (1152, 576, True, True),
          (576, 288, True, False), (288, 144, False, False)]


class ST:
    def __init__(self, path):
        f = open(path, "rb")
        n = struct.unpack("<Q", f.read(8))[0]
        self.h = json.loads(f.read(n))
        self.base = 8 + n
        self.f = f

    def __call__(self, name):
        e = self.h[name]
        a, b = e["data_offsets"]
        self.f.seek(self.base + a)
        raw = np.frombuffer(self.f.read(b - a), dtype=np.uint16)
        assert e["dtype"] == "BF16"
        return (raw.astype(np.uint32) << 16).view(np.float32).reshape(e["shape"])


def conv(x, w, b, k):
    """x: C x H x W; w: O x C x [1 x] k x k; 'same' padding, stride 1."""
    w = w.reshape(w.shape[0], w.shape[1], k, k)
    c, h, wd = x.shape
    if k == 1:
        return (w[:, :, 0, 0] @ x.reshape(c, -1)).reshape(-1, h, wd) + b[:, None, None]
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros((w.shape[0], h * wd), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += w[:, :, dy, dx] @ xp[:, dy:dy + h, dx:dx + wd].reshape(c, -1)
    return out.reshape(-1, h, wd) + b[:, None, None]


def rms(x, gamma):
    n = np.sqrt((x * x).sum(0, keepdims=True))
    return x / np.maximum(n, 1e-12) * np.sqrt(x.shape[0]) * gamma.reshape(-1, 1, 1)


def silu(x):
    return x / (1 + np.exp(-x))


def resblock(st, p, x):
    i = x.shape[0]
    o = st(p + ".residual.2.weight").shape[0]
    h = conv(x, st(p + ".shortcut.weight"), st(p + ".shortcut.bias"), 1) if i != o else x
    y = conv(silu(rms(x, st(p + ".residual.0.gamma"))), st(p + ".residual.2.weight"), st(p + ".residual.2.bias"), 3)
    y = conv(silu(rms(y, st(p + ".residual.3.gamma"))), st(p + ".residual.6.weight"), st(p + ".residual.6.bias"), 3)
    return y + h


def attention(st, p, x):
    c, h, w = x.shape
    y = rms(x, st(p + ".norm.gamma"))
    qkv = conv(y, st(p + ".to_qkv.weight"), st(p + ".to_qkv.bias"), 1).reshape(3 * c, h * w).T  # (hw, 3c)
    q, k, v = qkv[:, :c], qkv[:, c:2 * c], qkv[:, 2 * c:]
    s = q @ k.T / np.sqrt(c)
    s = np.exp(s - s.max(-1, keepdims=True))
    a = (s / s.sum(-1, keepdims=True)) @ v                                         # (hw, c)
    return conv(a.T.reshape(c, h, w), st(p + ".proj.weight"), st(p + ".proj.bias"), 1) + x


def dupup(x, out_c, ft, fs=2):
    """DupUp3D for the first (only) frame: repeat_interleave, the channels
    spread over (ft, fs, fs), frame ft - 1 kept."""
    c, h, w = x.shape
    rep = out_c * ft * fs * fs // c
    x = np.repeat(x, rep, axis=0).reshape(out_c, ft, fs, fs, h, w)[:, ft - 1]   # out_c, fs, fs, h, w
    return x.transpose(0, 3, 1, 4, 2).reshape(out_c, h * fs, w * fs)


def decode(st, z):
    x = conv(z, st("conv2.weight"), st("conv2.bias"), 1)
    x = conv(x, st("decoder.conv1.weight"), st("decoder.conv1.bias"), 3)
    x = resblock(st, "decoder.middle.0", x)
    x = attention(st, "decoder.middle.1", x)
    x = resblock(st, "decoder.middle.2", x)
    for s, (i, o, up, tup) in enumerate(STAGES):
        x0 = x
        for j in range(3):
            x = resblock(st, "decoder.upsamples.%d.upsamples.%d" % (s, j), x)
        if up:
            x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
            p = "decoder.upsamples.%d.upsamples.3.resample.1" % s
            x = conv(x, st(p + ".weight"), st(p + ".bias"), 3)
            x = x + dupup(x0, o, 2 if tup else 1)
        print("stage %d: %s, rms %.4g" % (s, x.shape, float(np.sqrt((x * x).mean()))), flush=True)
    x = silu(rms(x, st("decoder.head.0.gamma")))
    x = conv(x, st("decoder.head.2.weight"), st("decoder.head.2.bias"), 3)
    return np.clip(x, -1, 1).astype(np.float32)


if __name__ == "__main__":
    vae, latf, h, w, out = sys.argv[1:6]
    h, w = int(h), int(w)
    z = np.fromfile(latf, dtype=np.float32).reshape(64, h, w)
    decode(ST(vae), z).tofile(out)
