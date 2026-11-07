# Qwen-Image-2.1-Turbo: the design, and how we build it

These notes come from stable-diffusion.cpp's implementation, read at
`5a1ba5d` (2026-11-07):

- `src/model/diffusion/qwen_image_2_1.hpp`, the transformer
- `src/conditioning/conditioner.hpp`, the prompt
- `src/model/te/llm.hpp`, the text encoder
- `src/model/vae/wan_vae.hpp`, the decoder
- `src/runtime/denoiser.hpp`, the schedule

The configs are the HF repos' own. Details still to confirm against a
reference are marked **(check)**.

## One picture, end to end

```
prompt ──► template ──► Qwen3-VL-8B (text only, causal) ──► hidden states of the last block
                                                            (system turn dropped)
                                                                   │  P tokens × 4096
noise z (64 × H/16 × W/16) ──┐                                      ▼
                             └──► DiT, 8 steps; the prompt prefix is computed once and cached
                                        │  v = f(z, σ)
                                        ▼  z ← z + (σ' − σ)·v
                                   latent ──► × std + mean (per channel)
                                           ──► VAE decoder ──► RGBA in −1..1
```

At 1024×1024 the latent is 64×64, so there are 4,096 image tokens. A
typical prompt is 50–300 tokens.

## 1. The prompt

The template (the system turn is fixed):

```
<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n
<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
```

- **The text encoder** is Qwen3-VL-8B's text model.
  - Shape: 36 layers of width 4096, 32 query heads and 8 KV heads of 128,
    FFN 12288, QK-norm, RMSNorm eps 1e-6.
  - It is the qwen3 architecture illatio already runs.
  - RoPE: theta 5e6, with interleaved M-RoPE sections (24, 20, 20). For
    text, all three position axes are equal, so every frequency turns by
    the same position. That is plain 1-D RoPE.
- **The output** is the hidden state after block 36, before the final norm
  (sd.cpp's `out_layers = {num_layers}`; `num_layers + 1` would be the
  normed one).
- **Dropping the system turn.** The tokens of
  `<|im_start|>system\n…<|im_end|>\n` are cut off, which leaves P tokens.
  The rest of the template stays in, including the trailing
  `assistant\n`.
- **Attention is causal.** There is no padding: batch size 1, no mask
  beyond causality.
- **Editing** (deferred) would add Qwen3-VL's ViT for reference images. Its
  outputs replace `<|image_pad|>` runs, plus deepstack embeddings added
  after the early layers. Reference latents would also go to the DiT.

## 2. The DiT (7.0B, 32 blocks, width 4096)

There are no biases anywhere. The tensor names below are diffusers'
(297 tensors).

**Inputs, once per picture:**

- **`txt_in`** projects the prompt:
  1. a zero-centred RMSNorm, whose weight is used as 1 + w;
  2. a 4096→4096 linear;
  3. GELU (ggml's tanh form) **(check)**;
  4. a 4096→4096 linear.
- **`img_in`** is a 64→4096 linear per latent pixel (patch size 1).
- **Positions**, 3 axes, with head dimensions split 16/56/56, theta 10000:
  - The text token at index i is at (i, i, i).
  - The image starts at position p0 = P. A pixel (h, w) of an H×W latent
    is at (P, h − (H − ⌊H/2⌋), w − (W − ⌊W/2⌋)), so the image is centred
    on 0.
  - Rotation is by interleaved pairs (2i, 2i+1), FLUX style.

**Timestep, once per step:**

- Feed t = 1000·σ. Embed both t and 0 as 256 sinusoidal values (cos half,
  then sin; period 10000) **(check)**.
- Then `linear_1` → SiLU → `linear_2`, then SiLU.
- **`modulation.1`** (4096 → 4×4096) is shared by every block. Applied to
  [embed(t), embed(0)], it gives two rows per piece. Each of the four
  pieces is (scale1, gate1, scale2, gate2):
  - row 0 (from t) modulates the image tokens;
  - row 1 (from 0) modulates the prompt tokens.
- There are no shifts. A scale is applied as 1 + s, a gate as tanh(g).

**Block.** x is the joint sequence [prompt tokens | image tokens]:

```
h = LayerNorm(x)            (no affine, eps 1e-6)
h = h * (1 + scale1[row])
q, k, v = h·Wq, h·Wk, h·Wv; RMSNorm q and k per head (weights norm_q, norm_k); RoPE
attention (scale 1/√128):
    prompt queries: keys = prompt only, causal
    image queries:  keys = prompt + image, no mask
x = x + tanh(gate1[row]) * (attn · Wo)
h = LayerNorm(x) * (1 + scale2[row])
x = x + tanh(gate2[row]) * (W_out · (silu(h·W_gate) * (h·W_proj)))     (SwiGLU, 12288)
```

**Output.** Keep the image tokens only. Then:

1. LayerNorm, times (1 + `norm_out.linear`(embed(t))).
2. `proj_out` (4096 → 64).
3. Reshape to the 64×H×W velocity.

**The prefix cache, the key fact.** Prompt tokens never attend to image
tokens, and they are modulated by embed(0), not embed(t). So their keys and
values are the same at every step.

- **Step 1** runs the joint sequence and stores each layer's prompt K and V
  (32 × P × 4096 × 2; about 160 MB in f16 at P = 300).
- **Steps 2–8** run only the 4,096 image tokens. They attend over
  [cached prompt K/V | image K/V].

This is illatio's prompt reuse, nearly verbatim.

**Cost per step at 1024²:**

- Products: 2 × 7.0B × 4,096 ≈ 57 TFLOP.
- Attention: 4 × 32 × 4096 × 4096 × (4096 + P) ≈ 9 TFLOP.

That is about 66 TFLOP a step, and about 0.53 PFLOP for 8 steps.

## 3. The schedule

Turbo's 8 steps use a fixed sigma list, which sd.cpp names
`qwen-image-2.1-turbo`:

```
σ = 1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0
```

- **CFG is 1**, so there is one DiT pass per step and no negative prompt.
- **Initial noise:** z ~ N(0, 1), shape 64 × H/16 × W/16.
- **Update:** z ← z + (σ_{i+1} − σ_i) · v, a rectified-flow Euler step,
  where v is the DiT output. This assumes v = noise − x₀ **(check)**.
- **Before decoding**, denormalize per channel: z · std[c] + mean[c]. The
  64 means and 64 standard deviations are in the VAE config.
- **Resolution:** sides must be multiples of 32. The model was trained up
  to about 2,752 on a side. 1024² is the cheap case.
- **Not needed for Turbo:** the base model's dynamic-shift schedule
  (base_shift 0.5, max_shift 0.9 at 8,192 tokens, terminal 0.02). It
  matters only for the 40-step base model.

## 4. The VAE decoder (Wan 2.2 lineage, 2-D for a single frame)

- **Configuration:** `decoder_base_dim` 144, `dim_mult` [1, 2, 4, 8, 8],
  z_dim 64, 4 output channels (RGBA).
- **Channel widths, deepest first:** 1152, 1152, 1152, 576, 288, 144.

```
conv3×3 64→1152
middle: ResBlock(1152), Attention(1152: one head over all pixels), ResBlock(1152)
4 up stages, each: 3 × ResBlock(in→out), nearest ×2 + conv3×3, plus a shortcut
    the shortcut (DupUp3D) repeats channels and spreads them 2×2 into space
    1152→1152, 1152→1152, 1152→576, 576→288 (64 → 1024 pixels)
last stage: 3 × ResBlock(288→144), no upsampling
head: RMS_norm, SiLU, conv3×3 144→4
ResBlock: RMS_norm → SiLU → conv3×3 → RMS_norm → SiLU → conv3×3, + (1×1 conv shortcut if widths differ)
RMS_norm: per pixel, x / rms over channels × gamma (eps 1e-12)
```

- **Single frame.** For one frame, each causal 3-D convolution reduces to
  the last temporal slice of its kernel, so the decoder is a plain 2-D
  one.
- **Temporal upsampling** (first three stages) is skipped on frame 0: no
  `time_conv`.
- **The shortcut's temporal factor 2 still changes how channels are
  grouped** into the 2×2 spread. This is the subtlest part of the decoder;
  verify it against dumped tensors **(check)**.
- **sd.cpp's 1/128 conv scale** keeps FP16 in range. It is a numerical
  device and does not apply at FP32 **(check)**.
- **Cost:** about 2.3 TFLOP per resolution level, about 12 TFLOP in all at
  1024², roughly a fifth of one DiT step. The largest activation is
  144 × 1024² floats, 600 MB in FP32.
- **Output:** RGBA in −1..1, which maps to 0..255. Alpha is kept, since
  transparent output is a feature.

## What illatio already has, and what's new

| Piece | illatio | New |
|---|---|---|
| GGUF reader, Q8_0 / K-quant / BF16 weights | yes | |
| safetensors reader (the VAE ships only as BF16 safetensors) | | small: JSON header + raw data |
| Qwen tokenizer, chat template | yes | |
| Qwen3 text encoder (CPU, CUDA, Metal) | yes (Qwen3-4B) | return block 36's hidden state; 8B shape |
| Matmul, RMSNorm, SwiGLU, flash attention | yes | non-causal attention over [prefix + self]; mixed-mask first step |
| LayerNorm without affine, adaLN scale/gate, 3-axis interleaved RoPE, sinusoidal timestep | | small kernels |
| conv3×3 (implicit GEMM), nearest ×2, DupUp spread, spatial attention | | the decoder's kernels |
| Tensor-core GEMM (BF16 mma; later INT8 W8A8) | partly (prefill paths) | the main speed work: diffusion is compute-bound |
| Image output | pigment encodes PNG | RGBA → PNG |

**Recommendation: ex-nihilo depends on illatio as a library**, not a
fork. It would use illatio's GGUF reader, kernels, CUDA/Metal FFI,
tokenizer and qwen3 forward. If the coupling gets awkward, the shared
runtime can be split out later.

## Weights

| Piece | File | Size |
|---|---|---|
| DiT | `unsloth/Qwen-Image-2.1-Turbo-GGUF`, Q8_0 (F16 is 14.2 GB) | 7.6 GB |
| Text encoder | `Qwen/Qwen3-VL-8B-Instruct-GGUF`, Q8_0 (the mmproj is only needed for editing) | ~8.7 GB |
| VAE | `Comfy-Org/Qwen-Image-2.1`, `vae/qwen_image_2.1_vae_bf16.safetensors` | 0.7 GB |

About 17 GB in all.

- **Disk.** The root disk has 12 GB free and `/mnt/ollama` isn't writable,
  so the weights go in `/dev/shm` (92 GB free, RAM-backed, lost on
  reboot).
- **On the 3090:** DiT 7.6 GB + encoder 8.7 GB + activations and decoder
  (~2 GB) fits in 24 GB. Alternatively, encode first and free the encoder.
- **On the Mac:** everything at BF16 is 32 GB, which fits in 48 GB.

## How we verify

sd.cpp is the specification, not a test oracle; we don't build it. The
ladder is the one the V4.1 vision port used:

1. A **numpy reference** for each piece, written from the sources above:
   encoder, one DiT step, sampler, decoder. It runs on small shapes, for
   example a 256² image (16×16 latent, 256 image tokens).
2. **Lisp against numpy** on the same inputs and the same noise. Each
   piece must reach cosine 1.0 and a max difference of about 1e-5 at
   FP32.
3. **End to end**, by eye, on a few fixed prompts and seeds. A cheap
   automatic check: have illatio's V4.1 vision encoder describe the result
   ("a red apple on a wooden table", a sign that reads "HELLO 42").

Bit-exact agreement with sd.cpp images isn't a goal: its default noise
comes from a CUDA-style Philox generator. If we ever want it, we'd load
its noise.

## Milestones

1. **Weights in.** GGUF DiT and encoder; safetensors VAE. List the
   tensors against the names above.
2. **Prompt.** Qwen3-VL-8B text forward, returning block 36's hidden
   states. Compare to numpy.
3. **One DiT step on the CPU**, slow but exact: joint first step and
   prefix-cached step. Compare to numpy at 256².
4. **First picture on the CPU**: 8 steps, denormalize, decode, PNG,
   512². Minutes, but it proves the whole path.
5. **CUDA.**
   - Correctness on the P40 first, at FP32.
   - Then the 3090's tensor cores: BF16, then INT8. The V4.1 server on
     8082 is stopped during this.
   - Prefix cache on the GPU. Target: about 15 s at 1024² at BF16.
6. **Metal** for the Mac.
7. **Serve it**: OpenAI's `/v1/images/generations`, and a tool for
   operandi.

Later: editing (Qwen3-VL ViT, VAE encoder, reference latents), higher
resolutions, then video (Wan 2.2-5B shares this VAE family).
