# Image (and later video) generation: choosing a model

Research for a new repo, a sibling of illatio: a pure-Common-Lisp diffusion
engine. The targets are the two machines illatio was sized for, a MacBook Pro
with 48 GB and a single 24 GB NVIDIA card. As with illatio, the steps are:
pick a known-good model, understand its design against a C reference, then
plan the port.

Status as of 2026-11-06.

## Where the field is

Open image generation has settled on one recipe:

- **Text encoder.** A small LLM reads the prompt, and the model uses its
  hidden states rather than its logits. Every current model in the table
  below uses a Qwen3-family encoder.
- **Diffusion transformer (DiT).** It works on a compressed picture (a
  latent grid), runs 8–50 times (steps), and turns noise into a picture
  along a straight path (rectified flow / flow matching, sampled with
  Euler steps).
- **Convolutional decoder (VAE).** It turns the final latent back into
  pixels, once at the end.

Distilled "Turbo" checkpoints cut the step count to 8–16 and drop
classifier-free guidance, which otherwise doubles the work of every step.

Two other approaches exist but don't lead. Autoregressive image tokens
(GLM-Image, Janus) rank below the DiTs. Huge MoE models (HunyuanImage 3.0,
80B) don't fit these machines.

### Open-weights leaderboard

Artificial Analysis text-to-image arena, open-weights entries, fetched
2026-11-06. Architecture details come from the HF configs.

| Elo | Model | Released | DiT | Text encoder | VAE | Steps (distilled) | License |
|---|---|---|---|---|---|---|---|
| 1045 | **Qwen-Image-2.1 (+Turbo)** | Sep/Oct 2026 | 7B, 32 single-stream layers × 4096, causal condition | Qwen3-VL-8B | own; Wan lineage, 16×, 64 ch, RGBA | 8, CFG 1 | Qwen Research (non-commercial) |
| 1043 | FLUX 3 Image Klein | Oct 2026 | 5.6B | Qwen3-VL-4B (+ optional 9B prompt upsampler) | FLUX.2 (16×, 128 ch) | 16, CFG 1, "flow-map sampling" | FLUX Kommunity (gated; commercial output allowed under $5M revenue) |
| 1039 | Fibo 2.0 (Bria) | Oct 2026 | 8B | — | — | — | non-commercial |
| 1027 | Ideogram 4.0 | Jun 2026 | 9.3B | — | — | — | non-commercial |
| 978 | HiDream-O1-Image | May 2026 | 8B | — | — | — | MIT |
| 936 | Z-Image-Turbo | Nov 2025 | 6.15B, 30 single-stream layers × 3840 | Qwen3-4B | FLUX.1 (8×, 16 ch) | 8, CFG 0 | Apache 2.0 |
| 915 | ERNIE-Image Turbo | Apr 2026 | 8B, 36 single-stream layers × 4096 | Mistral-3 | FLUX.2 | 8 | Apache 2.0 |
| 840 | FLUX.2 [klein] 4B | Jan 2026 | 4B, 5 double + 20 single | Qwen3-4B | FLUX.2 | 4 | Apache 2.0 |

The top three are within the arena's noise of each other. Below them there
is a clear drop: 1045 to 936 is roughly a 65/35 preference split in
head-to-head comparisons.

### The reference implementation

stable-diffusion.cpp (leejet, MIT, active today) is the counterpart to what
ds4 and llama.cpp were for illatio.

- It covers every model in the table, plus Wan 2.1/2.2, LTX-2.3/2.5 and
  HunyuanVideo 1.5.
- It had day-0 support for Qwen-Image-2.1 and has FLUX 3 Klein.
- It runs on CUDA, Metal and Vulkan.
- It publishes GGUFs (leejet/*-GGUF; unsloth has its own GGUFs too).
- It works within 4 GB of VRAM by offloading.

illatio already reads GGUF and has Q8_0 and K-quant kernels on CUDA and
Metal, so the weights could load with the existing reader.

## Recommendation: Qwen-Image-2.1-Turbo

Why this model:

1. **Quality.** It is #1 among open weights, about 110 Elo above the
   Apache-licensed options.
2. **Fits both machines** without tricks:

   | Component | BF16 | Q8_0 |
   |---|---|---|
   | DiT | 14.2 GB | 7.6 GB |
   | Qwen3-VL-8B encoder | 17.5 GB | ~8.7 GB |
   | VAE | 0.7 GB | — |

   - On the Mac, the encoder and DiT both stay resident, at BF16 or Q8.
   - On the 24 GB card, quantize the encoder to Q8 or run it first and
     free it.
3. **Eight steps with CFG 1.** One DiT pass per step. The sigma schedule is
   a fixed 8-entry list that sd.cpp spells out.
4. **The encoder is mostly done.** Qwen3-VL's text path is the qwen3
   architecture illatio already runs. Interleaved M-RoPE collapses to
   ordinary RoPE when every position's three axes are equal, which is true
   for text-only prompts. Image editing would add Qwen3-VL's ViT, a second
   port of the same kind as V4.1's encoder.
5. **"causal_condition" plus prefix KV-cache reuse.** The prompt tokens'
   keys and values are computed once and reused across all 8 steps, an
   LLM-style idea that illatio's KV machinery already fits. This needs
   confirming against sd.cpp (design phase).
6. **Its VAE is Wan-lineage** (residual causal convolutions, the
   `AutoencoderKLWan` family). It is the same building block Wan 2.2's video
   VAE uses, so it is a stepping stone to video.

Costs:

- **The license is research / non-commercial.** That is fine for a personal
  engine. It would matter only if a product shipped the weights. The Lisp
  code is unaffected.
- **Native resolution is high.** The examples use 2048² and 1696×2528.
  1024² is 4,096 latent tokens; 2048² is 16,384, about 5× the work, because
  attention grows with the square of the token count.
- **Its VAE is new**: 5 levels, decoder base 144. Its exact layer list comes
  from sd.cpp or diffusers.

Fallback: **Z-Image-Turbo** (Apache 2.0).

- Its text encoder is Qwen3-4B, which illatio already runs as-is.
- Its VAE is FLUX.1's, the most documented conv decoder.
- Its DiT is similar in kind: single-stream, 3-axis RoPE, QK-norm, adaLN.
- It is a good first light if Qwen-Image's new pieces stall. It is also the
  pick if the license matters.

Not chosen:

- **FLUX 3 Klein.** Comparable quality. But it is gated, uses a new
  "flow-map" sampler, takes 16 steps (twice the work), and needs a 9B prompt
  upsampler: the card warns short prompts give generic output without it.
- **ERNIE-Image.** Lower Elo, and a Mistral-3 encoder that illatio doesn't
  have.

## Rough cost per image (1024², 8 steps)

FLOPs per step ≈ 2 × parameters × tokens, plus attention at
4 × layers × width × tokens². For Qwen-Image-2.1 at 1024² (about 4.4K
tokens including the prompt), that is about 62 + 10 ≈ 72 TFLOP a step, or
0.58 PFLOP an image. Everything else is small: the encoder runs once over
about 300 tokens, and the VAE decode is about 2–3 TFLOP.

| Machine | Peak | Assumed achieved | 1024² | 2048² |
|---|---|---|---|---|
| RTX 3090, BF16 tensor cores, FP32 accumulate | 71 TFLOPS | ~35 | ~17 s | ~90 s |
| RTX 3090, INT8 tensor cores (W8A8) | 284 TOPS | ~120 | ~5 s | ~25 s |
| M4 Pro, 20-core GPU | ~9 TFLOPS | ~6 | ~100 s | ~8 min |
| Tesla P40 (FP32, no tensor cores) | 12 TFLOPS | ~8 | ~75 s | — |

- **Compute bound.** Unlike LLM decode, diffusion steps are big matrix
  products, so the tensor cores matter and memory bandwidth mostly
  doesn't.
- **INT8.** GeForce cards run INT8 at 4× the rate of BF16 with FP32
  accumulation. That is why ComfyUI ships `int8_convrot` checkpoints for
  every one of these models, and it is the main performance lever on the
  3090.
- **The Mac** is about 6× slower than the 3090 at BF16. A minute and a half
  per 1024² image is usable, not fast. Smaller images scale roughly with
  pixel count.
- **On this box** the 3090 holds V4.1's experts, so generation would contend
  for it. The P40 is free but slow, and it is the right place for
  bring-up and correctness work.

## Video, for later

On the Artificial Analysis video arena, the only open-weights models are
LTX's and one MiniMax model (H3, 1160). Wan 2.2 and HunyuanVideo 1.5 are
older and unranked there, but widely run.

| Model | Size | Encoder | VAE | License | Notes |
|---|---|---|---|---|---|
| LTX-2.5 (Lightricks, Jul 2026) | 22B | Gemma-4-12B | video + audio VAEs, ×2 spatial/temporal upscalers | LTX community (free under $10M revenue) | Joint audio and video. Distilled to 8 steps. At 24 GB it needs INT8 plus offload. |
| Wan 2.2 TI2V-5B (Aug 2025) | 5B, 30 layers × 3072 | umT5-XXL | 16×16×4, 48 ch | Apache 2.0 | 720p, 24 fps, 5 s. About 27K tokens per clip × ~50 steps ≈ 27 PFLOP, roughly 10+ min on a 3090. |
| HunyuanVideo 1.5 (Nov 2025) | 8.3B, 54 layers | Qwen2.5-VL + ByT5 | 16×16×4, 32 ch | Tencent community (excludes EU/UK/KR) | |

Video is the same machinery (DiT, flow matching, a causal 3D-conv VAE) with
time added and about 10× the tokens. With a working image engine,
Wan 2.2-5B is the tractable first video port: Apache-licensed, and its VAE
family matches Qwen-Image-2.1's. LTX-2.5 is the quality target once there
is an INT8 path.

## What to understand before planning (design phase)

Read these from sd.cpp's `qwen_image` code and diffusers'
`QwenImage21Transformer2DModel` / `AutoencoderKLQwenImage21`:

1. **The text encoder.** Which hidden layer it reads, the chat template it
   wraps the prompt in, and whether it drops the template tokens afterwards.
2. **The DiT block.**
   - How the timestep modulates each block (adaLN).
   - QK-norm.
   - The 3-axis RoPE split (16/56/56 of 128).
   - What "causal condition" and "mixed-granularity attention" mean
     concretely.
   - Which keys and values are reusable across steps.
3. **The VAE decoder.** Its layer list: residual blocks, causal padding,
   upsampling, latent mean and std normalization. Also how RGBA output is
   produced.
4. **The Turbo sigma list** and the Euler update.
5. **The noise RNG.** Bit-exact comparison against sd.cpp needs the same
   starting noise, so we either load its noise or match its generator.

The verification ladder is the same one illatio used:

1. Each component against a numpy reference on dumped tensors.
2. One full step against sd.cpp.
3. The final image compared by eye and by PSNR.

## Sources

- [Artificial Analysis open-weights text-to-image leaderboard](https://artificialanalysis.ai/image/leaderboard/text-to-image/open-weights)
- [Artificial Analysis text-to-video leaderboard](https://artificialanalysis.ai/video/leaderboard/text-to-video?open-weights=true)
- [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), [Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)
- [black-forest-labs/flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein), [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (paper arXiv 2511.22699)
- [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image)
- [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5), [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers), [HunyuanVideo-1.5](https://huggingface.co/hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v)
- [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp), including its docs for qwen_image_2.1, z_image and flux3
- [vdf.ai: local image models, VRAM and Mac](https://vdf.ai/blog/best-local-image-generation-models/)
