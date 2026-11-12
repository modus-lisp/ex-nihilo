# ex-nihilo

Image generation in **pure Common Lisp**: a diffusion engine, a sibling of
[illatio](https://github.com/modus-lisp/illatio), sized for the same two
machines (a 48 GB MacBook Pro, a single 24 GB NVIDIA card).

*Ex nihilo*, Latin: "out of nothing" -- a picture from pure noise.

Nothing runs yet.  [docs/research.md](docs/research.md) chooses the model
(Qwen-Image-2.1-Turbo, with Z-Image-Turbo as the fallback) and the
reference implementation (stable-diffusion.cpp).
[docs/design.md](docs/design.md) is the model read closely -- the prompt
encoder, the DiT and its prefix cache, the schedule, the decoder -- with what
illatio already provides and the milestones.

## Serving

`bin/ex-nihilo` serves OpenAI's image API on loopback (no authentication):

    bin/ex-nihilo [--port 9940] [--weights /mnt/ollama/exn] [--encoder FILE] [--device 0] [--encoder-device 1|cpu]

    curl -s localhost:9940/v1/images/generations \
      -d '{"prompt": "a lighthouse at dusk, oil painting", "size": "1536x1024", "seed": 7}'

`POST /v1/images/generations` takes `prompt`, `size` (`WxH`, sides multiples
of 16 from 256 to 2048, at most 1536x1024's pixels), `n` (1-4),
`response_format` (`b64_json`, or `url` for a `data:` URL), `background`
(`transparent` for RGBA) and `seed`; it answers `{"created", "data": [{"b64_json",
"seed"}]}`. `GET /` is a page to try it from a browser; `GET /v1/models` and
`GET /health` as illatio's. Requests take turns on the GPU.

A picture can be made from another: `image` (a `data:` URL or bare base64 of
a PNG, JPEG or WebP) is encoded by the VAE's encoder, noised, and the steps run
from there, the prompt describing the whole result. `strength` (0-1, default
0.8) is the share of Turbo's eight steps that run: about 0.5 keeps the picture
close, 0.8 restyles and relights it with its composition kept, 0.9 keeps only
its layout, 1 ignores it. Without `size` the result takes the source's shape
(up to a megapixel).

In illatio's server instead, beside a chat model on the same GPUs:
`bin/illatio-server -m CHAT-MODEL --gpu --images /mnt/ollama/exn` (illatio
loads this system for the flag; `src/tenant.lisp`).  The two take turns: these
models wait in pinned host memory and come back for a picture, the chat
model moved out meanwhile.

`--encoder` loads another Qwen3-VL-8B GGUF in place of the weights directory's
`te-q8_0.gguf`, for example AtomicChat's abliterated encoder
(`Qwen-Image-2.1-Turbo-Uncensored-Q8_0.gguf`), which is the same architecture
with its refusal direction projected out.
