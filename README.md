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

    bin/ex-nihilo [--port 9940] [--weights /dev/shm/exn] [--device 0] [--encoder-device 1|cpu]

    curl -s localhost:9940/v1/images/generations \
      -d '{"prompt": "a lighthouse at dusk, oil painting", "size": "1536x1024", "seed": 7}'

`POST /v1/images/generations` takes `prompt`, `size` (`WxH`, sides multiples
of 16 from 256 to 2048, at most 1536x1024's pixels), `n` (1-4),
`response_format` (`b64_json`, or `url` for a `data:` URL), `background`
(`transparent` for RGBA) and `seed`; it answers `{"created", "data": [{"b64_json",
"seed"}]}`. `GET /` is a page to try it from a browser; `GET /v1/models` and
`GET /health` as illatio's. Requests take turns on the GPU.
