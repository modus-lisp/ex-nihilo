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
