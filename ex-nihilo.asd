;;;; ex-nihilo.asd

(defsystem "ex-nihilo"
  :description "Image generation in pure Common Lisp: Qwen-Image-2.1-Turbo's
                prompt encoder, diffusion transformer and decoder, on
                illatio's weights, kernels and devices."
  :version "0.0.1"
  :author "ynniv"
  :license "MIT"
  :depends-on ("illatio" "json-simple" "png-pure")
  :serial t
  :components
  ((:module "src"
    :serial t
    :components
    ((:file "packages")
     (:file "safetensors")   ; the decoder's weights: a JSON header and raw tensors
     (:file "weights")       ; the three files opened and checked against the model's shape
     (:file "encoder")       ; the prompt: Qwen3-VL's text model, its last block's states
     (:file "cuda-dit" :if-feature (:and :sbcl :linux))   ; the DiT on an NVIDIA GPU (kernels: src/cuda/)
     (:file "cuda-vae" :if-feature (:and :sbcl :linux))   ; the decoder there
     (:file "generate")))))  ; noise, eight steps, the decoder: a PNG

(defsystem "ex-nihilo/server"
  :description "OpenAI's image API: POST /v1/images/generations, and a page to try it."
  :depends-on ("ex-nihilo" "illatio/server")
  :components ((:module "src" :components ((:file "server")))))
