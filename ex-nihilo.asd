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
     (:file "host")          ; the host's share of a step, either GPU's
     (:file "cuda-dit" :if-feature (:and :sbcl :linux))   ; the DiT on an NVIDIA GPU (kernels: src/cuda/)
     (:file "cuda-vae" :if-feature (:and :sbcl :linux))   ; the decoder there
     (:file "cuda-vision" :if-feature (:and :sbcl :linux)) ; Qwen3-VL's vision tower, for editing
     (:file "metal-dit" :if-feature (:and :sbcl :darwin))  ; the DiT on Apple silicon (kernels: src/metal/)
     (:file "metal-vae" :if-feature (:and :sbcl :darwin))  ; the decoder there
     (:file "generate")))))  ; noise, eight steps, the decoder: a PNG

(defsystem "ex-nihilo/server"
  :description "OpenAI's image API: POST /v1/images/generations, and a page to try it."
  :depends-on ("ex-nihilo" "illatio/server")
  :components ((:module "src" :serial t
                :components ((:file "server")
                             (:file "tenant" :if-feature (:and :sbcl :linux))))))   ; in illatio's server, sharing its GPUs
