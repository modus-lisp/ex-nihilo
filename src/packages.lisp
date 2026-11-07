;;;; packages.lisp

(defpackage #:ex-nihilo
  (:use #:cl)
  (:nicknames #:exn)
  (:local-nicknames (#:il #:illatio))   ; illatio's internals: its CUDA layer, its loaders
  ;; illatio's reader, vectors and conversions, used as they are
  (:import-from #:illatio
                #:read-gguf #:gguf-tensor #:gguf-tensor-order #:gguf-metadata #:meta
                #:tensor-info-dims #:tensor-info-type #:tensor-elements #:tensor-bytes
                #:f32vec #:bits-single #:half-to-single)
  (:export #:read-safetensors #:st-names #:st-shape #:st-dtype #:st-f32
           #:open-dit #:open-vae #:check-weights
           #:load-encoder #:encode-prompt
           #:upload-dit #:dit-velocity #:upload-vae #:vae-decode
           #:load-engine #:generate #:write-picture))
