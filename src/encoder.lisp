;;;; encoder.lisp — the prompt: Qwen3-VL-8B's text model (illatio's qwen3
;;;; forward, on the CPU) read for its last block's states, before the final
;;;; norm, with the template's system turn cut off (docs/design.md, section 1).

(in-package #:ex-nihilo)

(defparameter +system-turn+
  (format nil "<|im_start|>system~%Comprehend and analyze the provided prompt.<|im_end|>~%"))

(defun prompt-text (prompt)
  (format nil "~A<|im_start|>user~%~A<|im_end|>~%<|im_start|>assistant~%"
          +system-turn+ (if (string= prompt "") " " prompt)))

(defun load-encoder (path &key device (threads (il::default-thread-count)))
  "The prompt encoder: the qwen3vl GGUF at PATH, its text model -- on CUDA
   device DEVICE (illatio's dense path), or on the CPU when DEVICE is NIL."
  #+(and sbcl linux) (when device (return-from load-encoder (il::load-model-cuda path :device device)))
  (il::load-model path :threads threads))

(defun encode-prompt (model prompt)
  "(values states p): the P x 4096 floats the DiT reads for PROMPT."
  (let* ((ids (il::tokenize model (prompt-text prompt) :add-bos nil))
         (skip (length (il::tokenize model +system-turn+ :add-bos nil)))
         (d (il::model-d model)) (n (length ids)) (p (- n skip))
         (ctx (il::make-context model :n-ctx (+ n 8)))
         (all (f32vec (* n d))))
    (unwind-protect
         (loop with gpu = #+(and sbcl linux) (il::model-cuda model) #-(and sbcl linux) nil
               with step = (if gpu il::*cuda-chunk* il::*prefill-chunk*)
               with done = 0
               while (< done n)
               do (let* ((chunk (subseq ids done (min n (+ done step)))) (k (length chunk)))
                    ;; each pass leaves every token's state after the last block in its scratch
                    (cond #+(and sbcl linux)
                          (gpu
                           (il::cuda-forward-n ctx chunk nil)
                           (let ((x (il::cctx-x (il::cuda-state ctx))))
                             (il::with-cuda
                               (il::cuda-current (il::cmodel-dev gpu))
                               (sb-sys:with-pinned-objects (all)
                                 (il::cuda-download x (sb-sys:sap+ (sb-sys:vector-sap all) (* 4 done d)) (* 4 k d)))
                               (il::cuda-sync (il::cmodel-dev gpu)))))
                          (t
                           (il::forward-batch ctx chunk :logits nil)
                           (replace all (il::batch-scratch-x (il::batch-scratch ctx)) :start1 (* done d) :end2 (* k d))))
                    (incf done k)))
      (il::free-context ctx))
    (values (subseq all (* skip d)) p)))
