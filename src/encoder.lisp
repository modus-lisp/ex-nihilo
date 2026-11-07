;;;; encoder.lisp — the prompt: Qwen3-VL-8B's text model (illatio's qwen3
;;;; forward, on the CPU) read for its last block's states, before the final
;;;; norm, with the template's system turn cut off (docs/design.md, section 1).

(in-package #:ex-nihilo)

(defparameter +system-turn+
  (format nil "<|im_start|>system~%Comprehend and analyze the provided prompt.<|im_end|>~%"))

(defun prompt-text (prompt)
  (format nil "~A<|im_start|>user~%~A<|im_end|>~%<|im_start|>assistant~%"
          +system-turn+ (if (string= prompt "") " " prompt)))

(defun load-encoder (path &key (threads (il::default-thread-count)))
  "The prompt encoder: the qwen3vl GGUF at PATH, its text model."
  (il::load-model path :threads threads))

(defun encode-prompt (model prompt)
  "(values states p): the P x 4096 floats the DiT reads for PROMPT."
  (let* ((ids (il::tokenize model (prompt-text prompt) :add-bos nil))
         (skip (length (il::tokenize model +system-turn+ :add-bos nil)))
         (d (il::model-d model)) (n (length ids)) (p (- n skip))
         (ctx (il::make-context model :n-ctx (+ n 8)))
         (all (f32vec (* n d))))
    (unwind-protect
         (loop with done = 0
               while (< done n)
               do (let* ((chunk (subseq ids done (min n (+ done il::*prefill-chunk*)))) (k (length chunk)))
                    (il::forward-batch ctx chunk :logits nil)
                    ;; the scratch's rows: each token's state after the last block
                    (replace all (il::batch-scratch-x (il::batch-scratch ctx)) :start1 (* done d) :end2 (* k d))
                    (incf done k)))
      (il::free-context ctx))
    (values (subseq all (* skip d)) p)))
