;;;; weights.lisp — the three files, opened and held against the model's
;;;; shape (docs/design.md) before anything reads a weight: the DiT's GGUF,
;;;; the decoder's safetensors, the prompt encoder's GGUF.

(in-package #:ex-nihilo)

;;; --- the DiT: 32 single-stream blocks of 4096, no biases -----------------------

(defparameter *weights*
  #+darwin (merge-pathnames "models/exn/" (user-homedir-pathname))
  #-darwin #p"/mnt/ollama/exn/"
  "Where the weights are (README.md, Weights): the DiT, the decoder, the
   prompt encoder and its vision tower.")

(defun weight-file (name) (namestring (merge-pathnames name *weights*)))

(defparameter +dit-prefix+ "model.diffusion_model.")

(defstruct (dit-shape (:conc-name dit-))
  (width 4096) (layers 32) (head-dim 128) (ffn 12288) (latent 64) (context 4096) (freq 256))

(defun dit-expected (sh)
  "(name . dims) of every tensor the DiT has, dims in ggml's order (the
   contiguous axis first)."
  (let ((w (dit-width sh)) (f (dit-ffn sh)))
    (append
     `(("img_in.weight" ,(dit-latent sh) ,w)
       ("txt_in.text_norm.weight" ,(dit-context sh))
       ("txt_in.in_layer.weight" ,(dit-context sh) ,w)
       ("txt_in.out_layer.weight" ,w ,w)
       ("time_text_embed.timestep_embedder.linear_1.weight" ,(dit-freq sh) ,w)
       ("time_text_embed.timestep_embedder.linear_2.weight" ,w ,w)
       ("modulation.1.weight" ,w ,(* 4 w))
       ("norm_out.linear.weight" ,w ,w)
       ("proj_out.weight" ,w ,(dit-latent sh)))
     (loop for i below (dit-layers sh)
           append (flet ((n (s) (format nil "transformer_blocks.~D.~A" i s)))
                    `((,(n "attn.norm_q.weight") ,(dit-head-dim sh))
                      (,(n "attn.norm_k.weight") ,(dit-head-dim sh))
                      (,(n "attn.to_q.weight") ,w ,w)
                      (,(n "attn.to_k.weight") ,w ,w)
                      (,(n "attn.to_v.weight") ,w ,w)
                      (,(n "attn.to_out.0.weight") ,w ,w)
                      (,(n "img_mlp.gate_up.weight") ,w ,(* 2 f))
                      (,(n "img_mlp.out.weight") ,f ,w)))))))

(defun open-dit (path &optional (sh (make-dit-shape)))
  "The DiT's GGUF at PATH, its every tensor checked for name and shape.
   Signals an error naming each one that differs."
  (let* ((g (read-gguf path)) (problems '()) (expected (dit-expected sh)))
    (unless (equal (meta g "general.architecture") "qwen_image")
      (push (format nil "architecture ~S, not \"qwen_image\"" (meta g "general.architecture")) problems))
    (loop for (name . dims) in expected
          for ti = (gguf-tensor g (concatenate 'string +dit-prefix+ name) nil)
          do (cond ((null ti) (push (format nil "missing ~A" name) problems))
                   ((not (equal (tensor-info-dims ti) dims))
                    (push (format nil "~A is ~A, not ~A" name (tensor-info-dims ti) dims) problems))))
    (let ((extra (- (length (gguf-tensor-order g)) (length expected))))
      (unless (zerop extra) (push (format nil "~D tensors more than the model has" extra) problems)))
    (when problems (error "~A:~{~%  ~A~}" path (reverse problems)))
    g))

;;; --- the decoder: Wan 2.2's, 2-D, 64 latent channels in, RGBA out ---------------

(defparameter +vae-stages+
  ;; (in out spatial-up temporal-up), deepest first: 64 -> 1024 pixels
  '((1152 1152 t t) (1152 1152 t t) (1152 576 t t) (576 288 t nil) (288 144 nil nil)))

(defun vae-expected (&key (z 64) (out 4) (stages +vae-stages+))
  "(name . shape) of every tensor the decoder uses, shapes row-major."
  (let ((top (first (first stages))) (r '()))
    (labels ((add (name &rest shape) (push (cons name shape) r))
             (conv (name o i k &optional (frames t))
               (if frames (add (format nil "~A.weight" name) o i 1 k k) (add (format nil "~A.weight" name) o i k k))
               (add (format nil "~A.bias" name) o))
             (res (p i o)
               (add (format nil "~A.residual.0.gamma" p) i 1 1 1)
               (conv (format nil "~A.residual.2" p) o i 3)
               (add (format nil "~A.residual.3.gamma" p) o 1 1 1)
               (conv (format nil "~A.residual.6" p) o o 3)
               (unless (= i o) (conv (format nil "~A.shortcut" p) o i 1))))
      (conv "conv2" z z 1)
      (conv "decoder.conv1" top z 3)
      (res "decoder.middle.0" top top)
      (add "decoder.middle.1.norm.gamma" top 1 1)
      (conv "decoder.middle.1.to_qkv" (* 3 top) top 1 nil)
      (conv "decoder.middle.1.proj" top top 1 nil)
      (res "decoder.middle.2" top top)
      (loop for (i o up tup) in stages for s from 0
            do (dotimes (j 3)
                 (res (format nil "decoder.upsamples.~D.upsamples.~D" s j) (if (zerop j) i o) o))
               (when up
                 (conv (format nil "decoder.upsamples.~D.upsamples.3.resample.1" s) o o 3 nil)
                 ;; video's: unused on a single frame, but there
                 (when tup (conv (format nil "decoder.upsamples.~D.upsamples.3.time_conv" s) (* 2 o) o 1))))
      (add "decoder.head.0.gamma" (second (car (last stages))) 1 1 1)
      (conv "decoder.head.2" out (second (car (last stages))) 3))
    (nreverse r)))

(defparameter +vae-enc-stages+
  ;; (in out spatial-down temporal-down), the picture's end first: 1024 -> 64 pixels
  '((96 96 t nil) (96 192 t t) (192 384 t t) (384 768 t t) (768 768 nil nil)))

(defun vae-encoder-expected (&key (z 64) (in 4) (stages +vae-enc-stages+))
  "(name . shape) of every tensor the encoder uses (a picture to a latent):
   Wan 2.2's encoder, two residual blocks a stage, the mean and log-variance
   out of its head, CONV1 after it."
  (let ((top (second (car (last stages)))) (r '()))
    (labels ((add (name &rest shape) (push (cons name shape) r))
             (conv (name o i k &optional (frames t))
               (if frames (add (format nil "~A.weight" name) o i 1 k k) (add (format nil "~A.weight" name) o i k k))
               (add (format nil "~A.bias" name) o))
             (res (p i o)
               (add (format nil "~A.residual.0.gamma" p) i 1 1 1)
               (conv (format nil "~A.residual.2" p) o i 3)
               (add (format nil "~A.residual.3.gamma" p) o 1 1 1)
               (conv (format nil "~A.residual.6" p) o o 3)
               (unless (= i o) (conv (format nil "~A.shortcut" p) o i 1))))
      (conv "encoder.conv1" (first (first stages)) in 3)
      (loop for (i o down tdown) in stages for s from 0
            do (dotimes (j 2)
                 (res (format nil "encoder.downsamples.~D.downsamples.~D" s j) (if (zerop j) i o) o))
               (when down
                 (conv (format nil "encoder.downsamples.~D.downsamples.2.resample.1" s) o o 3 nil)
                 (when tdown (conv (format nil "encoder.downsamples.~D.downsamples.2.time_conv" s) o o 1))))
      (res "encoder.middle.0" top top)
      (add "encoder.middle.1.norm.gamma" top 1 1)
      (conv "encoder.middle.1.to_qkv" (* 3 top) top 1 nil)
      (conv "encoder.middle.1.proj" top top 1 nil)
      (res "encoder.middle.2" top top)
      (add "encoder.head.0.gamma" top 1 1 1)
      (conv "encoder.head.2" (* 2 z) top 3)
      (conv "conv1" (* 2 z) (* 2 z) 1))
    (nreverse r)))

(defun open-vae (path)
  "The decoder's safetensors at PATH, every tensor it uses checked."
  (let ((st (read-safetensors path)) (problems '()))
    (loop for (name . shape) in (append (vae-expected) (vae-encoder-expected))
          for e = (gethash name (st-tensors st))
          do (cond ((null e) (push (format nil "missing ~A" name) problems))
                   ((not (equal (second e) shape))
                    (push (format nil "~A is ~A, not ~A" name (second e) shape) problems))))
    (when problems (error "~A:~{~%  ~A~}" path (reverse problems)))
    st))

;;; --- all three ------------------------------------------------------------------

(defun check-weights (&key (dit (weight-file "dit-q8_0.gguf")) (vae (weight-file "vae.safetensors"))
                           (encoder (weight-file "te-q8_0.gguf")))
  "Open each file and report what it holds."
  (let ((g (open-dit dit)))
    (format t "DiT: ~D tensors as expected, ~,2F GB (~{~A~^, ~})~%"
            (length (gguf-tensor-order g))
            (/ (loop for n in (gguf-tensor-order g) sum (tensor-bytes (gguf-tensor g n))) 1d9)
            (remove-duplicates (loop for n in (gguf-tensor-order g) collect (tensor-info-type (gguf-tensor g n))))))
  (let ((st (open-vae vae)))
    (format t "VAE: ~D decoder tensors as expected (~D in the file)~%"
            (length (vae-expected)) (hash-table-count (st-tensors st))))
  (when (probe-file encoder)
    (let ((g (read-gguf encoder)))
      (format t "encoder: ~A, ~D layers of ~D, ~D tensors~%"
              (meta g "general.architecture")
              (meta g (format nil "~A.block_count" (meta g "general.architecture")))
              (meta g (format nil "~A.embedding_length" (meta g "general.architecture")))
              (length (gguf-tensor-order g))))))
