;;;; cuda-vision.lisp — Qwen3-VL-8B's vision tower on an NVIDIA GPU, so the
;;;; prompt encoder can see a picture to edit (Qwen-Image-2.1's editing: the
;;;; picture is in the prompt).  The mmproj GGUF's ViT -- 27 blocks of 1152,
;;;; 16 heads of 72, a learned 48 x 48 position table resampled to the
;;;; picture's grid and a 2-D rotary in each attention -- its patches 16 px in
;;;; 2 x 2 merge blocks; then the merger, four patches to one token of the
;;;; language model's 4096, and three more (deepstack) after blocks 8, 16 and
;;;; 24, added to its first three layers' outputs.  As transformers'
;;;; Qwen3VLVisionModel; inspect/ref_vision.py is the numpy reference.
;;;; Kernels: illatio's d41_mm_f32, v_bias, v_gelu (erf), v_add; the VAE's
;;;; x_softmax_rows; the DiT's x_gelu_tanh; src/cuda/qvision.cu.

(in-package #:ex-nihilo)

(defun load-vision (path &key (device 1))
  "The vision tower in the mmproj GGUF at PATH on CUDA device DEVICE."
  (let* ((g (read-gguf path)) (dev (exn-device device)) (v (%make-q3vision :dev dev)) (bytes 0)
         (t0 (get-internal-real-time)))
    (check-vision g path)
    (il::with-cuda
      (il::cuda-current dev)
      (multiple-value-bind (tensors pos side) (vision-tensors g)
        (setf (q3vision-pos v) pos (q3vision-side v) side)
        (loop for (name . f) in tensors
              do (incf bytes (* 4 (length f)))
                 (setf (gethash name (q3vision-w v)) (il::cuda-upload-vector (il::cuda-alloc dev (* 4 (length f)) :zero nil) f))))
      (il::cuda-sync dev))
    (format *error-output* "ex-nihilo: the vision tower (~,2F GB as floats) on ~A, ~,1Fs~%" (/ bytes 1d9)
            (il::cudev-name dev) (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
    v))

;;; --- the tower ------------------------------------------------------------------------

(defun vision-encode (v px h w)
  "PX (H x W RGBA floats, -1 .. 1; H, W multiples of 32) through the tower:
   (values merged deepstack m), MERGED the M x 4096 rows the language model
   reads in place of the picture's M = H W / 1024 image-pad tokens (row-major
   over the 32 px grid), DEEPSTACK a list of three more, device buffers on the
   tower's device (its scratch: valid until the next call)."
  (assert (and (zerop (mod h 32)) (zerop (mod w 32))))
  (let* ((dev (q3vision-dev v)) (gh (floor h +qv-patch+)) (gw (floor w +qv-patch+)) (n (* gh gw))
         (np (* 16 (ceiling n 16))) (m (floor n 4)) (d +qv-width+) (nh +qv-heads+) (hd +qv-head-dim+)
         (hdp +qv-head-pad+) (f +qv-ffn+) (order (merge-order gh gw)))
    (il::with-cuda
      (il::cuda-current dev)
      (labels ((wt (name) (or (gethash name (q3vision-w v)) (error "no vision tensor ~A" name)))
               (b (key floats)
                 (let ((e (gethash key (q3vision-buf v))))
                   (if (and e (>= (cdr e) floats)) (car e)
                       (progn (when e (il::cuda-free (car e)))
                              (car (setf (gethash key (q3vision-buf v))
                                         (cons (il::cuda-alloc dev (* 4 floats) :zero nil) floats)))))))
               (mm (x y rows cols count &key (xs cols) (ys rows) w)
                 ;; Y (COUNT rows, YS apart) = W X: W ROWS x COLS, X's rows XS apart
                 (il::launch dev "d41_mm_f32" (list (ceiling count 64) (ceiling rows 64)) 256
                             (list cols rows count xs ys w x y)))
               (linear (name x y rows cols count &optional res)
                 (mm x y rows cols count :w (wt (format nil "~A.weight" name)))
                 (il::launch dev "v_bias" (ceiling (* count rows) 256) 256
                             (list (list :i64 (* count rows)) rows y (wt (format nil "~A.bias" name)) res)))
               (ln (name x y width count)
                 (il::launch dev "q3v_layernorm" count 256
                             (list width 1f-6 x (wt (format nil "~A.weight" name)) (wt (format nil "~A.bias" name)) y)))
               (merger (x out post prefix)
                 ;; four patches' rows are one row of 4608: LayerNorm before (per patch) or after, two layers
                 (let ((t1 (b :mt1 (* n d))) (t2 (b :mt2 (* n d))))
                   (if post
                       (ln (format nil "~A.norm" prefix) x t1 (* 4 d) m)
                       (ln "v.post_ln" x t1 d n))
                   (linear (if post (format nil "~A.fc1" prefix) "mm.0") t1 t2 (* 4 d) (* 4 d) m)
                   (il::launch dev "v_gelu" (ceiling (* m 4 d) 256) 256 (list (list :i64 (* m 4 d)) t2))
                   (linear (if post (format nil "~A.fc2" prefix) "mm.2") t2 out +qv-out+ (* 4 d) m))))
        (let ((x (b :x (* n d))) (y (b :y (* n d))) (hb (b :h (* n d))) (qkv (b :qkv (* n 3 d)))
              (q (b :q (* nh n hdp))) (k (b :k (* nh n hdp))) (vt (b :vt (* nh hd np))) (s (b :s (* n np)))
              (att (b :att (* n d))) (up (b :up (* n f))) (cs (b :cs (* n 2 hd))) (pos (b :pos (* n d)))
              (pat (b :pat (* n 768))) (merged (b :merged (* m +qv-out+)))
              (ds (loop for i below 3 collect (b (intern (format nil "DS~D" i) :keyword) (* m +qv-out+)))))
          (il::cuda-upload-vector pat (vision-patches px h w order))
          (il::cuda-upload-vector pos (vision-pos-embed v order gh gw))
          (il::cuda-upload-vector cs (vision-rotary order))
          (linear "v.patch_embd" pat x d 768 n pos)
          (il::cuda-zero vt)                                  ; V^T's columns past N stay zero
          (dotimes (i +qv-depth+)
            (flet ((blk (s) (format nil "v.blk.~D.~A" i s)))
              (ln (blk "ln1") x hb d n)
              (linear (blk "attn_qkv") hb qkv (* 3 d) d n)
              (il::launch dev "q3v_heads" (list n nh) 128 (list n np nh hd hdp qkv cs q k vt))
              (dotimes (hh nh)
                (let ((qh (cons q (* 4 hh n hdp))) (kh (cons k (* 4 hh n hdp))) (vh (cons vt (* 4 hh hd np))))
                  (mm qh s n hdp n :ys np :w kh)
                  (il::launch dev "x_softmax_rows" n 256 (list n n np (coerce (/ (sqrt (float hd 1d0))) 'single-float) s))
                  (mm s (cons att (* 4 hh hd)) hd np n :ys d :w vh)))
              (linear (blk "attn_out") att y d d n x) (rotatef x y)
              (ln (blk "ln2") x hb d n)
              (linear (blk "ffn_up") hb up f d n)
              (il::launch dev "x_gelu_tanh" (ceiling (* n f) 256) 256 (list (list :i64 (* n f)) up))
              (linear (blk "ffn_down") up y d f n x) (rotatef x y)
              (let ((j (position i +qv-deepstack+)))
                (when j (merger x (nth j ds) t (format nil "v.deepstack.~D" i))))))
          (merger x merged nil nil)
          (values merged ds m))))))

(defun vision-release (v)
  "V's scratch freed (VISION-ENCODE's rows with it): the prompt read."
  (maphash (lambda (k e) (declare (ignore k)) (il::cuda-free (car e))) (q3vision-buf v))
  (clrhash (q3vision-buf v)))

(defun download-rows (buf floats dev)
  (let ((v (f32vec floats)))
    (il::with-cuda
      (il::cuda-current dev)
      (sb-sys:with-pinned-objects (v) (il::cuda-download buf (sb-sys:vector-sap v) (* 4 floats)))
      (il::cuda-sync dev))
    v))
