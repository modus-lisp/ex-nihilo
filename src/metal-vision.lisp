;;;; metal-vision.lisp — Qwen3-VL-8B's vision tower on Apple silicon, so the
;;;; prompt encoder (illatio's dense path on the same GPU) can see a picture to
;;;; edit: cuda-vision.lisp's tower on Metal, in floats.  The shapes, the
;;;; patches and the positions are host.lisp's.  Kernels: src/metal/exn.metal
;;;; (x_conv as the products, x_layernorm, x_q3v_heads, x_gelu_erf, the VAE's
;;;; x_softmax_rows, the DiT's x_gelu_tanh).  A block is a command buffer.

(in-package #:ex-nihilo)

(defun load-vision (path &key device)
  "The vision tower in the mmproj GGUF at PATH on the GPU (DEVICE: ignored)."
  (declare (ignore device))
  (let* ((g (read-gguf path)) (gpu (exn-gpu)) (v (%make-q3vision :dev gpu)) (bytes 0)
         (t0 (get-internal-real-time)))
    (check-vision g path)
    (multiple-value-bind (tensors pos side) (vision-tensors g)
      (setf (q3vision-pos v) pos (q3vision-side v) side)
      (loop for (name . f) in tensors
            do (incf bytes (* 4 (length f)))
               (setf (gethash name (q3vision-w v)) (put-floats (il::gpu-buffer gpu (* 4 (length f))) f))))
    (format *error-output* "ex-nihilo: the vision tower (~,2F GB as floats) on ~A, ~,1Fs~%" (/ bytes 1d9)
            (il::gpu-name gpu) (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
    v))

(defun vision-release (v)
  "V's scratch given back (VISION-ENCODE's rows with it): the prompt read."
  (maphash (lambda (k b) (declare (ignore k)) (il::gpu-free b)) (q3vision-buf v))
  (clrhash (q3vision-buf v)))

(defun vision-encode (v px h w)
  "PX (H x W RGBA floats, -1 .. 1; H, W multiples of 32) through the tower:
   (values merged deepstack m), MERGED the M x 4096 rows the language model
   reads in place of the picture's M = H W / 1024 image-pad tokens (row-major
   over the 32 px grid), DEEPSTACK a list of three more, GPU buffers (V's
   scratch: valid until VISION-RELEASE)."
  (assert (and (zerop (mod h 32)) (zerop (mod w 32))))
  (let* ((gpu (q3vision-dev v)) (gh (floor h +qv-patch+)) (gw (floor w +qv-patch+)) (n (* gh gw))
         (np (* 16 (ceiling n 16))) (m (floor n 4)) (d +qv-width+) (nh +qv-heads+) (hd +qv-head-dim+)
         (hdp +qv-head-pad+) (f +qv-ffn+) (order (merge-order gh gw)))
    (labels ((wt (name) (or (gethash name (q3vision-w v)) (error "no vision tensor ~A" name)))
             (b (key floats)
               (let ((e (gethash key (q3vision-buf v))) (bytes (* 4 floats)))
                 (if (and e (>= (il::gbuf-bytes e) bytes)) e
                     (progn (when e (il::gpu-free e))
                            (setf (gethash key (q3vision-buf v)) (il::gpu-buffer gpu bytes))))))
             (mm (e x y rows cols count &key (xs cols) (ys rows) w bias)
               ;; Y (COUNT rows, YS apart) = W X (+ BIAS): W ROWS x COLS, X's rows XS apart
               (il::dispatch gpu e "x_conv" (list cols rows 1 count 1 1 xs ys (if bias 1 0))
                             (list w (or bias w) x y) (list (ceiling count 64) (ceiling rows 64)) 256))
             (ew (e kernel count args buffers)
               (il::dispatch gpu e kernel (list* count args) buffers (ceiling count 256) 256))
             (linear (e name x y rows cols count &optional res)
               (mm e x y rows cols count :w (wt (format nil "~A.weight" name)) :bias (wt (format nil "~A.bias" name)))
               (when res (ew e "x_add" (* count rows) '() (list y res))))
             (ln (e name x y width count)
               (il::dispatch gpu e "x_layernorm" (list width 1f-6)
                             (list x (wt (format nil "~A.weight" name)) (wt (format nil "~A.bias" name)) y) count 256))
             (merger (e x out post prefix)
               ;; four patches' rows are one row of 4608: LayerNorm before (per patch) or after, two layers
               (let ((t1 (b :mt1 (* n d))) (t2 (b :mt2 (* n d))))
                 (if post
                     (ln e (format nil "~A.norm" prefix) x t1 (* 4 d) m)
                     (ln e "v.post_ln" x t1 d n))
                 (linear e (if post (format nil "~A.fc1" prefix) "mm.0") t1 t2 (* 4 d) (* 4 d) m)
                 (ew e "x_gelu_erf" (* m 4 d) '() (list t2))
                 (linear e (if post (format nil "~A.fc2" prefix) "mm.2") t2 out +qv-out+ (* 4 d) m))))
      (let ((x (b :x (* n d))) (y (b :y (* n d))) (hb (b :h (* n d))) (qkv (b :qkv (* n 3 d)))
            (q (b :q (* nh n hdp))) (k (b :k (* nh n hdp))) (vt (b :vt (* nh hd np))) (s (b :s (* n np)))
            (att (b :att (* n d))) (up (b :up (* n f))) (cs (b :cs (* n 2 hd))) (pos (b :pos (* n d)))
            (pat (b :pat (* n 768))) (merged (b :merged (* m +qv-out+)))
            (ds (loop for i below 3 collect (b (intern (format nil "DS~D" i) :keyword) (* m +qv-out+)))))
        (put-floats pat (vision-patches px h w order))
        (put-floats pos (vision-pos-embed v order gh gw))
        (put-floats cs (vision-rotary order))
        (sb-alien:alien-funcall                            ; V^T's columns past N stay zero
         (sb-alien:extern-alien "memset" (function sb-sys:system-area-pointer sb-sys:system-area-pointer
                                                   sb-alien:int sb-alien:unsigned-long))
         (il::gbuf-sap vt) 0 (il::gbuf-bytes vt))
        (il::with-gpu-work (e gpu) (linear e "v.patch_embd" pat x d 768 n pos))
        (dotimes (i +qv-depth+)
          (flet ((blk (s) (format nil "v.blk.~D.~A" i s)))
            (il::with-gpu-work (e gpu)
              (ln e (blk "ln1") x hb d n)
              (linear e (blk "attn_qkv") hb qkv (* 3 d) d n)
              (il::dispatch gpu e "x_q3v_heads" (list n np nh hd hdp) (list qkv cs q k vt) (list n nh) 128)
              (dotimes (hh nh)
                (let ((qh (cons q (* 4 hh n hdp))) (kh (cons k (* 4 hh n hdp))) (vh (cons vt (* 4 hh hd np))))
                  (mm e qh s n hdp n :ys np :w kh)
                  (il::dispatch gpu e "x_softmax_rows" (list n n np (coerce (/ (sqrt (float hd 1d0))) 'single-float))
                                (list s) n 256)
                  (mm e s (cons att (* 4 hh hd)) hd np n :xs np :ys d :w vh)))
              (linear e (blk "attn_out") att y d d n x))
            (rotatef x y)
            (il::with-gpu-work (e gpu)
              (ln e (blk "ln2") x hb d n)
              (linear e (blk "ffn_up") hb up f d n)
              (ew e "x_gelu_tanh" (* n f) '() (list up))
              (linear e (blk "ffn_down") up y d f n x))
            (rotatef x y)
            (let ((j (position i +qv-deepstack+)))
              (when j (il::with-gpu-work (e gpu) (merger e x (nth j ds) t (format nil "v.deepstack.~D" i)))))))
        (il::with-gpu-work (e gpu) (merger e x merged nil nil))
        (values merged ds m)))))
