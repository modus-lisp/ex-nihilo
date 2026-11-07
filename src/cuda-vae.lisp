;;;; cuda-vae.lisp — the decoder on an NVIDIA GPU (docs/design.md, section 4):
;;;; Wan 2.2's, one frame, every convolution 2-D.  Channels last: a picture
;;;; is H x W rows of C floats.  Kernels: src/cuda/vae.cu, and illatio's
;;;; d41_mm_f32 (the 1 x 1 convolutions), v_bias, v_rms.

(in-package #:ex-nihilo)

(defstruct (gvae (:constructor %make-gvae))
  dev
  (w (make-hash-table :test 'equal))    ; name -> DBUF (3 x 3 weights as rows of (ky, kx, ci))
  (buf (make-hash-table)))              ; scratch: key -> (dbuf . floats)

(defun upload-vae (path &key (device 0))
  "The decoder in the safetensors at PATH on CUDA device DEVICE."
  (let* ((st (open-vae path)) (dev (exn-device device)) (v (%make-gvae :dev dev)) (bytes 0))
    (il::with-cuda
      (il::cuda-current dev)
      (loop for (name . shape) in (vae-expected)
            for f = (st-f32 st name)
            do (when (and (= (length shape) 5) (= (fifth shape) 3)) ; (o i 1 3 3) -> o rows of (ky kx i)
                 (destructuring-bind (o i one k k2) shape
                   (declare (ignore one k2))
                   (let ((r (f32vec (length f))))
                     (dotimes (co o)
                       (dotimes (ci i)
                         (dotimes (ky k)
                           (dotimes (kx k)
                             (setf (aref r (+ (* co 9 i) (* (+ (* ky 3) kx) i) ci))
                                   (aref f (+ (* (+ (* co i) ci) 9) (* ky 3) kx)))))))
                     (setf f r))))
               (when (and (= (length shape) 4) (= (third shape) 3)) ; resample.1: (o i 3 3)
                 (destructuring-bind (o i k k2) shape
                   (declare (ignore k k2))
                   (let ((r (f32vec (length f))))
                     (dotimes (co o)
                       (dotimes (ci i)
                         (dotimes (tap 9)
                           (setf (aref r (+ (* co 9 i) (* tap i) ci)) (aref f (+ (* (+ (* co i) ci) 9) tap))))))
                     (setf f r))))
               (incf bytes (* 4 (length f)))
               (setf (gethash name (gvae-w v)) (il::cuda-upload-vector (il::cuda-alloc dev (* 4 (length f)) :zero nil) f)))
      (il::cuda-sync dev))
    (format *error-output* "ex-nihilo: the decoder (~,2F GB as floats) on ~A~%" (/ bytes 1d9) (il::cudev-name dev))
    v))

(defun vae-sizes (h w)
  "(values out in): the most floats a stage's output and a stage's input
   (at its own size) take, for an H x W latent."
  (let ((hh h) (ww w) (out (* h w (first (first +vae-stages+)))) (in 0))
    (loop for (i o up) in +vae-stages+
          do (setf in (max in (* hh ww (max i o))))
             (when up (setf hh (* 2 hh) ww (* 2 ww)))
             (setf out (max out (* hh ww o))))
    (values out in)))

(defun vae-decode (v latent h w)
  "LATENT (64 x H x W floats, channel-major, denormalized) decoded: the
   picture as 16H x 16W x 4 floats (RGBA, -1 .. 1), row-major, channels
   last, and its height and width."
  (let ((dev (gvae-dev v)))
    (il::with-cuda
      (il::cuda-current dev)
      (multiple-value-bind (max-out max-in) (vae-sizes h w)
        (labels ((wt (name) (or (gethash name (gvae-w v)) (error "no decoder tensor ~A" name)))
                 (b (key floats)
                   (let ((e (gethash key (gvae-buf v))))
                     (if (and e (>= (cdr e) floats)) (car e)
                         (progn (when e (il::cuda-free (car e)))
                                (car (setf (gethash key (gvae-buf v))
                                           (cons (il::cuda-alloc dev (* 4 floats) :zero nil) floats)))))))
                 (ew (kernel count &rest args)
                   (il::launch dev kernel (ceiling count 256) 256 (list* (list :i64 count) args)))
                 (conv1 (name x y cin cout n &optional res)
                   (il::launch dev "d41_mm_f32" (list (ceiling n 64) (ceiling cout 64)) 256
                               (list cin cout n cin cout (wt (format nil "~A.weight" name)) x y))
                   (il::launch dev "v_bias" (ceiling (* n cout) 256) 256
                               (list (list :i64 (* n cout)) cout y (wt (format nil "~A.bias" name)) res)))
                 (conv3 (name x y cin cout hh ww)
                   (il::launch dev "x_conv3" (list (ceiling (* hh ww) 64) (ceiling cout 64)) 256
                               (list cin cout hh ww (wt (format nil "~A.weight" name)) (wt (format nil "~A.bias" name)) x y)))
                 (norm-silu (gamma x y c n)
                   (il::launch dev "v_rms" n 256 (list c 1f-12 x (wt gamma) y))
                   (ew "x_silu" (* n c) y))
                 (resblock (p x out i o hh ww)
                   ;; OUT = conv(silu(norm(conv(silu(norm(X)))))) + (X, or its 1 x 1 shortcut)
                   (let ((n (* hh ww)) (t1 (b :t1 max-in)) (t2 (b :t2 max-in)))
                     (norm-silu (format nil "~A.residual.0.gamma" p) x t1 i n)
                     (conv3 (format nil "~A.residual.2" p) t1 t2 i o hh ww)
                     (norm-silu (format nil "~A.residual.3.gamma" p) t2 t1 o n)
                     (cond ((= i o)
                            (conv3 (format nil "~A.residual.6" p) t1 out o o hh ww)
                            (il::launch dev "v_add" (ceiling (* n o) 256) 256 (list (list :i64 (* n o)) out x)))
                           (t
                            (conv3 (format nil "~A.residual.6" p) t1 t2 o o hh ww)
                            (conv1 (format nil "~A.shortcut" p) x out i o n t2)))))
                 (attention (x out c n)
                   ;; one head over the N pixels: S[t][r] = K[r].Q[t], softmax, A[t] = V^T . P[t]
                   (let ((t1 (b :at (* n c))) (qkv (b :qkv (* n 3 c))) (q (b :q (* n c))) (k (b :k (* n c)))
                         (vv (b :vv (* n c))) (vt (b :vt (* n c))) (s (b :s (* n n))))
                     (il::launch dev "v_rms" n 256 (list c 1f-12 x (wt "decoder.middle.1.norm.gamma") t1))
                     (conv1 "decoder.middle.1.to_qkv" t1 qkv c (* 3 c) n)
                     (ew "x_split3" (* n c) c qkv q k vv)
                     (il::launch dev "d41_mm_f32" (list (ceiling n 64) (ceiling n 64)) 256 (list c n n c n k q s))
                     (il::launch dev "x_softmax_rows" n 256 (list n n (coerce (/ (sqrt (float c 1d0))) 'single-float) s))
                     (il::launch dev "x_transpose" (list (ceiling c 32) (ceiling n 32)) '(32 8) (list n c vv vt))
                     (il::launch dev "d41_mm_f32" (list (ceiling n 64) (ceiling c 64)) 256 (list n c n n c vt s t1))
                     (conv1 "decoder.middle.1.proj" t1 out c c n x))))
          (let* ((n (* h w)) (top (first (first +vae-stages+)))
                 (x (b :x max-out)) (y (b :y max-out)) (keep (b :keep max-in)) (u (b :u max-out))
                 (hh h) (ww w))
            (macrolet ((swap () '(rotatef x y)))
              (let ((rows (f32vec (* n 64))) (a (b :a (* n 64))))
                (dotimes (c 64) (dotimes (i n) (setf (aref rows (+ (* i 64) c)) (aref latent (+ (* c n) i)))))
                (il::cuda-upload-vector a rows)
                (conv1 "conv2" a u 64 64 n)
                (conv3 "decoder.conv1" u x 64 top hh ww))
              (resblock "decoder.middle.0" x y top top hh ww) (swap)
              (attention x y top n) (swap)
              (resblock "decoder.middle.2" x y top top hh ww) (swap)
              (loop for (i o up tup) in +vae-stages+ for s from 0
                    do (let ((np (* hh ww)))
                         (il::cuda-copy-dd keep 0 x 0 (* 4 np i))
                         (dotimes (j 3)
                           (resblock (format nil "decoder.upsamples.~D.upsamples.~D" s j) x y (if (zerop j) i o) o hh ww)
                           (swap))
                         (when up
                           (il::launch dev "x_up2" (ceiling (* 4 np o) 256) 256 (list o hh ww x u))
                           (conv3 (format nil "decoder.upsamples.~D.upsamples.3.resample.1" s) u y o o (* 2 hh) (* 2 ww))
                           (il::launch dev "x_dupup_add" (ceiling (* 4 np o) 256) 256 (list i o (if tup 2 1) hh ww keep y))
                           (swap)
                           (setf hh (* 2 hh) ww (* 2 ww)))))
              ;; the head: norm, SiLU, 3 x 3 to RGBA
              (let* ((np (* hh ww)) (c (second (car (last +vae-stages+)))) (out (b :rgba (* np 4))))
                (norm-silu "decoder.head.0.gamma" x u c np)
                (conv3 "decoder.head.2" u out c 4 hh ww)
                (ew "x_clamp1" (* np 4) out)
                (let ((px (f32vec (* np 4))))
                  (sb-sys:with-pinned-objects (px)
                    (il::cuda-download out (sb-sys:vector-sap px) (* 4 np 4)))
                  (il::cuda-sync dev)
                  (values px hh ww))))))))))
