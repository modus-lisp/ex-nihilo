;;;; cuda-vae.lisp — the decoder (a latent to a picture) and the encoder (a
;;;; picture to a latent, for a picture made from another) on an NVIDIA GPU
;;;; (docs/design.md, section 4): Wan 2.2's, one frame, every convolution
;;;; 2-D.  Channels last: a picture is H x W rows of C floats.  Kernels:
;;;; src/cuda/vae.cu, and illatio's d41_mm_f32 (the 1 x 1 convolutions),
;;;; v_bias, v_rms, v_add.

(in-package #:ex-nihilo)

(defstruct (gvae (:constructor %make-gvae))
  dev
  (w (make-hash-table :test 'equal))    ; name -> DBUF (3 x 3 weights as rows of (ky, kx, ci))
  (buf (make-hash-table)))              ; scratch: key -> (dbuf . floats)

(defun upload-vae (path &key (device 0))
  "The decoder and encoder in the safetensors at PATH on CUDA device DEVICE.
   A 3 x 3 convolution's input channels are made a multiple of 16 (x_conv3's
   tiles): the encoder's first, of RGBA, padded with zero weights."
  (let* ((st (open-vae path)) (dev (exn-device device)) (v (%make-gvae :dev dev)) (bytes 0))
    (il::with-cuda
      (il::cuda-current dev)
      (loop for (name . f) in (vae-tensors st)
            do (incf bytes (* 4 (length f)))
               (setf (gethash name (gvae-w v)) (il::cuda-upload-vector (il::cuda-alloc dev (* 4 (length f)) :zero nil) f)))
      (il::cuda-sync dev))
    (format *error-output* "ex-nihilo: the decoder and encoder (~,2F GB as floats) on ~A~%" (/ bytes 1d9) (il::cudev-name dev))
    v))

(defun vae-release (v)
  "V's scratch freed: a decode at 1536 x 1024 leaves some 11 GB, which the
   DiT wants back for its next picture."
  (maphash (lambda (k e) (declare (ignore k)) (il::cuda-free (car e))) (gvae-buf v))
  (clrhash (gvae-buf v)))

;;; The building blocks both directions use, as local functions of the body:
;;; WT, B (scratch by key), EW, CONV1, CONV3, CONV3S2, NORM-SILU, RESBLOCK,
;;; ATTENTION.  MAX-IN: the floats a residual block's scratch must hold.
(defmacro with-vae-ops ((v dev max-in) &body body)
  `(let ((v ,v) (dev ,dev) (max-in ,max-in))
        (labels ((wt (name) (or (gethash name (gvae-w v)) (error "no VAE tensor ~A" name)))
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
                 (attention (prefix x out c n)
                   ;; one head over the N pixels: S[t][r] = K[r].Q[t], softmax, A[t] = V^T . P[t].  The
                   ;; second product sums over the pixels, and d41_mm_f32 takes its sums 16 at a time: S's
                   ;; rows and V^T's are NP long, N padded to a multiple of 16 with zeros
                   (let* ((np (* 16 (ceiling n 16)))
                          (t1 (b :at (* n c))) (qkv (b :qkv (* n 3 c))) (q (b :q (* n c))) (k (b :k (* n c)))
                          (vv (b :vv (* np c))) (vt (b :vt (* np c))) (s (b :s (* n np))))
                     (il::launch dev "v_rms" n 256 (list c 1f-12 x (wt (format nil "~A.norm.gamma" prefix)) t1))
                     (conv1 (format nil "~A.to_qkv" prefix) t1 qkv c (* 3 c) n)
                     (when (/= np n) (il::cuda-zero vv))   ; V's padding rows: zero
                     (ew "x_split3" (* n c) c qkv q k vv)
                     (il::launch dev "d41_mm_f32" (list (ceiling n 64) (ceiling n 64)) 256 (list c n n c np k q s))
                     (il::launch dev "x_softmax_rows" n 256 (list n n np (coerce (/ (sqrt (float c 1d0))) 'single-float) s))
                     (il::launch dev "x_transpose" (list (ceiling c 32) (ceiling np 32)) '(32 8) (list np c vv vt))
                     (il::launch dev "d41_mm_f32" (list (ceiling n 64) (ceiling c 64)) 256 (list np c n np c vt s t1))
                     (conv1 (format nil "~A.proj" prefix) t1 out c c n x)))
                 (conv3s2 (name x y cin cout hh ww)
                   (il::launch dev "x_conv3s2" (list (ceiling (* (floor hh 2) (floor ww 2)) 64) (ceiling cout 64)) 256
                               (list cin cout hh ww (wt (format nil "~A.weight" name)) (wt (format nil "~A.bias" name)) x y))))
          (declare (ignorable #'ew #'conv1 #'conv3 #'conv3s2 #'norm-silu #'resblock #'attention))
          ,@body)))

(defun vae-decode (v latent h w)
  "LATENT (64 x H x W floats, channel-major, denormalized) decoded: the
   picture as 16H x 16W x 4 floats (RGBA, -1 .. 1), row-major, channels
   last, and its height and width."
  (let ((dev (gvae-dev v)))
    (il::with-cuda
      (il::cuda-current dev)
      (multiple-value-bind (max-out max-in) (vae-sizes h w)
        (with-vae-ops (v dev max-in)
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
              (attention "decoder.middle.1" x y top n) (swap)
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

(defun vae-encode (v px h w)
  "PX (H x W x 4 floats, RGBA in -1 .. 1, rows of W pixels; H and W multiples
   of 16) encoded: the latent's mean, 64 x H/16 x W/16 floats channel-major,
   as the decoder takes it (denormalized)."
  (assert (and (zerop (mod h 16)) (zerop (mod w 16))))
  (let ((dev (gvae-dev v)) (n (* h w)))
    (il::with-cuda
      (il::cuda-current dev)
      (multiple-value-bind (max-out max-in) (vae-enc-sizes h w)
        (with-vae-ops (v dev max-in)
          (let ((x (b :x max-out)) (y (b :y max-out)) (keep (b :keep max-in)) (hh h) (ww w))
            (macrolet ((swap () '(rotatef x y)))
              ;; RGBA, padded to 16 channels (the first convolution's weights are, with zeros)
              (let ((rows (f32vec (* n 16))) (a (b :a (* n 16))))
                (dotimes (i n) (dotimes (c 4) (setf (aref rows (+ (* i 16) c)) (aref px (+ (* i 4) c)))))
                (il::cuda-upload-vector a rows)
                (conv3 "encoder.conv1" a x 16 (first (first +vae-enc-stages+)) hh ww))
              ;; the stages: two residual blocks, a stride-2 convolution, and AvgDown3D's shortcut around them
              (loop for (i o down tdown) in +vae-enc-stages+ for s from 0
                    do (let ((np (* hh ww)))
                         (il::cuda-copy-dd keep 0 x 0 (* 4 np i))
                         (dotimes (j 2)
                           (resblock (format nil "encoder.downsamples.~D.downsamples.~D" s j) x y (if (zerop j) i o) o hh ww)
                           (swap))
                         (cond (down
                                (conv3s2 (format nil "encoder.downsamples.~D.downsamples.2.resample.1" s) x y o o hh ww)
                                (setf hh (floor hh 2) ww (floor ww 2))
                                (il::launch dev "x_avgdown_add" (ceiling (* hh ww o) 256) 256
                                            (list i o (if tdown 2 1) (* 2 hh) (* 2 ww) keep y))
                                (swap))
                               (t (assert (= i o)) (ew "v_add" (* np o) x keep)))))
              (let* ((top (second (car (last +vae-enc-stages+)))) (np (* hh ww)))
                (resblock "encoder.middle.0" x y top top hh ww) (swap)
                (attention "encoder.middle.1" x y top np) (swap)
                (resblock "encoder.middle.2" x y top top hh ww) (swap)
                ;; the head: norm, SiLU, 3 x 3 to the mean and log-variance, CONV1
                (norm-silu "encoder.head.0.gamma" x y top np)
                (conv3 "encoder.head.2" y x top 128 hh ww)
                (conv1 "conv1" x y 128 128 np)
                (let ((rows (f32vec (* np 128))) (z (f32vec (* 64 np))))
                  (sb-sys:with-pinned-objects (rows)
                    (il::cuda-download y (sb-sys:vector-sap rows) (* 4 np 128)))
                  (il::cuda-sync dev)
                  (dotimes (c 64) (dotimes (i np) (setf (aref z (+ (* c np) i)) (aref rows (+ (* i 128) c)))))
                  z)))))))))
