;;;; metal-vae.lisp — the decoder (a latent to a picture) and the encoder (a
;;;; picture to a latent) on Apple silicon: cuda-vae.lisp's operations on
;;;; Metal.  Channels last: a picture is H x W rows of C floats.  Kernels:
;;;; src/metal/exn.metal (x_conv_h takes the 3 x 3 convolutions, the 1 x 1s
;;;; and the attention's products as half products summed in floats; x_conv
;;;; in floats, with *VAE-HALF* off).  Each operation is a command
;;;; buffer of its own, and waited for.

(in-package #:ex-nihilo)

(defvar *vae-half* t
  "Whether the convolutions run as halves on the matrix units (x_conv_h:
   inputs and weights rounded to halves, sums in floats) or in floats
   (x_conv).")

(defstruct (mvae (:constructor %make-mvae))
  gpu
  (w (make-hash-table :test 'equal))    ; name -> buffer of floats (3 x 3 weights as rows of (ky, kx, ci))
  (buf (make-hash-table)))              ; scratch: key -> buffer

(defun upload-vae (path &key device)
  "The decoder and encoder in the safetensors at PATH on the GPU (DEVICE: ignored)."
  (declare (ignore device))
  (let* ((st (open-vae path)) (gpu (exn-gpu)) (v (%make-mvae :gpu gpu)) (bytes 0))
    (loop for (name . f) in (vae-tensors st)
          do (incf bytes (* 4 (length f)))
             (setf (gethash name (mvae-w v)) (put-floats (il::gpu-buffer gpu (* 4 (length f))) f)))
    (format *error-output* "ex-nihilo: the decoder and encoder (~,2F GB as floats) on ~A~%"
            (/ bytes 1d9) (il::gpu-name gpu))
    v))

(defun vae-release (v)
  "V's scratch given back."
  (maphash (lambda (k b) (declare (ignore k)) (il::gpu-free b)) (mvae-buf v))
  (clrhash (mvae-buf v)))

(defun copy-bytes (dst src bytes &optional (dst-off 0) (src-off 0))
  "BYTES of shared buffer SRC into DST, on the host (the GPU's work on both done)."
  (sb-alien:alien-funcall
   (sb-alien:extern-alien "memcpy" (function sb-sys:system-area-pointer sb-sys:system-area-pointer
                                             sb-sys:system-area-pointer sb-alien:unsigned-long))
   (sb-sys:sap+ (il::gbuf-sap dst) dst-off) (sb-sys:sap+ (il::gbuf-sap src) src-off) bytes))

;;; cuda-vae.lisp's building blocks, as local functions of the body: WT, B
;;; (scratch by key), EW, CONV1, CONV3, CONV3S2, NORM-SILU, RESBLOCK,
;;; ATTENTION.  MAX-IN: the floats a residual block's scratch must hold.
(defmacro with-metal-vae-ops ((v max-in) &body body)
  `(let* ((v ,v) (gpu (mvae-gpu v)) (max-in ,max-in))
     (labels ((wt (name) (or (gethash name (mvae-w v)) (error "no VAE tensor ~A" name)))
              (b (key floats)
                (let ((e (gethash key (mvae-buf v))) (bytes (* 4 floats)))
                  (if (and e (>= (il::gbuf-bytes e) bytes)) e
                      (progn (when e (il::gpu-free e))
                             (setf (gethash key (mvae-buf v)) (il::gpu-buffer gpu bytes))))))
              (ew (kernel count args buffers)
                (il::with-gpu-work (e gpu)
                  (il::dispatch gpu e kernel (list* count args) buffers (ceiling count 256) 256)))
              (grid (kernel count args buffers)   ; COUNT threads, the kernel's arguments without it
                (il::with-gpu-work (e gpu)
                  (il::dispatch gpu e kernel args buffers (ceiling count 256) 256)))
              (conv (wts bias x y cin cout hh ww taps stride &key (ldx cin) (ldy cout))
                (let ((pixels (if (= stride 2) (* (floor hh 2) (floor ww 2)) (* hh ww)))
                      (args (list cin cout hh ww taps stride ldx ldy (if bias 1 0))))
                  (il::with-gpu-work (e gpu)
                    ;; halves on the matrix units when K stays within a tap (CIN a multiple of 16)
                    (cond ((and *vae-half* (zerop (mod cin 32)) (zerop (mod ldx 4)))
                           (il::dispatch gpu e "x_conv_h32" args (list wts (or bias wts) x y)
                                         (list (ceiling pixels 64) (ceiling cout 64)) 128 :shared 16384))
                          ((and *vae-half* (zerop (mod cin 16)) (zerop (mod ldx 4)))
                           (il::dispatch gpu e "x_conv_h16" args (list wts (or bias wts) x y)
                                         (list (ceiling pixels 64) (ceiling cout 64)) 128 :shared 16384))
                          (t (il::dispatch gpu e "x_conv" args (list wts (or bias wts) x y)
                                           (list (ceiling pixels 64) (ceiling cout 64)) 256))))))
              (conv1 (name x y cin cout n &optional res)
                (conv (wt (format nil "~A.weight" name)) (wt (format nil "~A.bias" name)) x y cin cout 1 n 1 1)
                (when res (ew "x_add" (* n cout) '() (list y res))))
              (conv3 (name x y cin cout hh ww)
                (conv (wt (format nil "~A.weight" name)) (wt (format nil "~A.bias" name)) x y cin cout hh ww 9 1))
              (conv3s2 (name x y cin cout hh ww)
                (conv (wt (format nil "~A.weight" name)) (wt (format nil "~A.bias" name)) x y cin cout hh ww 9 2))
              (norm-silu (gamma x y c n)
                (il::with-gpu-work (e gpu)
                  (il::dispatch gpu e "x_rms" (list c 1f-12) (list x (wt gamma) y) n 256)
                  (il::dispatch gpu e "x_silu" (list (* n c)) (list y) (ceiling (* n c) 256) 256)))
              (resblock (p x out i o hh ww)
                ;; OUT = conv(silu(norm(conv(silu(norm(X)))))) + (X, or its 1 x 1 shortcut)
                (let ((n (* hh ww)) (t1 (b :t1 max-in)) (t2 (b :t2 max-in)))
                  (norm-silu (format nil "~A.residual.0.gamma" p) x t1 i n)
                  (conv3 (format nil "~A.residual.2" p) t1 t2 i o hh ww)
                  (norm-silu (format nil "~A.residual.3.gamma" p) t2 t1 o n)
                  (cond ((= i o)
                         (conv3 (format nil "~A.residual.6" p) t1 out o o hh ww)
                         (ew "x_add" (* n o) '() (list out x)))
                        (t
                         (conv3 (format nil "~A.residual.6" p) t1 t2 o o hh ww)
                         (conv1 (format nil "~A.shortcut" p) x out i o n t2)))))
              (attention (prefix x out c n)
                ;; one head over the N pixels: S = Q K^T (S's rows NP long, N padded to 16
                ;; with zero rows of V), softmax, A = P V
                (let* ((np (* 16 (ceiling n 16)))
                       (t1 (b :at (* n c))) (qkv (b :qkv (* n 3 c))) (q (b :q (* n c))) (k (b :k (* n c)))
                       (vv (b :vv (* np c))) (vt (b :vt (* np c))) (s (b :s (* n np))))
                  (il::with-gpu-work (e gpu)
                    (il::dispatch gpu e "x_rms" (list c 1f-12) (list x (wt (format nil "~A.norm.gamma" prefix)) t1) n 256))
                  (conv1 (format nil "~A.to_qkv" prefix) t1 qkv c (* 3 c) n)
                  (when (/= np n)   ; V's padding rows: zero
                    (sb-alien:alien-funcall
                     (sb-alien:extern-alien "memset" (function sb-sys:system-area-pointer sb-sys:system-area-pointer
                                                               sb-alien:int sb-alien:unsigned-long))
                     (il::gbuf-sap vv) 0 (il::gbuf-bytes vv)))
                  (ew "x_split3" (* n c) (list c) (list qkv q k vv))
                  (conv k nil q s c n 1 n 1 1 :ldx c :ldy np)                    ; S[t][r] = K[r] . Q[t]
                  (il::with-gpu-work (e gpu)
                    (il::dispatch gpu e "x_softmax_rows" (list n n np (coerce (/ (sqrt (float c 1d0))) 'single-float))
                                  (list s) n 256)
                    (il::dispatch gpu e "x_transpose" (list np c) (list vv vt) (list (ceiling c 32) (ceiling np 32)) '(32 8)))
                  (conv vt nil s t1 np c 1 n 1 1 :ldx np :ldy c)                 ; A[t] = V^T . P[t]
                  (conv1 (format nil "~A.proj" prefix) t1 out c c n x))))
       (declare (ignorable #'ew #'grid #'conv1 #'conv3 #'conv3s2 #'norm-silu #'resblock #'attention))
       ,@body)))

(defun vae-decode (v latent h w)
  "LATENT (64 x H x W floats, channel-major, denormalized) decoded: the
   picture as 16H x 16W x 4 floats (RGBA, -1 .. 1), row-major, channels
   last, and its height and width."
  (multiple-value-bind (max-out max-in) (vae-sizes h w)
    (with-metal-vae-ops (v max-in)
      (let* ((n (* h w)) (top (first (first +vae-stages+)))
             (x (b :x max-out)) (y (b :y max-out)) (keep (b :keep max-in)) (u (b :u max-out))
             (hh h) (ww w))
        (macrolet ((swap () '(rotatef x y)))
          (let ((rows (f32vec (* n 64))) (a (b :a (* n 64))))
            (dotimes (c 64) (dotimes (i n) (setf (aref rows (+ (* i 64) c)) (aref latent (+ (* c n) i)))))
            (put-floats a rows)
            (conv1 "conv2" a u 64 64 n)
            (conv3 "decoder.conv1" u x 64 top hh ww))
          (resblock "decoder.middle.0" x y top top hh ww) (swap)
          (attention "decoder.middle.1" x y top n) (swap)
          (resblock "decoder.middle.2" x y top top hh ww) (swap)
          (loop for (i o up tup) in +vae-stages+ for s from 0
                do (let ((np (* hh ww)))
                     (copy-bytes keep x (* 4 np i))
                     (dotimes (j 3)
                       (resblock (format nil "decoder.upsamples.~D.upsamples.~D" s j) x y (if (zerop j) i o) o hh ww)
                       (swap))
                     (when up
                       (grid "x_up2" (* 4 np o) (list o hh ww) (list x u))
                       (conv3 (format nil "decoder.upsamples.~D.upsamples.3.resample.1" s) u y o o (* 2 hh) (* 2 ww))
                       (grid "x_dupup_add" (* 4 np o) (list i o (if tup 2 1) hh ww) (list keep y))
                       (swap)
                       (setf hh (* 2 hh) ww (* 2 ww)))))
          (let* ((np (* hh ww)) (c (second (car (last +vae-stages+)))) (out (b :rgba (* np 4))))
            (norm-silu "decoder.head.0.gamma" x u c np)
            (conv3 "decoder.head.2" u out c 4 hh ww)
            (ew "x_clamp1" (* np 4) '() (list out))
            (values (get-floats out (* np 4)) hh ww)))))))

(defun vae-encode (v px h w)
  "PX (H x W x 4 floats, RGBA in -1 .. 1, rows of W pixels; H and W multiples
   of 16) encoded: the latent's mean, 64 x H/16 x W/16 floats channel-major,
   as the decoder takes it (denormalized)."
  (assert (and (zerop (mod h 16)) (zerop (mod w 16))))
  (let ((n (* h w)))
    (multiple-value-bind (max-out max-in) (vae-enc-sizes h w)
      (with-metal-vae-ops (v max-in)
        (let ((x (b :x max-out)) (y (b :y max-out)) (keep (b :keep max-in)) (hh h) (ww w))
          (macrolet ((swap () '(rotatef x y)))
            ;; RGBA, padded to 16 channels (the first convolution's weights are, with zeros)
            (let ((rows (f32vec (* n 16))) (a (b :a (* n 16))))
              (dotimes (i n) (dotimes (c 4) (setf (aref rows (+ (* i 16) c)) (aref px (+ (* i 4) c)))))
              (put-floats a rows)
              (conv3 "encoder.conv1" a x 16 (first (first +vae-enc-stages+)) hh ww))
            ;; the stages: two residual blocks, a stride-2 convolution, and AvgDown3D's shortcut around them
            (loop for (i o down tdown) in +vae-enc-stages+ for s from 0
                  do (let ((np (* hh ww)))
                       (copy-bytes keep x (* 4 np i))
                       (dotimes (j 2)
                         (resblock (format nil "encoder.downsamples.~D.downsamples.~D" s j) x y (if (zerop j) i o) o hh ww)
                         (swap))
                       (cond (down
                              (conv3s2 (format nil "encoder.downsamples.~D.downsamples.2.resample.1" s) x y o o hh ww)
                              (setf hh (floor hh 2) ww (floor ww 2))
                              (grid "x_avgdown_add" (* hh ww o) (list i o (if tdown 2 1) (* 2 hh) (* 2 ww)) (list keep y))
                              (swap))
                             (t (assert (= i o)) (ew "x_add" (* np o) '() (list x keep))))))
            (let* ((top (second (car (last +vae-enc-stages+)))) (np (* hh ww)))
              (resblock "encoder.middle.0" x y top top hh ww) (swap)
              (attention "encoder.middle.1" x y top np) (swap)
              (resblock "encoder.middle.2" x y top top hh ww) (swap)
              ;; the head: norm, SiLU, 3 x 3 to the mean and log-variance, CONV1
              (norm-silu "encoder.head.0.gamma" x y top np)
              (conv3 "encoder.head.2" y x top 128 hh ww)
              (conv1 "conv1" x y 128 128 np)
              (let ((rows (get-floats y (* np 128))) (z (f32vec (* 64 np))))
                (dotimes (c 64) (dotimes (i np) (setf (aref z (+ (* c np) i)) (aref rows (+ (* i 128) c)))))
                z))))))))
