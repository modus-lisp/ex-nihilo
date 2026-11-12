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

(defparameter +qv-width+ 1152)
(defparameter +qv-heads+ 16)
(defparameter +qv-head-dim+ 72)
(defparameter +qv-head-pad+ 80 "A head's rows for the products, 16 floats at a time: 72 and zeros.")
(defparameter +qv-ffn+ 4304)
(defparameter +qv-depth+ 27)
(defparameter +qv-patch+ 16)
(defparameter +qv-out+ 4096)
(defparameter +qv-deepstack+ '(8 16 24))

(defstruct (q3vision (:constructor %make-q3vision))
  dev
  (w (make-hash-table :test 'equal))   ; name -> DBUF of floats (a matrix row-major, out x in)
  pos side                             ; the learned position table (host floats), its side
  (buf (make-hash-table)))             ; scratch: key -> (dbuf . floats)

(defun load-vision (path &key (device 1))
  "The vision tower in the mmproj GGUF at PATH on CUDA device DEVICE."
  (let* ((g (read-gguf path)) (dev (exn-device device)) (v (%make-q3vision :dev dev)) (bytes 0)
         (t0 (get-internal-real-time)))
    (unless (equal (meta g "clip.projector_type") "qwen3vl_merger")
      (error "~A: not Qwen3-VL's vision tower (projector ~A)" path (meta g "clip.projector_type")))
    (il::with-cuda
      (il::cuda-current dev)
      (flet ((up (name f)
               (incf bytes (* 4 (length f)))
               (setf (gethash name (q3vision-w v)) (il::cuda-upload-vector (il::cuda-alloc dev (* 4 (length f)) :zero nil) f))))
        (dolist (name (gguf-tensor-order g))
          (cond ((string= name "v.position_embd.weight")
                 (setf (q3vision-pos v) (il::load-f32 g name)
                       (q3vision-side v) (isqrt (floor (length (q3vision-pos v)) +qv-width+))))
                ((string= name "v.patch_embd.weight.1"))   ; summed into the first (one frame, twice)
                ((string= name "v.patch_embd.weight")
                 (let ((a (il::load-f32 g name)) (b (il::load-f32 g "v.patch_embd.weight.1")))
                   (dotimes (i (length a)) (incf (aref a i) (aref b i)))
                   (up name a)))
                (t (up name (il::load-f32 g name)))))
        (il::cuda-sync dev)))
    (format *error-output* "ex-nihilo: the vision tower (~,2F GB as floats) on ~A, ~,1Fs~%" (/ bytes 1d9)
            (il::cudev-name dev) (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
    v))

;;; --- the host's part: patches, positions, the rotary's angles ---------------------------

(defun merge-order (gh gw)
  "The patches' (row . col) in merge-block order: block row, block column, then the 2 x 2 inside."
  (let ((out '()))
    (dotimes (br (floor gh 2))
      (dotimes (bc (floor gw 2))
        (dotimes (ir 2) (dotimes (ic 2) (push (cons (+ (* 2 br) ir) (+ (* 2 bc) ic)) out)))))
    (nreverse out)))

(defun vision-patches (px h w order)
  "PX (H x W RGBA floats, -1 .. 1) as N rows of 768: each patch's (channel, row, column), in ORDER."
  (declare (ignore h))
  (let ((out (f32vec (* (length order) 768))) (p +qv-patch+))
    (loop for (r . c) in order for k from 0
          do (dotimes (ch 3)
               (dotimes (ky p)
                 (dotimes (kx p)
                   (setf (aref out (+ (* k 768) (* ch 256) (* ky 16) kx))
                         (aref px (+ (* 4 (+ (* (+ (* r p) ky) w) (* c p) kx)) ch)))))))
    out))

(defun vision-pos-embed (v order gh gw)
  "Each patch's learned position: the table resampled bilinearly to GH x GW, corners aligned."
  (let* ((side (q3vision-side v)) (tab (q3vision-pos v)) (d +qv-width+)
         (out (f32vec (* (length order) d))))
    (flet ((taps (i size)
             (let* ((src (/ (* i (- side 1d0)) (max 1 (- size 1)))) (f (floor src)) (fr (- src f)))
               (list (cons (min f (1- side)) (- 1 fr)) (cons (min (1+ f) (1- side)) fr)))))
      (loop for (r . c) in order for k from 0
            do (loop for (hi . hw) in (taps r gh)
                     do (loop for (wi . ww) in (taps c gw)
                              do (let ((wt (coerce (* hw ww) 'single-float)) (base (* d (+ (* hi side) wi))))
                                   (unless (zerop wt)
                                     (dotimes (j d)
                                       (incf (aref out (+ (* k d) j)) (* wt (aref tab (+ base j)))))))))))
    out))

(defun vision-rotary (order)
  "Each patch's 72 cosines then 72 sines: 18 frequencies of its row, 18 of its column, those 36 twice."
  (let* ((hd +qv-head-dim+) (nf (floor hd 4)) (out (f32vec (* (length order) 2 hd))))
    (loop for (r . c) in order for k from 0
          do (dotimes (j (floor hd 2))
               (let* ((i (mod j nf)) (pos (if (< j nf) r c))
                      (a (* pos (/ 1d0 (expt 10000d0 (/ (* 2 i) (/ hd 2))))))
                      (cs (coerce (cos a) 'single-float)) (sn (coerce (sin a) 'single-float)))
                 (dolist (jj (list j (+ j (floor hd 2))))
                   (setf (aref out (+ (* k 2 hd) jj)) cs (aref out (+ (* k 2 hd) hd jj)) sn)))))
    out))

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
