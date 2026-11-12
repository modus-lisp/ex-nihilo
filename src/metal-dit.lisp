;;;; metal-dit.lisp — the DiT on Apple silicon: cuda-dit.lisp's steps on
;;;; Metal (docs/design.md, section 2).  The GGUF is mapped and wrapped as one
;;;; buffer, its Q8_0 matrices read where they lie by illatio's d4_mm_q8_0
;;;; (weights and inputs to halves, half products summed in floats); its few
;;;; other matrices (img_in, txt_in, norm_out) go to the GPU as floats and
;;;; are multiplied in floats (x_conv as a product): txt_in's inner rows
;;;; reach 8e5, past a half's range.  The attention's Q, K
;;;; and V are halves (src/metal/exn.metal's x_attend_h).  A layer is a
;;;; command buffer: a step's work in one would run long enough to trip the
;;;; system's watchdog on a busy display.

(in-package #:ex-nihilo)

;;; --- the device, with our kernels beside illatio's ------------------------------

(defvar *gpu* nil "The Metal device, its library holding illatio's kernels and ours.")

(defun exn-gpu ()
  (or *gpu*
      (setf *gpu*
            (il::open-gpu
             (with-output-to-string (o)
               (write-string (il::metal-source) o)
               (dolist (f (sort (directory (merge-pathnames "metal/*.metal" (asdf:system-relative-pathname "ex-nihilo" "src/")))
                                #'string< :key #'namestring))
                 (write-string (uiop:read-file-string f) o)))))))

(defun put-floats (buf v &optional (byte-offset 0))
  "V's floats into shared buffer BUF from BYTE-OFFSET."
  (declare (type f32vec v))
  (sb-sys:with-pinned-objects (v)
    (sb-alien:alien-funcall
     (sb-alien:extern-alien "memcpy" (function sb-sys:system-area-pointer sb-sys:system-area-pointer
                                               sb-sys:system-area-pointer sb-alien:unsigned-long))
     (sb-sys:sap+ (il::gbuf-sap buf) byte-offset) (sb-sys:vector-sap v) (* 4 (length v))))
  buf)

(defun get-floats (buf n &optional (byte-offset 0))
  "N floats of shared buffer BUF from BYTE-OFFSET."
  (let ((v (f32vec n)))
    (sb-sys:with-pinned-objects (v)
      (sb-alien:alien-funcall
       (sb-alien:extern-alien "memcpy" (function sb-sys:system-area-pointer sb-sys:system-area-pointer
                                                 sb-sys:system-area-pointer sb-alien:unsigned-long))
       (sb-sys:vector-sap v) (sb-sys:sap+ (il::gbuf-sap buf) byte-offset) (* 4 n)))
    v))

(defun half-buffer (gpu v)
  "Floats V on the GPU as halves."
  (let ((b (il::gpu-buffer gpu (* 2 (length v)))))
    (dotimes (i (length v) b)
      (setf (sb-sys:sap-ref-16 (il::gbuf-sap b) (* 2 i)) (il::single-to-half (aref v i))))))

;;; --- weights --------------------------------------------------------------------

(defstruct (mdit (:constructor %make-mdit))
  gpu gguf file
  (w (make-hash-table :test 'equal))   ; name -> (kernel buf offset row-bytes rows cols); kernel NIL: floats
  (vec (make-hash-table :test 'equal)) ; name -> buffer of floats
  (buf (make-hash-table))              ; scratch by key: a buffer
  ;; the prompt's keys and values, a layer each: (k . v) buffers of P x 4096 halves
  prefix (prefix-length 0) prefix-key)

(defun upload-dit (path &key device)
  "The DiT in the GGUF at PATH on the GPU (DEVICE: ignored, there is one)."
  (declare (ignore device))
  (let* ((g (open-dit path)) (gpu (exn-gpu)) (t0 (get-internal-real-time)) (floats 0)
         (d (%make-mdit :gpu gpu :gguf g)))
    (multiple-value-bind (sap bytes) (il::mmap-file path)
      (setf (mdit-file d) (il::gpu-wrap gpu sap bytes)))
    (dolist (name (gguf-tensor-order g))
      (let* ((ti (gguf-tensor g name)) (short (subseq name (length +dit-prefix+)))
             (dims (tensor-info-dims ti)) (cols (first dims)) (rows (reduce #'* (rest dims))))
        (cond ((null (rest dims))   ; a vector: the norms' weights, as floats
               (let ((v (il::load-f32 g name)))
                 (when (string= short "txt_in.text_norm.weight") ; zero-centred: used as 1 + w
                   (map-into v (lambda (x) (+ 1f0 x)) v))
                 (setf (gethash short (mdit-vec d))
                       (put-floats (il::gpu-buffer gpu (* 4 (length v))) v))))
              ((eq (tensor-info-type ti) :q8_0)
               (setf (gethash short (mdit-w d))
                     (list "d4_mm_q8_0" (mdit-file d) (+ (il::gguf-data-offset g) (il::tensor-info-offset ti))
                           (* 34 (floor cols 32)) rows cols)))
              (t                    ; bf16, f32: to floats
               (let ((v (il::load-f32 g name)))
                 (incf floats (length v))
                 (setf (gethash short (mdit-w d))
                       (list nil (put-floats (il::gpu-buffer gpu (* 4 (length v))) v) 0 (* 4 cols) rows cols)))))))
    (format *error-output* "ex-nihilo: the DiT (~,1F GB, ~,1FM weights as floats) on ~A, ~,1Fs~%"
            (/ (il::gbuf-bytes (mdit-file d)) 1d9) (/ floats 1d6) (il::gpu-name gpu)
            (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
    d))

;;; --- a step's pieces --------------------------------------------------------------

(defun scratch (d key bytes)
  "D's scratch buffer KEY, at least BYTES long."
  (let ((b (gethash key (mdit-buf d))))
    (if (and b (>= (il::gbuf-bytes b) bytes))
        b
        (progn (when b (il::gpu-free b))
               (setf (gethash key (mdit-buf d)) (il::gpu-buffer (mdit-gpu d) bytes))))))

(defun dit-release (d)
  "D's scratch and kept prompt keys and values given back."
  (maphash (lambda (k b) (declare (ignore k)) (il::gpu-free b)) (mdit-buf d))
  (clrhash (mdit-buf d))
  (when (mdit-prefix d)
    (loop for e across (mdit-prefix d) when e do (il::gpu-free (car e)) (il::gpu-free (cdr e))))
  (setf (mdit-prefix d) nil (mdit-prefix-key d) nil (mdit-prefix-length d) 0))

(defvar *dit-layers* 32 "The layers a step runs: fewer, to look at a step's state part way.")

(defvar *dit-trace* nil
  "When true, each layer's largest values (the stream, the attention's
   output, the MLP's inner rows) are printed after it: where a step first
   overflows.")

(defun trace-max (name buf floats)
  (let ((mx 0f0) (bad 0))
    (dotimes (i floats)
      (let ((x (sb-sys:sap-ref-single (il::gbuf-sap buf) (* 4 i))))
        (if (or (/= x x) (> (abs x) most-positive-single-float)) (incf bad) (setf mx (max mx (abs x))))))
    (format *error-output* " ~A ~,1F~:[~; (~D not finite)~]" name mx (plusp bad) bad)))

(defun at (buf byte-offset) (cons buf byte-offset))

(defun mm (d e name x y n)
  "Y (N rows) = W X (N rows) for D's matrix NAME: illatio's D4_MM, or x_conv
   as a product in floats."
  (destructuring-bind (kernel w off rbytes rows cols) (or (gethash name (mdit-w d)) (error "no matrix ~A" name))
    (multiple-value-bind (xb xo) (if (consp x) (values (car x) (cdr x)) (values x 0))
      (multiple-value-bind (yb yo) (if (consp y) (values (car y) (cdr y)) (values y 0))
        (if kernel
            (il::dispatch (mdit-gpu d) e kernel (list cols rows n rbytes cols rows)
                          (list (at w off) (at xb xo) (at yb yo))
                          (list (ceiling n 64) (ceiling rows 64) 1) 128 :shared 16384)
            (il::dispatch (mdit-gpu d) e "x_conv" (list cols rows 1 n 1 1 cols rows 0)
                          (list (at w off) (at w off) (at xb xo) (at yb yo))
                          (list (ceiling n 64) (ceiling rows 64)) 256))))))

(defun elementwise (d e kernel count args buffers)
  (il::dispatch (mdit-gpu d) e kernel (list* count args) buffers (ceiling count 256) 256))

(defun copy-words (d e src dst words)
  (il::dispatch (mdit-gpu d) e "x_copy" (list words) (list src dst) (ceiling words 256) 256))

;;; --- a step -------------------------------------------------------------------------

(defun dit-velocity (d context p latent h w sigma &key (prefix-key nil) ref)
  "The DiT's velocity at noise level SIGMA: LATENT (64 x H x W floats,
   channel-major) in, the same shape out.  As cuda-dit.lisp's: CONTEXT the
   encoder's P x 4096 states; PREFIX-KEY keeps the prompt's keys and values;
   REF, (A LATENT RH RW), a picture to edit in the prefix."
  (let* ((gpu (mdit-gpu d)) (nimg (* h w))
         (ptext p) (nref (if ref (* (third ref) (fourth ref)) 0)) (a (if ref (first ref) 0))
         (p (+ p nref))
         (cached (and prefix-key (equal prefix-key (mdit-prefix-key d)) (= p (mdit-prefix-length d))))
         (n0 (if cached p 0))              ; the first row this pass computes
         (n (if cached nimg (+ p nimg)))   ; rows it computes
         (l (+ p nimg))                    ; keys every image row sees
         (wd 4096) (ffn 12288))
    (flet ((b (key floats) (scratch d key (* 4 floats)))
           (bh (key halves) (scratch d key (* 2 halves)))
           (vec (name) (or (gethash name (mdit-vec d)) (error "no vector ~A" name))))
      (let ((emb (b :emb (* 2 256))) (e1 (b :e1 (* 2 wd))) (e2 (b :e2 (* 2 wd)))
            (mod (b :mod (* 2 4 wd))) (sc (b :sc wd))
            (x (b :x (* l wd))) (y (b :y (* l wd))) (qf (b :q (* n wd))) (kf (b :k (* n wd)))
            (vf (b :v (* n wd))) (qh (bh :qh (* n wd))) (kh (bh :kh (* l wd))) (vh (bh :vh (* l wd)))
            (att (b :att (* n wd))) (hh (b :h (* l wd)))
            (gu (b :gu (* l 2 ffn))) (ff (b :ff (* l ffn))) (cs (b :cs (* l 128)))
            (img (b :img (* nimg 64))))
        ;; the host's part: the timestep, the context, the latent as rows, the rotary table
        (put-floats emb (concatenate 'f32vec (timestep-embedding (* 1000d0 sigma)) (timestep-embedding 0d0)))
        (let ((tokens (f32vec (* nimg 64))))  ; channel-major to a row of 64 a pixel
          (dotimes (c 64) (dotimes (i nimg) (setf (aref tokens (+ (* i 64) c)) (aref latent (+ (* c nimg) i)))))
          (put-floats img tokens))
        (put-floats cs (rope-table (nthcdr n0 (dit-positions p h w (and ref (list a (third ref) (fourth ref)))))))
        (unless cached
          (put-floats (b :ctx (* ptext wd)) context))
        (il::with-gpu-work (e gpu)
          ;; the timestep: t and 0 through the embedder, SiLU, the shared modulation
          (mm d e "time_text_embed.timestep_embedder.linear_1.weight" emb e1 2)
          (elementwise d e "x_silu" (* 2 wd) '() (list e1))
          (mm d e "time_text_embed.timestep_embedder.linear_2.weight" e1 e2 2)
          (elementwise d e "x_silu" (* 2 wd) '() (list e2))
          (mm d e "modulation.1.weight" e2 mod 2)
          (mm d e "norm_out.linear.weight" e2 sc 1)
          ;; the joint sequence: the prompt through txt_in (unless cached), the image through img_in
          (unless cached
            (let ((c (b :ctx (* ptext wd))) (t1 (b :t1 (* ptext wd))))
              (il::dispatch gpu e "x_rms" (list wd 1f-6) (list c (vec "txt_in.text_norm.weight") t1) ptext 256)
              (mm d e "txt_in.in_layer.weight" t1 hh ptext)
              (elementwise d e "x_gelu_tanh" (* ptext wd) '() (list hh))
              (cond ((null ref) (mm d e "txt_in.out_layer.weight" hh x ptext))
                    (t (error "editing a picture is not on Metal yet")))))
          (mm d e "img_in.weight" img (at x (* 4 (- p n0) wd)) nimg))
        (when *dit-trace*
          (format *error-output* "~&before:")
          (trace-max "img" img (* nimg 64)) (trace-max "e2" e2 (* 2 wd)) (trace-max "mod" mod (* 8 wd))
          (unless cached
            (trace-max "ctx" (b :ctx (* ptext wd)) (* ptext wd)) (trace-max "t1" (b :t1 (* ptext wd)) (* ptext wd))
            (trace-max "h" hh (* ptext wd)))
          (trace-max "x-text" x (* (- p n0) wd))
          (trace-max "x-all" x (* l wd))
          (terpri *error-output*))
        (unless cached
          (setf (mdit-prefix d) (or (mdit-prefix d) (make-array 32 :initial-element nil))
                (mdit-prefix-key d) nil))
        (dotimes (il *dit-layers*)
          (flet ((blk (s) (format nil "transformer_blocks.~D.~A" il s))
                 (piece (i) (at mod (* 4 i wd))))   ; a row of MOD: four pieces of WD floats
            (let ((pre (aref (mdit-prefix d) il)))
              (when (and prefix-key (not cached)
                         (not (and pre (>= (il::gbuf-bytes (car pre)) (* 2 p wd)))))
                (when pre (il::gpu-free (car pre)) (il::gpu-free (cdr pre)))
                (setf pre (setf (aref (mdit-prefix d) il)
                                (cons (il::gpu-buffer gpu (* 2 p wd)) (il::gpu-buffer gpu (* 2 p wd))))))
              (il::with-gpu-work (e gpu)
                (il::dispatch gpu e "x_ln_mod" (list wd n n0 p (* 4 wd)) (list x (piece 0) y) n 256)
                ;; this pass's keys and values go after the prompt's, as halves
                (mm d e (blk "attn.to_q.weight") y qf n)
                (mm d e (blk "attn.to_k.weight") y kf n)
                (mm d e (blk "attn.to_v.weight") y vf n)
                (il::dispatch gpu e "x_qk_norm_rope_h" (list n) (list (vec (blk "attn.norm_q.weight")) cs qf qh)
                              (list n 32) 64)
                (il::dispatch gpu e "x_qk_norm_rope_h" (list n) (list (vec (blk "attn.norm_k.weight")) cs kf
                                                                     (at kh (* 2 n0 wd)))
                              (list n 32) 64)
                (elementwise d e "x_to_half" (* n wd) '() (list vf (at vh (* 2 n0 wd))))
                (cond (cached
                       (copy-words d e (car pre) kh (* p (floor wd 2)))
                       (copy-words d e (cdr pre) vh (* p (floor wd 2))))
                      (prefix-key
                       (copy-words d e kh (car pre) (* p (floor wd 2)))
                       (copy-words d e vh (cdr pre) (* p (floor wd 2)))))
                (il::dispatch gpu e "x_attend_h" (list n n0 p l a (+ a nref) (coerce (/ (sqrt 128d0)) 'single-float))
                              (list qh kh vh att) (list 32 (ceiling n 32)) 128 :shared 32768)
                (mm d e (blk "attn.to_out.0.weight") att hh n)
                (elementwise d e "x_gate_add" (* n wd) (list wd n0 p (* 4 wd)) (list x hh (piece 1)))
                (il::dispatch gpu e "x_ln_mod" (list wd n n0 p (* 4 wd)) (list x (piece 2) y) n 256)
                (mm d e (blk "img_mlp.gate_up.weight") y gu n)
                (elementwise d e "x_swiglu2" (* n ffn) (list ffn) (list gu ff))
                (mm d e (blk "img_mlp.out.weight") ff hh n)
                (elementwise d e "x_gate_add" (* n wd) (list wd n0 p (* 4 wd)) (list x hh (piece 3))))
              (when *dit-trace*
                (format *error-output* "~&layer ~D:" il)
                (trace-max "x" x (* l wd)) (trace-max "att" att (* n wd)) (trace-max "gu" gu (* n 2 ffn))
                (trace-max "ff" ff (* n ffn)) (terpri *error-output*)))))
        (when (and prefix-key (not cached))
          (setf (mdit-prefix-key d) prefix-key (mdit-prefix-length d) p))
        ;; the image's rows: LayerNorm * (1 + scale(t)), proj_out
        (let ((xi (at x (* 4 (- p n0) wd))) (out (b :out (* nimg 64))))
          (il::with-gpu-work (e gpu)
            (il::dispatch gpu e "x_ln_mod" (list wd nimg 0 0 0) (list xi sc y) nimg 256)
            (mm d e "proj_out.weight" y out nimg))
          (let ((rows (get-floats out (* nimg 64))) (v (f32vec (* nimg 64))))
            (dotimes (c 64) (dotimes (i nimg) (setf (aref v (+ (* c nimg) i)) (aref rows (+ (* i 64) c)))))
            v))))))
