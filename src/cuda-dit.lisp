;;;; cuda-dit.lisp — the DiT on an NVIDIA GPU: its weights uploaded, one
;;;; step's velocity computed (docs/design.md, section 2).  Kernels: illatio's
;;;; products (d4_mm_q8_0 on the tensor cores, g53_mm_bf16, d41_mm_f32), v_rms,
;;;; and src/cuda/dit.cu.
;;;;
;;;; The prompt's rows never see the image's and are modulated at t = 0, so
;;;; their keys and values are the same at every step: the first step runs
;;;; the joint sequence and keeps each layer's prompt K and V; the later
;;;; steps run only the image's rows against them.

(in-package #:ex-nihilo)

;;; --- the device, with our kernels beside illatio's ------------------------------

(defvar *devices* (make-hash-table) "index -> a CUDEV whose module holds illatio's kernels and ours")

(defun exn-source ()
  (with-output-to-string (o)
    (write-string (il::cuda-source) o)
    (dolist (f (sort (directory (merge-pathnames "cuda/*.cu" (asdf:system-relative-pathname "ex-nihilo" "src/")))
                     #'string< :key #'namestring))
      (with-open-file (s f)
        (loop for line = (read-line s nil) while line do (write-line line o))))))

(defun exn-device (&optional (index 0))
  "CUDA device INDEX as illatio opens it, with a module of its kernels and ex-nihilo's."
  (or (gethash index *devices*)
      (setf (gethash index *devices*)
            (let* ((base (il::open-cuda index)) (dev (copy-structure base)))
              (il::with-cuda
                (il::cuda-current base)
                (let ((cubin (il::cuda-cubin (exn-source) (il::cudev-arch base))))
                  (il::with-out (module)
                    (sb-sys:with-pinned-objects (cubin)
                      (il::cu "cuModuleLoadData" (sb-sys:system-area-pointer (il::out-ptr module))
                              (sb-sys:system-area-pointer (sb-sys:vector-sap cubin))))
                    (setf (il::cudev-module dev) (sb-sys:int-sap module)
                          (il::cudev-functions dev) (make-hash-table :test 'equal)
                          (il::cudev-shared-set dev) (make-hash-table :test 'equal)))))
              dev))))

;;; --- weights --------------------------------------------------------------------

(defstruct (gdit (:constructor %make-gdit))
  dev gguf
  (w (make-hash-table :test 'equal))   ; name -> (:q8 qs d rows cols) | (:bf16 buf rows cols) | (:f32 buf rows cols)
  (vec (make-hash-table :test 'equal)) ; name -> DBUF of floats
  (buf (make-hash-table))              ; scratch by key: (dbuf . floats)
  ;; the prompt's keys and values, a layer each: (k . v) DBUFs of P x 4096
  prefix (prefix-length 0) prefix-key)

(defun upload-dit (path &key (device 0))
  "The DiT in the GGUF at PATH on CUDA device DEVICE."
  (let* ((g (open-dit path)) (dev (exn-device device)) (d (%make-gdit :dev dev :gguf g))
         (file (il::map-file-vector path)) (t0 (get-internal-real-time)) (bytes 0))
    (il::with-cuda
      (il::cuda-current dev)
      (let ((raw (il::cuda-alloc dev (* 4096 24576 34/32) :zero nil)))
        (dolist (name (gguf-tensor-order g))
          (let* ((ti (gguf-tensor g name)) (short (subseq name (length +dit-prefix+)))
                 (dims (tensor-info-dims ti)) (cols (first dims)) (rows (reduce #'* (rest dims)))
                 (src (sb-sys:sap+ (sb-sys:vector-sap file) (+ (il::gguf-data-offset g) (il::tensor-info-offset ti))))
                 (nb (tensor-bytes ti)))
            (incf bytes nb)
            (cond ((null (rest dims))   ; a vector: the norms' weights, as floats
                   (let ((v (il::load-f32 g name)))
                     (when (string= short "txt_in.text_norm.weight") ; zero-centred: used as 1 + w
                       (map-into v (lambda (x) (+ 1f0 x)) v))
                     (setf (gethash short (gdit-vec d))
                           (il::cuda-upload-vector (il::cuda-alloc dev (* 4 (length v)) :zero nil) v))))
                  ((eq (tensor-info-type ti) :q8_0)
                   (let* ((n (* rows cols)) (qs (il::cuda-alloc dev n :zero nil))
                          (sc (il::cuda-alloc dev (* 2 (floor n 32)) :zero nil)))
                     (il::cuda-upload raw src nb)
                     (il::launch dev "repack_q8" (ceiling (floor n 32) 8) '(32 8) (list raw qs sc (list :i64 (floor n 32))))
                     (setf (gethash short (gdit-w d)) (list :q8 qs sc rows cols))))
                  ((member (tensor-info-type ti) '(:bf16 :f32))
                   (let ((b (il::cuda-alloc dev nb :zero nil)))
                     (il::cuda-upload b src nb)
                     (setf (gethash short (gdit-w d)) (list (tensor-info-type ti) b rows cols))))
                  (t (error "~A: a ~A matrix" name (tensor-info-type ti))))))
        (il::cuda-sync dev)
        (il::cuda-free raw)))
    (format *error-output* "ex-nihilo: the DiT (~,1F GB) on ~A, ~,1Fs~%" (/ bytes 1d9) (il::cudev-name dev)
            (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
    d))

;;; --- a step's pieces --------------------------------------------------------------

(defun scratch (d key floats)
  "D's scratch buffer KEY, at least FLOATS long."
  (let ((e (gethash key (gdit-buf d))))
    (if (and e (>= (cdr e) floats))
        (car e)
        (progn (when e (il::cuda-free (car e)))
               (car (setf (gethash key (gdit-buf d))
                          (cons (il::cuda-alloc (gdit-dev d) (* 4 floats) :zero nil) floats)))))))

(defun dit-release (d)
  "D's scratch and kept prompt keys and values freed: a picture's steps
   done, the decoder wants the room (an edit's sequence is twice a
   picture's, and the scratch it leaves would be parked and brought back
   with the weights every time)."
  (maphash (lambda (k e) (declare (ignore k)) (il::cuda-free (car e))) (gdit-buf d))
  (clrhash (gdit-buf d))
  (when (gdit-prefix d)
    (loop for e across (gdit-prefix d) when e do (il::cuda-free (car e)) (il::cuda-free (cdr e))))
  (setf (gdit-prefix d) nil (gdit-prefix-key d) nil (gdit-prefix-length d) 0))

(defvar *int8* t
  "Whether many rows through a Q8_0 matrix go as int8 activations (illatio's
   MM_Q8_MMA2: X quantized to Q8_0 by blocks of 32, the int8 tensor cores --
   ggml's way for a batch) rather than as halves (D4_MM_Q8_0).")

(defun mm (d name x y n)
  "Y (N rows) = W X (N rows) for D's matrix NAME, whatever its format."
  (destructuring-bind (kind a b &optional c e) (or (gethash name (gdit-w d)) (error "no matrix ~A" name))
    (let ((dev (gdit-dev d)))
      (ecase kind
        (:q8 (if (and *int8* (>= n 16) (il::mm2-p dev e))
                 (let* ((blocks (floor (* n e) 32))
                        (xq (scratch d :xq (ceiling (* n e) 4))) (xs (scratch d :xs blocks)))
                   (il::launch dev "quant_q8" (ceiling blocks 8) 256 (list x xq xs nil (list :i64 blocks)))
                   (il::cmat-mul dev (il::%make-cmat :name name :type :q8_0 :rows c :cols e :qs a :d b) xq xs y n))
                 (il::launch dev "d4_mm_q8_0" (list (ceiling n 64) (ceiling c 64)) 128
                             (list e c n 0 e c a b x y))))
        (:bf16 (il::launch dev "g53_mm_bf16" (list (ceiling n 64) (ceiling b 64)) 256
                           (list c b n c b a x y)))
        (:f32 (il::launch dev "d41_mm_f32" (list (ceiling n 64) (ceiling b 64)) 256
                          (list c b n c b a x y)))))))

(defun elementwise (d kernel count &rest args)
  (il::launch (gdit-dev d) kernel (ceiling count 256) 256 (list* (list :i64 count) args)))

;;; --- a step -------------------------------------------------------------------------

(defun dit-velocity (d context p latent h w sigma &key (prefix-key nil) ref)
  "The DiT's velocity at noise level SIGMA: LATENT (64 x H x W floats,
   channel-major) in, the same shape out.  CONTEXT: the encoder's P x 4096
   states.  With PREFIX-KEY, the prompt's keys and values are kept under it
   after the first call and reused by the next ones with the same key.
   REF, a picture to edit: (A LATENT RH RW), its normalized latent (64 x RH x
   RW) put in the prefix after the encoder's first A states, in place of its
   image slots (those states left out of CONTEXT); P counts the text only."
  (let* ((dev (gdit-dev d)) (nimg (* h w))
         (ptext p) (nref (if ref (* (third ref) (fourth ref)) 0)) (a (if ref (first ref) 0))
         (p (+ p nref))                    ; the prefix: text and the reference picture
         (cached (and prefix-key (equal prefix-key (gdit-prefix-key d)) (= p (gdit-prefix-length d))))
         (n0 (if cached p 0))              ; the first row this pass computes
         (n (if cached nimg (+ p nimg)))   ; rows it computes
         (l (+ p nimg))                    ; keys every image row sees
         (wd 4096) (ffn 12288))
    (il::with-cuda
      (il::cuda-current dev)
      (flet ((b (key floats) (scratch d key floats))
             (vec (name) (or (gethash name (gdit-vec d)) (error "no vector ~A" name))))
        ;; the timestep: t and 0 through the embedder, SiLU, the shared modulation
        (let ((emb (b :emb (* 2 256))) (e1 (b :e1 (* 2 wd))) (e2 (b :e2 (* 2 wd)))
              (mod (b :mod (* 2 4 wd))) (sc (b :sc wd)))
          (il::cuda-upload-vector emb (concatenate 'il::f32vec (timestep-embedding (* 1000d0 sigma))
                                                   (timestep-embedding 0d0)))
          (mm d "time_text_embed.timestep_embedder.linear_1.weight" emb e1 2)
          (elementwise d "x_silu" (* 2 wd) e1)
          (mm d "time_text_embed.timestep_embedder.linear_2.weight" e1 e2 2)
          (elementwise d "x_silu" (* 2 wd) e2)
          (mm d "modulation.1.weight" e2 mod 2)
          (mm d "norm_out.linear.weight" e2 sc 1)
          ;; the joint sequence: the prompt through txt_in (unless cached), the image through img_in
          (let ((x (b :x (* l wd))) (y (b :y (* l wd))) (q (b :q (* l wd))) (kb (b :k (* l wd)))
                (vb (b :v (* l wd))) (att (b :att (* l wd))) (hh (b :h (* l wd)))
                (gu (b :gu (* l 2 ffn))) (ff (b :ff (* l ffn))) (cs (b :cs (* l 128)))
                (img (b :img (* nimg 64))))
            (unless cached
              (let ((c (b :ctx (* ptext wd))) (t1 (b :t1 (* ptext wd))))
                (il::cuda-upload-vector c context)
                (il::launch dev "v_rms" ptext 256 (list wd 1f-6 c (vec "txt_in.text_norm.weight") t1))
                (mm d "txt_in.in_layer.weight" t1 hh ptext)
                (elementwise d "x_gelu_tanh" (* ptext wd) hh)
                (cond ((null ref) (mm d "txt_in.out_layer.weight" hh x ptext))
                      (t ;; text before the picture, the picture through img_in, the text after
                       (mm d "txt_in.out_layer.weight" hh c ptext)
                       (il::cuda-copy-dd x 0 c 0 (* 4 a wd))
                       (il::cuda-copy-dd x (* 4 (+ a nref) wd) c (* 4 a wd) (* 4 (- ptext a) wd))
                       (let ((rows (f32vec (* nref 64))) (rl (second ref)) (ri (b :ri (* nref 64))))
                         (dotimes (ch 64) (dotimes (i nref) (setf (aref rows (+ (* i 64) ch)) (aref rl (+ (* ch nref) i)))))
                         (il::cuda-upload-vector ri rows)
                         (mm d "img_in.weight" ri (cons x (* 4 a wd)) nref))))))
            (let ((tokens (f32vec (* nimg 64))))  ; channel-major to a row of 64 a pixel
              (dotimes (c 64) (dotimes (i nimg) (setf (aref tokens (+ (* i 64) c)) (aref latent (+ (* c nimg) i)))))
              (il::cuda-upload-vector img tokens))
            (mm d "img_in.weight" img (cons x (* 4 (- p n0) wd)) nimg)
            (il::cuda-upload-vector cs (rope-table (nthcdr n0 (dit-positions p h w (and ref (list a (third ref) (fourth ref)))))))
            (unless cached
              (setf (gdit-prefix d) (or (gdit-prefix d) (make-array 32 :initial-element nil))
                    (gdit-prefix-key d) nil))
            (dotimes (il 32)
              (flet ((blk (s) (format nil "transformer_blocks.~D.~A" il s))
                     (piece (i) (cons mod (* 4 i wd))))
                (il::launch dev "x_ln_mod" n 256 (list wd n n0 p (* 4 wd) x (piece 0) y))
                ;; this pass's keys and values go after the prompt's
                (let ((kr (cons kb (* 4 n0 wd))) (vr (cons vb (* 4 n0 wd))))
                  (mm d (blk "attn.to_q.weight") y q n)
                  (mm d (blk "attn.to_k.weight") y kr n)
                  (mm d (blk "attn.to_v.weight") y vr n)
                  (il::launch dev "x_qk_norm_rope" (list n 32) 64 (list n (vec (blk "attn.norm_q.weight")) cs q))
                  (il::launch dev "x_qk_norm_rope" (list n 32) 64 (list n (vec (blk "attn.norm_k.weight")) cs kr)))
                (let ((pre (aref (gdit-prefix d) il)))
                  (cond (cached
                         (il::cuda-copy-dd kb 0 (car pre) 0 (* 4 p wd))
                         (il::cuda-copy-dd vb 0 (cdr pre) 0 (* 4 p wd)))
                        (prefix-key
                         (unless (and pre (>= (il::dbuf-bytes (car pre)) (* 4 p wd)))
                           (when pre (il::cuda-free (car pre)) (il::cuda-free (cdr pre)))
                           (setf pre (setf (aref (gdit-prefix d) il)
                                           (cons (il::cuda-alloc dev (* 4 p wd) :zero nil)
                                                 (il::cuda-alloc dev (* 4 p wd) :zero nil)))))
                         (il::cuda-copy-dd (car pre) 0 kb 0 (* 4 p wd))
                         (il::cuda-copy-dd (cdr pre) 0 vb 0 (* 4 p wd)))))
                (if (>= (il::cudev-arch dev) 80)
                    (il::launch dev "x_attend_mma" (list 32 (ceiling n 128)) 256
                                (list q kb vb att n n0 p l a (+ a nref) (coerce (/ (sqrt 128d0)) 'single-float)))
                    (il::launch dev "x_attend" (list 32 (ceiling n 64)) 256
                                (list q kb vb att 32 128 n n0 p l a (+ a nref) (coerce (/ (sqrt 128d0)) 'single-float))
                                :shared (* 4 (+ (* 64 132) (* 32 132) (* 32 128) (* 32 68)))))
                (mm d (blk "attn.to_out.0.weight") att hh n)
                (elementwise d "x_gate_add" (* n wd) wd n0 p (* 4 wd) x hh (piece 1))
                (il::launch dev "x_ln_mod" n 256 (list wd n n0 p (* 4 wd) x (piece 2) y))
                (mm d (blk "img_mlp.gate_up.weight") y gu n)
                (elementwise d "x_swiglu2" (* n ffn) ffn gu ff)
                (mm d (blk "img_mlp.out.weight") ff hh n)
                (elementwise d "x_gate_add" (* n wd) wd n0 p (* 4 wd) x hh (piece 3))))
            (when (and prefix-key (not cached))
              (setf (gdit-prefix-key d) prefix-key (gdit-prefix-length d) p))
            ;; the image's rows: LayerNorm * (1 + scale(t)), proj_out
            (let ((xi (cons x (* 4 (- p n0) wd))) (out (b :out (* nimg 64))))
              (il::launch dev "x_ln_mod" nimg 256 (list wd nimg 0 0 0 xi sc y))
              (mm d "proj_out.weight" y out nimg)
              (let ((rows (f32vec (* nimg 64))) (v (f32vec (* nimg 64))))
                (sb-sys:with-pinned-objects (rows)
                  (il::cuda-download out (sb-sys:vector-sap rows) (* 4 nimg 64)))
                (il::cuda-sync dev)
                (dotimes (c 64) (dotimes (i nimg) (setf (aref v (+ (* c nimg) i)) (aref rows (+ (* i 64) c)))))
                v))))))))
