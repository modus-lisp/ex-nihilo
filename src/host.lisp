;;;; host.lisp — the host's share of the DiT and the decoder, whichever GPU
;;;; runs them (cuda-*.lisp, metal-*.lisp): the timestep's embedding, the
;;;; rotary tables and positions, and the decoder's buffer sizes.

(in-package #:ex-nihilo)

;;; --- the DiT ------------------------------------------------------------------------

(defun timestep-embedding (tt &optional (dim 256) (period 10000d0))
  "cos then sin of TT times each of DIM/2 frequencies (ggml_timestep_embedding)."
  (let* ((half (floor dim 2)) (v (f32vec dim)))
    (dotimes (j half v)
      (let ((a (* tt (exp (/ (* (- (log period)) j) half)))))
        (setf (aref v j) (coerce (cos a) 'single-float)
              (aref v (+ j half)) (coerce (sin a) 'single-float))))))

(defparameter +rope-axes+ '(16 56 56))

(defun rope-table (positions)
  "cos, sin of each token's 64 angles (POSITIONS: a list of (a b c)), 128 floats a token."
  (let ((v (f32vec (* 128 (length positions)))))
    (loop for pos in positions for tk from 0
          do (let ((j 0))
               (loop for d in +rope-axes+ for p in pos
                     do (loop for i below (floor d 2)
                              do (let ((ang (* p (/ 1d0 (expt 10000d0 (/ (* 2 i) d))))))
                                   (setf (aref v (+ (* tk 128) (* 2 j))) (coerce (cos ang) 'single-float)
                                         (aref v (+ (* tk 128) (* 2 j) 1)) (coerce (sin ang) 'single-float))
                                   (incf j))))))
    v))

(defun image-positions (frame h w)
  "An H x W block's positions: FRAME, and its rows and columns centred on zero."
  (loop for r below h
        append (loop for c below w collect (list frame (- r (- h (floor h 2))) (- c (- w (floor w 2)))))))

(defun dit-positions (p h w &optional ref)
  "Every row's (frame row col): the prompt's text at (i, i, i), the H x W image
   after it at the position reached, centred.  With REF, (A RH RW): the prompt
   is A text rows, a RH x RW reference picture, then the rest of its P rows'
   text; the picture's frame is A, and the text after it goes on from A +
   max(RH, RW) (diffusers' QwenImage21Rope)."
  (if (null ref)
      (append (loop for i below p collect (list i i i)) (image-positions p h w))
      (destructuring-bind (a rh rw) ref
        (let* ((after (- p a (* rh rw))) (pos (+ a (max rh rw))))
          (append (loop for i below a collect (list i i i))
                  (image-positions a rh rw)
                  (loop for i below after collect (let ((x (+ pos i))) (list x x x)))
                  (image-positions (+ pos after) h w))))))

;;; --- the decoder --------------------------------------------------------------------

(defun vae-sizes (h w)
  "(values out in): the most floats a stage's output and a stage's input
   (at its own size) take, for an H x W latent."
  (let ((hh h) (ww w) (out (* h w (first (first +vae-stages+)))) (in 0))
    (loop for (i o up) in +vae-stages+
          do (setf in (max in (* hh ww (max i o))))
             (when up (setf hh (* 2 hh) ww (* 2 ww)))
             (setf out (max out (* hh ww o))))
    (values out in)))

(defun vae-enc-sizes (h w)
  "(values out in) for the encoder on an H x W picture: the most floats a
   stage's output and a stage's input (at its own size) take."
  (let ((hh h) (ww w) (out (* h w (first (first +vae-enc-stages+)))) (in (* h w 16)))
    (loop for (i o down) in +vae-enc-stages+
          do (setf in (max in (* hh ww (max i o))) out (max out (* hh ww o)))
             (when down (setf hh (floor hh 2) ww (floor ww 2))))
    (values out in)))

(defun vae-tensors (st)
  "The decoder's and encoder's tensors in the safetensors ST as the GPUs take
   them: (name . floats), each 3 x 3 convolution's weights as COUT rows of
   (ky, kx, ci).  A (o i 1 3 3) convolution's input channels are made a
   multiple of 16 (the convolutions' tiles take 16 at a time): the
   encoder's first, of RGBA, padded with zero weights."
  (loop for (name . shape) in (append (vae-expected) (vae-encoder-expected))
        collect (let ((f (st-f32 st name)))
                  (when (and (= (length shape) 5) (= (fifth shape) 3)) ; (o i 1 3 3) -> o rows of (ky kx ip)
                    (destructuring-bind (o i one k k2) shape
                      (declare (ignore one k2))
                      (let* ((ip (* 16 (ceiling i 16))) (r (f32vec (* o 9 ip))))
                        (dotimes (co o)
                          (dotimes (ci i)
                            (dotimes (ky k)
                              (dotimes (kx k)
                                (setf (aref r (+ (* co 9 ip) (* (+ (* ky 3) kx) ip) ci))
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
                  (cons name f))))

;;; --- Qwen3-VL's vision tower (cuda-vision.lisp, metal-vision.lisp) -----------------------

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
  dev                                  ; the CUDA device, or the Metal GPU
  (w (make-hash-table :test 'equal))   ; name -> a buffer of floats (a matrix row-major, out x in)
  pos side                             ; the learned position table (host floats), its side
  (buf (make-hash-table)))             ; scratch: key -> (dbuf . floats)

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

(defun check-vision (g path)
  (unless (equal (meta g "clip.projector_type") "qwen3vl_merger")
    (error "~A: not Qwen3-VL's vision tower (projector ~A)" path (meta g "clip.projector_type"))))

(defun vision-tensors (g)
  "The tower's tensors in the mmproj GGUF G as floats: (values ((name .
   floats) ...) position-table side).  The second patch embedding (one
   frame, given twice) is summed into the first; the position table stays
   on the host, to be resampled to each picture's grid."
  (let ((out '()) pos side)
    (dolist (name (gguf-tensor-order g))
      (cond ((string= name "v.position_embd.weight")
             (setf pos (il::load-f32 g name) side (isqrt (floor (length pos) +qv-width+))))
            ((string= name "v.patch_embd.weight.1"))
            ((string= name "v.patch_embd.weight")
             (let ((a (il::load-f32 g name)) (b (il::load-f32 g "v.patch_embd.weight.1")))
               (dotimes (i (length a)) (incf (aref a i) (aref b i)))
               (push (cons name a) out)))
            (t (push (cons name (il::load-f32 g name)) out))))
    (values (nreverse out) pos side)))
