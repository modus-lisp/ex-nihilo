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
