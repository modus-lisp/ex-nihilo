;;;; generate.lisp — a picture from a prompt (docs/design.md, section 3):
;;;; noise, Turbo's eight Euler steps of the DiT's velocity, the latent
;;;; denormalized and decoded, written as a PNG.

(in-package #:ex-nihilo)

(defparameter +turbo-sigmas+
  '(1.0d0 0.978453d0 0.95418d0 0.926626d0 0.89508d0 0.845148d0 0.704534d0 0.414568d0 0d0)
  "Qwen-Image-2.1-Turbo's schedule (its checkpoint's, as sd.cpp's qwen-image-2.1-turbo preset).")

;;; the decoder's latent statistics (its config's latents_mean, latents_std)
(defparameter +latent-mean+
  #(0.5126 0.7721 -0.0631 1.3506 -0.7855 -2.1025 -0.3458 1.3722 1.8873 -1.7177 -0.651 0.2732 0.7562 -0.6163
    -1.0277 3.8363 2.021 0.0472 0.932 2.0087 2.4954 -0.1391 -1.4249 1.8464 -0.5236 1.2826 3.7046 -1.3035
    2.7286 -1.4518 -1.9036 -1.9955 -0.0342 -1.0265 -0.7636 3.0555 0.0746 -3.0751 -0.1076 1.7376 -1.0914
    -1.9435 -0.2784 -1.368 0.4809 -0.4433 0.3764 0.5729 -2.0595 1.096 -1.326 -2.0211 -5.0179 0.5275 4.0162
    1.8505 0.3026 1.9373 1.4937 0.2632 0.5547 -1.7121 -0.1562 0.0304))
(defparameter +latent-std+
  #(3.2001 3.2936 3.4321 3.0091 3.1061 4.0379 4.0705 3.791 3.0785 3.65 3.9308 3.0904 2.8778 3.7675 3.732
    5.0756 3.2864 4.0397 3.1317 4.0443 2.9249 3.9454 3.0988 4.2489 3.4896 3.8513 3.9323 3.4719 3.7498 4.283
    3.5694 4.2467 3.9037 3.2947 5.077 3.5075 3.27 3.4767 2.8063 5.1125 3.5327 4.7833 3.1286 4.1819 3.8527
    3.8312 3.5605 4.3875 3.9624 4.0168 3.5643 4.055 5.5614 4.2963 4.408 3.4959 3.8747 3.7608 3.5735 3.149
    3.7662 3.6746 3.4563 3.8161))

(defstruct (engine (:constructor %make-engine))
  encoder dit vae
  vision                                      ; Qwen3-VL's vision tower, for editing; or NIL
  (prompts (make-hash-table :test 'equal)))   ; prompt -> (states . p), the last few

(defun load-engine (&key (dit "/mnt/ollama/exn/dit-q8_0.gguf") (vae "/mnt/ollama/exn/vae.safetensors")
                         (encoder "/mnt/ollama/exn/te-q8_0.gguf") (vision "/mnt/ollama/exn/mmproj-f16.gguf")
                         (device 0) (encoder-device 1))
  "The models: the DiT and decoder on CUDA device DEVICE, the encoder on
   ENCODER-DEVICE (NIL: the CPU), and beside it, when the encoder is on a
   GPU and the file VISION is there, the vision tower that editing wants."
  (%make-engine :encoder (load-encoder encoder :device encoder-device) :dit (upload-dit dit :device device)
                :vae (upload-vae vae :device device)
                :vision (and encoder-device vision (probe-file vision)
                             #+(and sbcl linux) (load-vision vision :device encoder-device))))

(defun gaussian-noise (n seed)
  "N standard normal floats from SEED (Box-Muller over SBCL's generator)."
  (let ((v (f32vec n)) (rs (sb-ext:seed-random-state seed)))
    (loop for i from 0 below n by 2
          do (let* ((u1 (- 1d0 (random 1d0 rs))) (u2 (random 1d0 rs))
                    (r (sqrt (* -2d0 (log u1)))) (a (* 2 pi u2)))
               (setf (aref v i) (coerce (* r (cos a)) 'single-float))
               (when (< (1+ i) n) (setf (aref v (1+ i)) (coerce (* r (sin a)) 'single-float)))))
    v))

(defun prompt-states (e prompt)
  (or (gethash prompt (engine-prompts e))
      (progn (when (> (hash-table-count (engine-prompts e)) 16) (clrhash (engine-prompts e)))
             (setf (gethash prompt (engine-prompts e))
                   (multiple-value-bind (st p) (encode-prompt (engine-encoder e) prompt) (cons st p))))))

(defun start-sigmas (sigmas strength)
  "SIGMAS from where a picture made from another starts: STRENGTH (0 .. 1]
   the share of the steps run (1: all, the source unused; at least one).
   Not the noise level itself: Turbo's schedule spends most of its steps
   above 0.85, and below that a source hardly changes -- even from 0.978 its
   composition holds, the sky and light remade."
  (let* ((steps (1- (length sigmas)))
         (k (max 0 (min (1- steps) (round (* (- 1 strength) steps))))))
    (nthcdr k sigmas)))

(defun normalized-latent (e px width height)
  "PX (RGBA floats, WIDTH x HEIGHT) encoded, each channel normalized."
  (let* ((x (prog1 (vae-encode (engine-vae e) px height width)
              #+(and sbcl linux) (vae-release (engine-vae e))))
         (n (floor (length x) 64)))
    (dotimes (c 64 x)
      (let ((m (float (aref +latent-mean+ c) 1f0)) (sd (float (aref +latent-std+ c) 1f0)))
        (dotimes (i n)
          (let ((k (+ (* c n) i))) (setf (aref x k) (/ (- (aref x k) m) sd))))))))

(defvar *edits* 0 "A count, to tell one edit's prompt keys from another's.")

(defun edit (e prompt picture pw ph &key (width pw) (height ph) (seed 0) (sigmas +turbo-sigmas+) (verbose t))
  "PICTURE (RGBA floats in -1 .. 1, PW x PH, multiples of 32) edited as
   PROMPT says, Qwen-Image-2.1's way: the picture seen by the prompt encoder
   (its vision tower) and put in the DiT's prefix as a latent, the new one
   made from noise beside it.  (values rgba width height), as GENERATE."
  (assert (and (zerop (mod pw 32)) (zerop (mod ph 32)) (zerop (mod width 16)) (zerop (mod height 16))))
  (unless (engine-vision e) (error "editing wants the vision tower (mmproj-f16.gguf) and the encoder on a GPU"))
  (let* ((t0 (get-internal-real-time)) (h (floor height 16)) (w (floor width 16)) (n (* h w))
         (z (gaussian-noise (* 64 n) seed)) (key (list :edit (incf *edits*))))
    (flet ((elapsed () (/ (- (get-internal-real-time) t0) internal-time-units-per-second)))
      (multiple-value-bind (states p a)
          (encode-prompt (engine-encoder e) prompt :vision (engine-vision e) :picture (list picture ph pw))
        (when verbose (format *error-output* "ex-nihilo: prompt and picture, ~D tokens, ~,1Fs~%" p (elapsed)))
        (let ((ref (list a (normalized-latent e picture pw ph) (floor ph 16) (floor pw 16))))
          (when verbose (format *error-output* "ex-nihilo: the picture encoded, ~,1Fs~%" (elapsed)))
          (loop for (s s2) on sigmas while s2
                do (let ((v (dit-velocity (engine-dit e) states p z h w s :prefix-key key :ref ref)))
                     (dotimes (i (length z)) (incf (aref z i) (* (coerce (- s2 s) 'single-float) (aref v i))))))))
      (when verbose (format *error-output* "ex-nihilo: ~D steps, ~,1Fs~%" (1- (length sigmas)) (elapsed)))
      (decode-latent e z h w verbose #'elapsed))))

(defun decode-latent (e z h w verbose elapsed)
  #+(and sbcl linux) (dit-release (engine-dit e))
  (let ((n (* h w)))
    (dotimes (c 64)
      (let ((m (float (aref +latent-mean+ c) 1f0)) (sd (float (aref +latent-std+ c) 1f0)))
        (dotimes (i n) (setf (aref z (+ (* c n) i)) (+ m (* sd (aref z (+ (* c n) i))))))))
    (multiple-value-bind (px ph pw) (vae-decode (engine-vae e) z h w)
      #+(and sbcl linux) (vae-release (engine-vae e))
      (when verbose (format *error-output* "ex-nihilo: decoded, ~,1Fs~%" (funcall elapsed)))
      (values px pw ph))))

(defun generate (e prompt &key (width 1024) (height 1024) (seed 0) (sigmas +turbo-sigmas+) (verbose t)
                               init (strength 0.8))
  "The picture PROMPT asks for: (values rgba width height), RGBA floats in
   -1 .. 1, rows of WIDTH pixels.  WIDTH and HEIGHT multiples of 16.  With
   INIT, a picture (RGBA floats as returned, WIDTH x HEIGHT), made from it:
   INIT encoded, noised to the sigma START-SIGMAS picks for STRENGTH (flow
   matching: (1 - s) x + s noise), the steps from there."
  (assert (and (zerop (mod width 16)) (zerop (mod height 16))))
  (let* ((t0 (get-internal-real-time)) (h (floor height 16)) (w (floor width 16)) (n (* h w))
         (z (gaussian-noise (* 64 n) seed)))
    (flet ((elapsed () (/ (- (get-internal-real-time) t0) internal-time-units-per-second)))
      (when init
        (assert (= (length init) (* 4 width height)))
        (setf sigmas (start-sigmas sigmas strength))
        (let ((x (normalized-latent e init width height)) (s (float (first sigmas) 1f0)))
          (dotimes (k (length z))
            (setf (aref z k) (+ (* (- 1 s) (aref x k)) (* s (aref z k))))))
        (when verbose (format *error-output* "ex-nihilo: the source encoded, from sigma ~,3F, ~,1Fs~%"
                              (first sigmas) (elapsed))))
      (destructuring-bind (states . p) (prompt-states e prompt)
        (when verbose (format *error-output* "ex-nihilo: prompt, ~D tokens, ~,1Fs~%" p (elapsed)))
        (loop for (s s2) on sigmas while s2
              do (let ((v (dit-velocity (engine-dit e) states p z h w s :prefix-key prompt)))
                   (dotimes (i (length z)) (incf (aref z i) (* (coerce (- s2 s) 'single-float) (aref v i))))))
        (when verbose (format *error-output* "ex-nihilo: ~D steps, ~,1Fs~%" (1- (length sigmas)) (elapsed)))
        (decode-latent e z h w verbose #'elapsed)))))

(defun picture-floats (img width height)
  "IMG (illatio's RGB-IMAGE) as WIDTH x HEIGHT RGBA floats in -1 .. 1, opaque:
   scaled to cover, the middle kept."
  (let* ((sw (il::rgb-image-width img)) (sh (il::rgb-image-height img))
         (k (max (/ width sw) (/ height sh)))
         (dw (max width (round (* sw k)))) (dh (max height (round (* sh k))))
         (ox (floor (- dw width) 2)) (oy (floor (- dh height) 2))
         (canvas (il::resize-bicubic img dw dh (f32vec (* dw dh 3)) dw 0 0))
         (px (f32vec (* width height 4))))
    (dotimes (y height px)
      (dotimes (x width)
        (let ((o (* 3 (+ (* (+ y oy) dw) x ox))) (q (* 4 (+ (* y width) x))))
          (dotimes (c 3) (setf (aref px (+ q c)) (- (/ (aref canvas (+ o c)) 127.5) 1)))
          (setf (aref px (+ q 3)) 1f0))))))

(defun rgba-octets (px &key (alpha t))
  "RGBA floats in -1 .. 1 as 8-bit samples: RGBA, or RGB over white without ALPHA."
  (let* ((n (floor (length px) 4)) (out (make-array (* n (if alpha 4 3)) :element-type '(unsigned-byte 8))))
    (flet ((byte8 (x) (max 0 (min 255 (round (* 127.5 (+ x 1)))))))
      (dotimes (i n out)
        (if alpha
            (dotimes (c 4) (setf (aref out (+ (* 4 i) c)) (byte8 (aref px (+ (* 4 i) c)))))
            (let ((a (/ (+ 1 (aref px (+ (* 4 i) 3))) 2)))
              (dotimes (c 3)
                (setf (aref out (+ (* 3 i) c))
                      (byte8 (+ (* a (aref px (+ (* 4 i) c))) (* (- 1 a) 1)))))))))))

(defun write-picture (px width height path &key (alpha t))
  (png-pure:write-png-file (rgba-octets px :alpha alpha) width height path :channels (if alpha 4 3)))
