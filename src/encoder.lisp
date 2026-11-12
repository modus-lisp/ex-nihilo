;;;; encoder.lisp — the prompt: Qwen3-VL-8B's text model (illatio's qwen3
;;;; forward, on the CPU) read for its last block's states, before the final
;;;; norm, with the template's system turn cut off (docs/design.md, section 1).

(in-package #:ex-nihilo)

(defparameter +system-turn+
  (format nil "<|im_start|>system~%Comprehend and analyze the provided prompt.<|im_end|>~%"))

(defun prompt-text (prompt)
  (format nil "~A<|im_start|>user~%~A<|im_end|>~%<|im_start|>assistant~%"
          +system-turn+ (if (string= prompt "") " " prompt)))

;;; A picture to edit goes in the user's turn ahead of the prompt, as
;;; diffusers' QwenImage21Pipeline puts it: "<image1>", then the vision
;;; tokens, one <|image_pad|> per 32 x 32 px, which the vision tower's rows
;;; replace.

(defparameter +image-pad+ 151655)

(defun edit-prompt-pieces (prompt)
  "The prompt's text before the picture's pads and after them."
  (values (format nil "~A<|im_start|>user~%<image1><|vision_start|>" +system-turn+)
          (format nil "<|vision_end|>~A<|im_end|>~%<|im_start|>assistant~%" (if (string= prompt "") " " prompt))))

(defun mrope-table (positions rd base)
  "Qwen3-VL's interleaved M-RoPE as illatio's rope rows (RD/2 cosines, RD/2
   sines), one per (t h w) of POSITIONS: frequency j turns by h where j mod 3
   is 1, by w where it is 2 (below 3 x 20), by t elsewhere."
  (let* ((half (floor rd 2)) (out (f32vec (* (length positions) rd))))
    (loop for (tp hp wp) in positions for s from 0
          do (multiple-value-bind (ct st) (il::rope-table rd tp base)
               (multiple-value-bind (ch sh) (il::rope-table rd hp base)
                 (multiple-value-bind (cw sw) (il::rope-table rd wp base)
                   (dotimes (j half)
                     (let ((axis (if (< j 60) (mod j 3) 0)))
                       (setf (aref out (+ (* s rd) j)) (aref (case axis (1 ch) (2 cw) (t ct)) j)
                             (aref out (+ (* s rd) half j)) (aref (case axis (1 sh) (2 sw) (t st)) j))))))))
    out))

(defun load-encoder (path &key device (threads (il::default-thread-count)))
  "The prompt encoder: the qwen3vl GGUF at PATH, its text model -- on CUDA
   device DEVICE (illatio's dense path), or on the CPU when DEVICE is NIL."
  #+(and sbcl linux) (when device (return-from load-encoder (il::load-model-cuda path :device device)))
  (il::load-model path :threads threads))

(defun encode-prompt (model prompt &key vision picture)
  "(values states p a): the P x 4096 floats the DiT reads for PROMPT.  With
   PICTURE, (PX H W) -- RGBA floats in -1 .. 1, H and W multiples of 32 --
   seen through VISION (LOAD-VISION's tower, on the model's GPU): the
   prompt's states are its text's, the picture's rows left out, and A of
   them come before the picture."
  (when picture
    (return-from encode-prompt (encode-edit-prompt model prompt vision picture)))
  (let* ((ids (il::tokenize model (prompt-text prompt) :add-bos nil))
         (skip (length (il::tokenize model +system-turn+ :add-bos nil)))
         (n (length ids)))
    (multiple-value-bind (all) (encoder-states model ids)
      (values (subseq all (* skip (il::model-d model))) (- n skip) 0))))

(defun encode-edit-prompt (model prompt vision picture)
  #-(and sbcl linux) (declare (ignore model prompt vision picture))
  #-(and sbcl linux) (error "editing a picture wants the encoder on a GPU")
  #+(and sbcl linux)
  (destructuring-bind (px h w) picture
    (unless (il::model-cuda model) (error "editing a picture wants the encoder on a GPU"))
    (multiple-value-bind (before after) (edit-prompt-pieces prompt)
      (let* ((b (il::tokenize model before :add-bos nil)) (e (il::tokenize model after :add-bos nil))
             (skip (length (il::tokenize model +system-turn+ :add-bos nil)))
             (gh (floor h 32)) (gw (floor w 32)) (m (* gh gw)) (a0 (length b))
             (ids (concatenate 'vector b (make-array m :initial-element +image-pad+) e))
             (n (length ids)) (d (il::model-d model))
             (positions (append (loop for i below a0 collect (list i i i))
                                (loop for r below gh
                                      append (loop for c below gw collect (list a0 (+ a0 r) (+ a0 c))))
                                (loop with p0 = (+ a0 (max gh gw)) for i below (length e)
                                      collect (list (+ p0 i) (+ p0 i) (+ p0 i))))))
        (multiple-value-bind (merged deep) (vision-encode vision px h w)
          (flet ((rows-in (pos0 k)        ; the picture's rows among this pass's: (row0 . start), count
                   (let ((lo (max pos0 a0)) (hi (min (+ pos0 k) (+ a0 m))))
                     (when (< lo hi) (values (- lo pos0) (- lo a0) (- hi lo))))))
            (let* ((dev (il::cmodel-dev (il::model-cuda model)))
                   (il::*after-embed*
                     (lambda (x pos0 k)
                       (multiple-value-bind (at from count) (rows-in pos0 k)
                         (when at
                           (il::cuda-copy-dd x (* 4 at d) merged (* 4 from d) (* 4 count d))))))
                   (il::*after-layer*
                     (lambda (li x pos0 k)
                       (when (< li (length deep))
                         (multiple-value-bind (at from count) (rows-in pos0 k)
                           (when at
                             (il::launch dev "v_add" (ceiling (* count d) 256) 256
                                         (list (list :i64 (* count d)) (cons x (* 4 at d))
                                               (cons (nth li deep) (* 4 from d))))))))))
              (let ((all (encoder-states model ids
                                         :rope (mrope-table positions (il::model-rope-dims model)
                                                            (il::model-rope-base model)))))
                ;; the text's states, the system turn and the picture's rows left out
                (let ((out (f32vec (* (- n skip m) d))))
                  (replace out all :start2 (* skip d) :end2 (* a0 d))
                  (replace out all :start1 (* (- a0 skip) d) :start2 (* (+ a0 m) d))
                  (vision-release vision)
                  (values out (- n skip m) (- a0 skip)))))))))))

(defun encoder-states (model ids &key rope)
  "Every token's state after the text model's last block, before its final
   norm: (length IDS) x 4096 floats.  ROPE: rope rows for the positions in
   place of illatio's (a GPU encoder's)."
  (let* ((d (il::model-d model)) (n (length ids))
         (ctx (il::make-context model :n-ctx (+ n 8)))
         (all (f32vec (* n d))))
    #+(and sbcl linux)
    (when rope
      (il::with-cuda
        (il::cuda-current (il::cmodel-dev (il::model-cuda model)))
        (il::cuda-upload-vector (il::cctx-rope (il::cuda-state ctx)) rope)))
    (unwind-protect
         (loop with gpu = #+(and sbcl linux) (il::model-cuda model) #-(and sbcl linux) nil
               with step = (if gpu il::*cuda-chunk* il::*prefill-chunk*)
               with done = 0
               while (< done n)
               do (let* ((chunk (subseq ids done (min n (+ done step)))) (k (length chunk)))
                    ;; each pass leaves every token's state after the last block in its scratch
                    (cond #+(and sbcl linux)
                          (gpu
                           (il::cuda-forward-n ctx chunk nil)
                           (let ((x (il::cctx-x (il::cuda-state ctx))))
                             (il::with-cuda
                               (il::cuda-current (il::cmodel-dev gpu))
                               (sb-sys:with-pinned-objects (all)
                                 (il::cuda-download x (sb-sys:sap+ (sb-sys:vector-sap all) (* 4 done d)) (* 4 k d)))
                               (il::cuda-sync (il::cmodel-dev gpu)))))
                          (t
                           (il::forward-batch ctx chunk :logits nil)
                           (replace all (il::batch-scratch-x (il::batch-scratch ctx)) :start1 (* done d) :end2 (* k d))))
                    (incf done k)))
      (il::free-context ctx))
    all))
