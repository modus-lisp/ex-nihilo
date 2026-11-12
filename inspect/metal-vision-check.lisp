;;;; metal-vision-check.lisp — the Metal vision tower against inspect/ref_vision.py (numpy).
;;;;
;;;;   python3 -I inspect/ref_vision.py MMPROJ PIXELS.f32 64 64 OUT_DIR GGUF_PY_DIR
;;;;   EXN_VREF=OUT_DIR sbcl --dynamic-space-size 16384 --non-interactive --load inspect/metal-vision-check.lisp
;;;;
;;;; PIXELS: 64 x 64 x 3 floats in -1 .. 1, raw little-endian (OUT_DIR/pixels.f32).  Prints
;;;; each output's largest error relative to its largest value (5e-6 when it was written).

(require :asdf)
(push (merge-pathnames "../" (make-pathname :name nil :type nil :defaults *load-truename*)) asdf:*central-registry*)
(push (merge-pathnames "../../illatio/" (make-pathname :name nil :type nil :defaults *load-truename*)) asdf:*central-registry*)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system "ex-nihilo"))
(in-package #:ex-nihilo)
(defun read-f32 (path n)
  (let ((v (f32vec n)))
    (with-open-file (s path :element-type '(unsigned-byte 32))
      (dotimes (i n v) (setf (aref v i) (il::bits-single (read-byte s)))))))
(let* ((h 64) (w 64) (rgb (read-f32 (format nil "~A/pixels.f32" (uiop:getenv "EXN_VREF")) (* h w 3))) (px (f32vec (* h w 4)))
       (v (load-vision (weight-file "mmproj-f16.gguf"))))
  (dotimes (i (* h w)) (dotimes (c 3) (setf (aref px (+ (* 4 i) c)) (aref rgb (+ (* 3 i) c)))) (setf (aref px (+ (* 4 i) 3)) 1.0))
  (multiple-value-bind (merged ds m) (let ((t0 (get-internal-real-time)))
                                       (multiple-value-prog1 (vision-encode v px h w)
                                         (format t "~&vision-encode ~Dx~D: ~,2F s~%" h w (/ (- (get-internal-real-time) t0) internal-time-units-per-second))))
    (flet ((cmp (name buf file)
             (let* ((n (* m 4096)) (got (get-floats buf n)) (want (read-f32 file n)) (err 0.0) (size 0.0))
               (dotimes (i n) (setf err (max err (abs (- (aref got i) (aref want i)))) size (max size (abs (aref want i)))))
               (format t "~&~8A max error ~,3E of max ~,3E (relative ~,2E)~%" name err size (/ err size)))))
      (cmp "merged" merged (format nil "~A/merged.f32" (uiop:getenv "EXN_VREF")))
      (loop for b in ds for i from 0 do (cmp (format nil "ds~D" i) b (format nil "~A/ds~D.f32" (uiop:getenv "EXN_VREF") i))))))
