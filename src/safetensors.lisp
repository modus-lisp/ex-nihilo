;;;; safetensors.lisp — the format the decoder ships in: an 8-byte
;;;; little-endian length, that many bytes of JSON naming each tensor's dtype,
;;;; shape (row-major, PyTorch's order) and byte span, then the bytes.

(in-package #:ex-nihilo)

(defstruct (safetensors (:constructor %make-safetensors) (:conc-name st-))
  path
  (data-offset 0 :type fixnum)
  (tensors (make-hash-table :test 'equal)) ; name -> (dtype shape start end)
  metadata)

(defun read-safetensors (path)
  "The tensor directory of the safetensors file at PATH."
  (with-open-file (s path :element-type '(unsigned-byte 8))
    (let ((n 0) (len (make-array 8 :element-type '(unsigned-byte 8))))
      (read-sequence len s)
      (dotimes (i 8) (setf n (logior n (ash (aref len i) (* 8 i)))))
      (let ((json (make-array n :element-type '(unsigned-byte 8)))
            (st (%make-safetensors :path path :data-offset (+ 8 n))))
        (unless (= (read-sequence json s) n) (error "~A: truncated header" path))
        (maphash (lambda (name v)
                   (if (string= name "__metadata__")
                       (setf (st-metadata st) v)
                       (let ((span (gethash "data_offsets" v)))
                         (setf (gethash name (st-tensors st))
                               (list (gethash "dtype" v) (coerce (gethash "shape" v) 'list)
                                     (aref span 0) (aref span 1))))))
                 (json-simple:parse json))
        st))))

(defun st-names (st)
  (sort (loop for k being the hash-keys of (st-tensors st) collect k) #'string<))

(defun st-entry (st name)
  (or (gethash name (st-tensors st)) (error "~A: no tensor named ~S" (st-path st) name)))

(defun st-shape (st name) (second (st-entry st name)))
(defun st-dtype (st name) (first (st-entry st name)))

(defun st-f32 (st name)
  "Tensor NAME as a flat single-float vector, in the file's (row-major)
   order: F32, F16 or BF16 on disk."
  (destructuring-bind (dtype shape start end) (st-entry st name)
    (let* ((n (reduce #'* shape)) (v (f32vec n))
           (width (cond ((string= dtype "F32") 4) ((member dtype '("F16" "BF16") :test #'string=) 2)
                        (t (error "~A: ~A is ~A" (st-path st) name dtype))))
           (buf (make-array (- end start) :element-type '(unsigned-byte 8))))
      (declare (type (simple-array (unsigned-byte 8) (*)) buf))
      (assert (= (- end start) (* n width)))
      (with-open-file (s (st-path st) :element-type '(unsigned-byte 8))
        (file-position s (+ (st-data-offset st) start))
        (unless (= (read-sequence buf s) (length buf)) (error "~A: truncated tensor ~A" (st-path st) name)))
      (flet ((u16 (i) (logior (aref buf i) (ash (aref buf (1+ i)) 8))))
        (cond ((string= dtype "BF16") (dotimes (i n) (setf (aref v i) (bits-single (ash (u16 (* 2 i)) 16)))))
              ((string= dtype "F16") (dotimes (i n) (setf (aref v i) (half-to-single (u16 (* 2 i))))))
              (t (dotimes (i n)
                   (setf (aref v i) (bits-single (logior (u16 (* 4 i)) (ash (u16 (+ 2 (* 4 i))) 16))))))))
      v)))
