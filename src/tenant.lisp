;;;; tenant.lisp — ex-nihilo inside illatio's server, sharing its GPUs.
;;;;
;;;; bin/illatio-server --images DIR loads this system (illatio does not
;;;; depend on ex-nihilo; it finds it beside itself) and calls MAKE-TENANT
;;;; before its chat model is loaded: the three models go to their devices,
;;;; then all of their device memory given up (illatio's PARK-MEMORY: on CUDA
;;;; to pinned host memory, on Metal released from its working set, a wrapped
;;;; file just unwrapped).  ATTACH-TENANT adds the image routes to illatio's.  A
;;;; request's pictures then borrow the GPUs between chat requests: the chat
;;;; model gives up what it holds as a cache (DeepSeek's expert pools,
;;;; LEND-DEVICES), these come back from host memory, the pictures are made,
;;;; they go back to host memory, the chat model's caches are made again.

(in-package #:ex-nihilo)

(defstruct (tenant (:constructor %make-tenant)) engine parking chat)

(defun make-tenant (&key weights encoder (device 0) (encoder-device 1))
  "ex-nihilo's engine from the weights in directory WEIGHTS (ENCODER, a
   file, in place of its te-q8_0.gguf), loaded, then parked in host memory."
  (let* ((dir (uiop:ensure-directory-pathname weights))
         (parking (il::make-parking "ex-nihilo"))
         (t0 (get-internal-real-time))
         (engine (il::with-parking-record (parking :const t)
                   (load-engine :dit (namestring (merge-pathnames "dit-q8_0.gguf" dir))
                                :vae (namestring (merge-pathnames "vae.safetensors" dir))
                                :encoder (or encoder (namestring (merge-pathnames "te-q8_0.gguf" dir)))
                                :vision (namestring (merge-pathnames "mmproj-f16.gguf" dir))
                                :device device :encoder-device encoder-device)))
         (bytes (il::park-memory parking)))
    (format *error-output* "ex-nihilo: loaded and parked (~,1F GB given up), ~,1Fs~%"
            (/ bytes 1d9) (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
    (%make-tenant :engine engine :parking parking)))

(defun tenant-run (tenant fn)
  "FN with the tenant's models on their devices, the chat model's caches
   given up meanwhile."
  (il::lend-devices (tenant-chat tenant)
                    (lambda ()
                      (let ((p (tenant-parking tenant)) (t0 (get-internal-real-time)))
                        (il::unpark-memory p)
                        (format *error-output* "ex-nihilo: unparked, ~,1Fs~%"
                                (/ (- (get-internal-real-time) t0) internal-time-units-per-second))
                        (unwind-protect
                             (il::with-parking-record (p)     ; scratch and caches made now are parked too
                               (funcall fn))
                          (let ((t1 (get-internal-real-time)) (bytes (il::park-memory p)))
                            (format *error-output* "ex-nihilo: parked (~,1F GB:~{ ~,1F on ~A~^,~}), ~,1Fs~%"
                                    (/ bytes 1d9) (parked-by-device p)
                                    (/ (- (get-internal-real-time) t1) internal-time-units-per-second))))))))

(defun parked-by-device (p)
  "(GB device-name ...) for P's parked buffers (on Metal, one GPU)."
  #+(and sbcl darwin) (list (/ (reduce #'+ (il::parking-parked p) :key #'il::gbuf-bytes) 1d9) "the GPU")
  #-(and sbcl darwin)
  (let ((by (make-hash-table)))
    (dolist (b (il::parking-parked p))
      (incf (gethash (il::cudev-index (il::dbuf-dev b)) by 0) (il::dbuf-bytes b)))
    (loop for i in (sort (loop for k being the hash-keys of by collect k) #'<)
          append (list (/ (gethash i by) 1d9) i))))

(defun attach-tenant (tenant chat-engine)
  "TENANT's routes added to illatio's server, its pictures made between
   CHAT-ENGINE's requests."
  (setf (tenant-chat tenant) chat-engine
        *engine* (tenant-engine tenant)
        *on-gpu* (lambda (fn) (tenant-run tenant fn)))
  (il::add-route "POST" +images-paths+ #'handle-images)
  (il::add-route "GET" '("/images") (lambda (s body) (declare (ignore body))
                                      (il::respond s "200 OK" +page+ :type "text/html; charset=utf-8")))
  (push (il::hobj "id" *model-id* "object" "model" "owned_by" "ex-nihilo") il::*more-models*)
  tenant)
