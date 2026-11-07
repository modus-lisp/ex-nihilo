;;;; server.lisp — OpenAI's image API in front of GENERATE, on illatio's
;;;; HTTP plumbing (its request reader, RESPOND, the cl-transport listener).
;;;;
;;;;   GET  /health                 -> {"status":"ok"}
;;;;   GET  /v1/models              -> the one model
;;;;   POST /v1/images/generations  -> {"created":..., "data":[{"b64_json":...}, ...]}
;;;;   GET  /                       -> a page to try it from a browser
;;;;
;;;; The request's fields: prompt (required); size "WxH" (default
;;;; 1024x1024, sides multiples of 16 from 256 to 2048, at most 1536x1024's
;;;; pixels; "auto" is the default); n, 1 to 4 pictures; response_format
;;;; "b64_json" (default) or "url" (a data: URL); background "transparent"
;;;; for RGBA rather than RGB; and, beyond OpenAI's, seed (n pictures take
;;;; seed, seed + 1, ...; random when absent).  model, quality, style and
;;;; output_format (png only) are accepted and otherwise ignored.  One
;;;; picture at a time: the GPU is one engine's.

(in-package #:ex-nihilo)

(defparameter *model-id* "qwen-image-2.1-turbo")
(defparameter +max-pixels+ (* 1536 1024))

(defvar *engine* nil)
(defvar *gpu-lock* (sb-thread:make-mutex :name "ex-nihilo engine"))

;;; --- base64 -----------------------------------------------------------------

(defparameter +base64+ "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")

(defun base64-encode (octets)
  (let* ((n (length octets)) (out (make-string (* 4 (ceiling n 3)) :initial-element #\=)))
    (loop for i from 0 below n by 3
          for o from 0 by 4
          do (let* ((k (min 3 (- n i)))
                    (w (logior (ash (aref octets i) 16)
                               (if (> k 1) (ash (aref octets (+ i 1)) 8) 0)
                               (if (> k 2) (aref octets (+ i 2)) 0))))
               (dotimes (j (1+ k))
                 (setf (char out (+ o j)) (char +base64+ (ldb (byte 6 (- 18 (* 6 j))) w))))))
    out))

;;; --- requests -----------------------------------------------------------------

(define-condition bad-request (error) ((message :initarg :message :reader bad-request-message))
  (:report (lambda (c s) (write-string (bad-request-message c) s))))

(defun bad (fmt &rest args) (error 'bad-request :message (apply #'format nil fmt args)))

(defun parse-size (size)
  "(values width height) for SIZE, \"WxH\" or \"auto\"."
  (if (or (null size) (string-equal size "auto"))
      (values 1024 1024)
      (let* ((x (position #\x size :test #'char-equal))
             (w (and x (parse-integer size :end x :junk-allowed t)))
             (h (and x (parse-integer size :start (1+ x) :junk-allowed t))))
        (unless (and w h) (bad "size must be WIDTHxHEIGHT, e.g. 1024x1024, not ~S" size))
        (unless (and (<= 256 w 2048) (<= 256 h 2048) (zerop (mod w 16)) (zerop (mod h 16)))
          (bad "size ~S: each side a multiple of 16 from 256 to 2048" size))
        (when (> (* w h) +max-pixels+)
          (bad "size ~S: at most ~D pixels (1536x1024)" size +max-pixels+))
        (values w h))))

(defun image-request (req)
  "REQ's fields checked: (values prompt width height n seed format alpha)."
  (let ((prompt (il::jget req "prompt")) (n (or (il::jget req "n") 1))
        (seed (il::jget req "seed")) (format (or (il::jget req "response_format") "b64_json"))
        (output (il::jget req "output_format")))
    (unless (and (stringp prompt) (plusp (length (string-trim " " prompt))))
      (bad "prompt must be a non-empty string"))
    (unless (and (integerp n) (<= 1 n 4)) (bad "n must be an integer from 1 to 4"))
    (unless (or (null seed) (and (integerp seed) (<= 0 seed (1- (expt 2 32)))))
      (bad "seed must be an integer from 0 to 2^32 - 1"))
    (unless (member format '("b64_json" "url") :test #'equal)
      (bad "response_format must be \"b64_json\" or \"url\""))
    (unless (member output '(nil "png") :test #'equal) (bad "output_format: only \"png\""))
    (multiple-value-bind (w h) (parse-size (il::jget req "size"))
      (values prompt w h n (or seed (random (expt 2 32) (make-random-state t))) format
              (equal (il::jget req "background") "transparent")))))

(defun images-object (req)
  (multiple-value-bind (prompt w h n seed format alpha) (image-request req)
    (let ((data (loop for k below n
                      collect (let ((png (sb-thread:with-mutex (*gpu-lock*)
                                           (multiple-value-bind (px pw ph)
                                               (generate *engine* prompt :width w :height h :seed (+ seed k))
                                             (png-pure:encode-png (rgba-octets px :alpha alpha) pw ph
                                                                  :channels (if alpha 4 3))))))
                                (if (equal format "url")
                                    (il::hobj "url" (concatenate 'string "data:image/png;base64," (base64-encode png))
                                              "revised_prompt" prompt "seed" (+ seed k))
                                    (il::hobj "b64_json" (base64-encode png) "revised_prompt" prompt
                                              "seed" (+ seed k)))))))
      (il::hobj "created" (- (get-universal-time) #.(encode-universal-time 0 0 0 1 1 1970 0))
                "data" (coerce data 'vector)
                "size" (format nil "~Dx~D" w h) "output_format" "png"))))

;;; --- the page -------------------------------------------------------------------

(defparameter +page+ "<!doctype html>
<html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>ex nihilo</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mute:#6b6b66;--line:#d9d9d4;--accent:#3d5a80}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--mute:#9a9a94;--line:#33332f;--accent:#8fb3e0}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.45 system-ui,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px}
h1{font-weight:600;font-size:20px;margin:0 0 16px}
textarea{width:100%;box-sizing:border-box;min-height:72px;font:inherit;padding:10px;border:1px solid var(--line);border-radius:6px;background:transparent;color:inherit}
.row{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:10px 0 18px}
select,input,button{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:transparent;color:inherit}
input{width:110px} button{background:var(--accent);color:var(--bg);border-color:var(--accent);cursor:pointer}
button:disabled{opacity:.5;cursor:wait} #status{color:var(--mute);font-size:14px}
#out img{max-width:100%;height:auto;display:block;margin:0 0 16px;border-radius:4px}
</style></head><body><main>
<h1>ex nihilo</h1>
<textarea id=p placeholder='a red apple on a wooden table, photograph'></textarea>
<div class=row>
<select id=s><option>1024x1024<option>512x512<option>768x768<option>1536x1024<option>1024x1536<option>1280x720<option>720x1280</select>
<input id=seed type=number min=0 placeholder=seed>
<button id=go>Generate</button><span id=status></span></div>
<div id=out></div>
<script>
const $=id=>document.getElementById(id);
async function go(){
  const prompt=$('p').value.trim(); if(!prompt) return;
  const body={prompt,size:$('s').value}; if($('seed').value!=='') body.seed=+$('seed').value;
  $('go').disabled=true; const t0=performance.now(); $('status').textContent='generating…';
  try{
    const r=await fetch('/v1/images/generations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json(); if(!r.ok) throw new Error(j.error?j.error.message:r.status);
    const d=j.data[0], img=new Image(); img.src='data:image/png;base64,'+d.b64_json; img.alt=prompt; img.title='seed '+d.seed;
    $('out').prepend(img); $('status').textContent=`seed ${d.seed}, ${((performance.now()-t0)/1000).toFixed(1)} s`;
  }catch(e){$('status').textContent=String(e.message||e)}
  finally{$('go').disabled=false}
}
$('go').onclick=go; $('p').onkeydown=e=>{if(e.key==='Enter'&&(e.metaKey||e.ctrlKey)) go()};
</script></main></body></html>")

;;; --- the listener -----------------------------------------------------------------

(defun handle-connection (s)
  (handler-case
      (multiple-value-bind (method path headers body) (il::read-request s)
        (declare (ignore headers))
        (let ((path (and path (subseq path 0 (or (position #\? path) (length path))))))
          (cond
            ((null method))
            ((and (string= method "GET") (string= path "/health"))
             (il::respond s "200 OK" (il::hobj "status" "ok")))
            ((and (string= method "GET") (string= path "/v1/models"))
             (il::respond s "200 OK" (il::hobj "object" "list"
                                               "data" (vector (il::hobj "id" *model-id* "object" "model"
                                                                        "owned_by" "ex-nihilo")))))
            ((and (string= method "GET") (string= path "/"))
             (il::respond s "200 OK" +page+ :type "text/html; charset=utf-8"))
            ((and (string= method "POST") (member path '("/v1/images/generations" "/images/generations")
                                                  :test #'string=))
             (let ((req (ignore-errors (json-simple:parse body))))
               (if (hash-table-p req)
                   (handler-case (il::respond s "200 OK" (images-object req))
                     (bad-request (e) (il::respond s "400 Bad Request" (il::error-object (princ-to-string e)))))
                   (il::respond s "400 Bad Request" (il::error-object "the body is not a JSON object")))))
            (t (il::respond s "404 Not Found"
                            (il::error-object (format nil "no route for ~A ~A" method path) "not_found"))))))
    (stream-error () nil)               ; the client went away
    (error (e)
      (ignore-errors (il::respond s "500 Internal Server Error"
                                  (il::error-object (princ-to-string e) "server_error"))))))

(defun serve (engine &key (host "127.0.0.1") (port 9940))
  "Answer OpenAI image generations with ENGINE on HOST:PORT; returns a
   function that stops the server.  Loopback by default: there is no
   authentication in front of it."
  (setf *engine* engine)
  (cl-transport:expose (lambda (stream peer) (declare (ignore peer)) (handle-connection stream))
                       :backend :tcp :host host :port port))

(defun main (args)
  "bin/ex-nihilo: [--host H] [--port P] [--weights DIR] [--device N] [--encoder-device N|cpu]"
  (let ((host "127.0.0.1") (port 9940) (dir "/dev/shm/exn/") (device 0) (encoder-device 1))
    (loop while args
          do (let ((flag (pop args)) (v (pop args)))
               (unless v (error "~A wants a value" flag))
               (cond ((string= flag "--host") (setf host v))
                     ((string= flag "--port") (setf port (parse-integer v)))
                     ((string= flag "--weights") (setf dir (uiop:ensure-directory-pathname v)))
                     ((string= flag "--device") (setf device (parse-integer v)))
                     ((string= flag "--encoder-device")
                      (setf encoder-device (if (string-equal v "cpu") nil (parse-integer v))))
                     (t (error "unknown flag ~A" flag)))))
    (let ((e (load-engine :dit (namestring (merge-pathnames "dit-q8_0.gguf" dir))
                          :vae (namestring (merge-pathnames "vae.safetensors" dir))
                          :encoder (namestring (merge-pathnames "te-q8_0.gguf" dir))
                          :device device :encoder-device encoder-device)))
      (serve e :host host :port port)
      (format *error-output* "ex-nihilo: listening on http://~A:~D/~%" host port)
      (loop (sleep 3600)))))
