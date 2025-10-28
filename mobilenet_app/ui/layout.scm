;; UI layout definitions
(define (make-vertical-layout)
  (let ((layout (make-layout 'vertical)))
    layout))

(define (make-horizontal-layout)
  (let ((layout (make-layout 'horizontal)))
    layout))

(define (create-title text)
  (make-textview text 'large))

(define (create-button text callback)
  (make-button text callback))

(define (create-image-view)
  (let ((image-view (make-imageview "")))
    (imageview-set-scale-type image-view 'fit-center)
    image-view))

(define (create-result-text size)
  (make-textview "" size))