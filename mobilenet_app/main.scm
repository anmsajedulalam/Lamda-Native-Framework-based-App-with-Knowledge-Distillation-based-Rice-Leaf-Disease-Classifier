;; main.scm - MobileNet Classifier for Lambda Native
(define main-layout #f)
(define prediction-text #f)
(define confidence-text #f)
(define preview-image #f)
(define class-names #f)
(define processing? #f)

(define (load-class-names)
  (let ((class-file "plugins/class_names.json"))
    (if (file-exists? class-file)
        (json-parse (file-read class-file))
        '("Class 0" "Class 1" "Class 2" "Class 3" "Class 4"))))

(define (on-create-handler)
  ;; Load class names
  (set! class-names (load-class-names))
  
  ;; Create main layout
  (set! main-layout (make-layout 'vertical))
  
  ;; Title
  (layout-add main-layout 
    (make-textview "MobileNet V4 Classifier" 'large) 10 10)
  
  ;; Buttons
  (let ((button-layout (make-layout 'horizontal)))
    (layout-add button-layout
      (make-button "Take Photo" take-photo) 10 10)
    (layout-add button-layout
      (make-button "Choose Image" choose-image) 10 10)
    (layout-add main-layout button-layout))
  
  ;; Image preview
  (set! preview-image (make-imageview ""))
  (imageview-set-scale-type preview-image 'fit-center)
  (layout-add main-layout preview-image 10 160 300 300)
  
  ;; Prediction results
  (set! prediction-text (make-textview "Prediction: None" 'medium))
  (layout-add main-layout prediction-text 10 470)
  
  (set! confidence-text (make-textview "Confidence: 0%" 'medium))
  (layout-add main-layout confidence-text 10 500)
  
  ;; Progress indicator
  (layout-add main-layout
    (make-textview "Status: Ready" 'small) 10 530)
  
  (layout-set main-layout))

(define (take-photo)
  (if processing?
      (show-message "Please wait, processing previous image...")
      (camera-capture photo-callback)))

(define (choose-image)
  (if processing?
      (show-message "Please wait, processing previous image...")
      (gallery-pick gallery-callback)))

(define (photo-callback image-path)
  (display "Photo captured: ") (display image-path) (newline)
  (process-image image-path))

(define (gallery-callback image-path)
  (display "Image selected: ") (display image-path) (newline)
  (process-image image-path))

(define (process-image image-path)
  (set! processing? #t)
  (textview-set prediction-text "Processing...")
  (textview-set confidence-text "Please wait...")
  
  ;; Simulate processing delay
  (thread-start
   (lambda ()
     (let ((result (classify-image image-path)))
       (ui-update
        (lambda ()
          (textview-set prediction-text 
            (string-append "Prediction: " (car result)))
          (textview-set confidence-text
            (string-append "Confidence: " (number->string (cadr result)) "%"))
          (set! processing? #f)))))))

(define (classify-image image-path)
  ;; For Lambda Native, we'll use a simulated classification
  ;; In a real app, you'd integrate the ONNX/TFLite runtime
  (let ((predicted-class (random (length class-names)))
        (confidence (+ 70 (random 30))))
    
    (display "Classifying image: ") (display image-path) (newline)
    (display "Predicted: ") (display (list-ref class-names predicted-class)) (newline)
    (display "Confidence: ") (display confidence) (display "%") (newline)
    
    ;; Simulate processing time
    (thread-sleep 2)
    
    (list (list-ref class-names predicted-class) confidence)))

(define (show-message msg)
  (ui-update
   (lambda ()
     (textview-set prediction-text msg)
     (textview-set confidence-text ""))))

;; Start the application
(define (main)
  (application-initialize)
  (activity-create "MainActivity")
  (application-run))

(main)