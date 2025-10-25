# pylint:disable=no-member

import numpy as np
import cv2 as cv
from picamera2 import Picamera2
import time

# Load the Haar Cascade classifier for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# List of people the model can recognize
people = ['Epan', 'Ikram' ,'Fatah Amin', 'Kamalla Haris', 'Barack Obama']  # Add more names if your model is trained for multiple people

# Create and load the face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('facetrained.yml')

# Initialize the Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# Confidence threshold to determine if a face is recognized
threshold = 80  # Adjust this value based on your requirements

# Allow camera to warm up
time.sleep(2)

while True:
    # Capture a frame
    frame = picam2.capture_array()

    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Process each detected face
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]

        # Predict the label and confidence for the face
        label, confidence = face_recognizer.predict(faces_roi)

        # Determine if the face is recognized or unknown
        if confidence < threshold:
            name = people[label]
        else:
            name = "Unknown"

        # Draw rectangle around the face and put the label
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(frame, f'{name} ({confidence:.2f})', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv.imshow('Live Face Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Close all windows
cv.destroyAllWindows()