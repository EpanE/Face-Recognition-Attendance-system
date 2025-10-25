# pylint:disable=no-member

import numpy as np
import cv2 as cv
import pandas as pd

# Load the list of people from the Excel file
people_df = pd.read_excel('people_list.xlsx')  # Update with your file path
people = people_df['Name'].tolist()

# Load the Haar Cascade classifier for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Create and load the face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('facetrained.yml')

# Initialize the video capture
cap = cv.VideoCapture(0)  # 0 for default camera, change if using a different camera

# Confidence threshold to determine if a face is recognized
threshold = 90  # Adjust this value based on your requirements

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

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
        if confidence < threshold and label < len(people):
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

# Release the video capture and close all windows
cap.release()
cv.destroyAllWindows()
