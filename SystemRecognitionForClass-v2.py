import cv2 as cv
import numpy as np
import pandas as pd
import os
from tkinter import *
from PIL import Image, ImageTk
from datetime import datetime

# Load the list of people from the Excel file
people_df = pd.read_excel('people_list.xlsx')
people = people_df['Name'].tolist()

# Initialize Excel file for attendance logging
attendance_file = 'new_attendance.xlsx'

if not os.path.exists(attendance_file):
    with pd.ExcelWriter(attendance_file, engine='openpyxl') as writer:
        pd.DataFrame(columns=['Name', 'Date', 'Time', 'Status']).to_excel(writer, sheet_name='Attendance Log',
                                                                          index=False)
        pd.DataFrame(columns=['Date', 'Time', 'Image Path']).to_excel(writer, sheet_name='Unknown Log', index=False)

# Load the Haar Cascade classifier for face detection
haar_cascade = cv.CascadeClassifier('haar_face.xml')

# Create and load the face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('facetrained.yml')

# Set confidence threshold
threshold = 85

# GUI Setup
root = Tk()
root.title("Face Recognition Attendance System")
root.geometry("800x600")

# Video Capture
cap = cv.VideoCapture(0)

# Tkinter Label for the video frame
video_label = Label(root)
video_label.pack()

# Tkinter Textbox for logs
log_text = Text(root, height=15, width=90)
log_text.pack()

def update_camera_view(frame):
    img = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.update()

def recognize_faces():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces_rect:
            faces_roi = gray[y:y + h, x:x + w]
            label, confidence = face_recognizer.predict(faces_roi)

            if confidence < threshold and label < len(people):
                name = people[label]
                status = 'Present'
                log_entry = f"{name} recognized at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Confidence: {confidence:.2f}"
                log_text.insert(END, log_entry + '\n')
                mark_attendance(name, 'Present')
            else:
                name = "Unknown"
                log_entry = f"Unknown face detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                log_text.insert(END, log_entry + '\n')
                save_unknown_face(frame, x, y, w, h)

            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, f'{name} ({confidence:.2f})', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Update the camera view
        update_camera_view(frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def mark_attendance(name, status):
    current_time = datetime.now().strftime('%H:%M:%S')
    current_date = datetime.now().strftime('%Y-%m-%d')
    new_entry = pd.DataFrame([[name, current_date, current_time, status]], columns=['Name', 'Date', 'Time', 'Status'])

    with pd.ExcelWriter(attendance_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        new_entry.to_excel(writer, sheet_name='Attendance Log', index=False, header=False)

def save_unknown_face(frame, x, y, w, h):
    cropped_face = frame[y:y + h, x:x + w]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_name = f"unknown_{timestamp}.png"
    file_path = os.path.join('unknown_faces', file_name)

    if not os.path.exists('unknown_faces'):
        os.makedirs('unknown_faces')

    cv.imwrite(file_path, cropped_face)
    current_time = datetime.now().strftime('%H:%M:%S')
    current_date = datetime.now().strftime('%Y-%m-%d')
    new_entry = pd.DataFrame([[current_date, current_time, file_path]], columns=['Date', 'Time', 'Image Path'])

    with pd.ExcelWriter(attendance_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        new_entry.to_excel(writer, sheet_name='Unknown Log', index=False, header=False)

start_button = Button(root, text="Start Recognition", command=recognize_faces)
start_button.pack()

root.mainloop()