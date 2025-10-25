import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
from PIL import Image, ImageTk


class FaceRecognitionSystem:
    def __init__(self, master):
        self.master = master
        self.master.title("Face Recognition System")
        self.master.geometry("800x600")

        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill='both')

        # Create tabs
        self.home_frame = ttk.Frame(self.notebook)
        self.capture_frame = ttk.Frame(self.notebook)
        self.training_frame = ttk.Frame(self.notebook)
        self.recognition_frame = ttk.Frame(self.notebook)

        self.notebook.add(self.home_frame, text='Home')
        self.notebook.add(self.capture_frame, text='Capture Images')
        self.notebook.add(self.training_frame, text='Train Model')
        self.notebook.add(self.recognition_frame, text='Face Recognition')

        self.setup_home_tab()
        self.setup_capture_tab()
        self.setup_training_tab()
        self.setup_recognition_tab()

        # Initialize variables
        self.cap = None
        self.current_frame = None
        self.face_cascade = cv2.CascadeClassifier('haar_face.xml')
        self.people_list = self.load_people_list()

    def setup_home_tab(self):
        label = ttk.Label(self.home_frame, text="Welcome to Face Recognition System", font=("Arial", 24))
        label.pack(pady=20)

        info_text = ("This system allows you to:\n"
                     "1. Capture images for new people\n"
                     "2. Train the face recognition model\n"
                     "3. Perform real-time face recognition")
        info_label = ttk.Label(self.home_frame, text=info_text, font=("Arial", 14))
        info_label.pack(pady=20)

    def setup_capture_tab(self):
        self.name_var = tk.StringVar()
        ttk.Label(self.capture_frame, text="Enter name:").pack(pady=10)
        ttk.Entry(self.capture_frame, textvariable=self.name_var).pack(pady=10)

        self.start_capture_btn = ttk.Button(self.capture_frame, text="Start Capture", command=self.start_capture)
        self.start_capture_btn.pack(pady=10)

        self.stop_capture_btn = ttk.Button(self.capture_frame, text="Stop Capture", command=self.stop_capture,
                                           state=tk.DISABLED)
        self.stop_capture_btn.pack(pady=10)

        self.capture_label = ttk.Label(self.capture_frame)
        self.capture_label.pack(pady=10)

    def setup_training_tab(self):
        self.train_btn = ttk.Button(self.training_frame, text="Train Model", command=self.train_model)
        self.train_btn.pack(pady=20)

        self.training_status = ttk.Label(self.training_frame, text="")
        self.training_status.pack(pady=10)

    def setup_recognition_tab(self):
        self.start_recognition_btn = ttk.Button(self.recognition_frame, text="Start Recognition",
                                                command=self.start_recognition)
        self.start_recognition_btn.pack(pady=10)

        self.stop_recognition_btn = ttk.Button(self.recognition_frame, text="Stop Recognition",
                                               command=self.stop_recognition, state=tk.DISABLED)
        self.stop_recognition_btn.pack(pady=10)

        self.recognition_label = ttk.Label(self.recognition_frame)
        self.recognition_label.pack(pady=10)

    def load_people_list(self):
        if os.path.exists('people_list.xlsx'):
            df = pd.read_excel('people_list.xlsx')
            return df['Name'].tolist()
        return []

    def save_people_list(self):
        df = pd.DataFrame({'Name': self.people_list})
        df.to_excel('people_list.xlsx', index=False)

    def start_capture(self):
        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Error", "Please enter a name.")
            return

        if name not in self.people_list:
            self.people_list.append(name)
            self.save_people_list()

        self.capture_dir = os.path.join("dataset", name)
        os.makedirs(self.capture_dir, exist_ok=True)

        self.cap = cv2.VideoCapture(0)
        self.capture_count = 0
        self.start_capture_btn.config(state=tk.DISABLED)
        self.stop_capture_btn.config(state=tk.NORMAL)
        self.capture_images()

    def capture_images(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if self.capture_count < 100:
                    img_name = os.path.join(self.capture_dir, f"{self.name_var.get()}_{self.capture_count}.jpg")
                    cv2.imwrite(img_name, gray[y:y + h, x:x + w])
                    self.capture_count += 1

            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = Image.fromarray(self.current_frame)
            self.current_frame = ImageTk.PhotoImage(self.current_frame)
            self.capture_label.config(image=self.current_frame)
            self.capture_label.image = self.current_frame

        if self.capture_count < 100:
            self.master.after(10, self.capture_images)
        else:
            self.stop_capture()
            messagebox.showinfo("Capture Complete", f"Captured 100 images for {self.name_var.get()}")

    def stop_capture(self):
        if self.cap is not None:
            self.cap.release()
        self.start_capture_btn.config(state=tk.NORMAL)
        self.stop_capture_btn.config(state=tk.DISABLED)
        self.capture_label.config(image='')

        # Ensure only 100 images are kept
        image_files = sorted(os.listdir(self.capture_dir))
        for file in image_files[100:]:
            os.remove(os.path.join(self.capture_dir, file))

    def train_model(self):
        self.training_status.config(text="Training in progress...")
        self.master.update()

        people = self.load_people_list()
        features = []
        labels = []

        for person in people:
            path = os.path.join("dataset", person)
            label = people.index(person)

            for img in os.listdir(path)[:100]:  # Limit to 100 images
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img_array is not None:
                    faces_rect = self.face_cascade.detectMultiScale(img_array, scaleFactor=1.1, minNeighbors=4)

                    for (x, y, w, h) in faces_rect:
                        faces_roi = img_array[y:y + h, x:x + w]
                        features.append(faces_roi)
                        labels.append(label)

        features = np.array(features, dtype='object')
        labels = np.array(labels)

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(features, labels)

        face_recognizer.save('facetrained.yml')
        np.save('features.npy', features)
        np.save('labels.npy', labels)

        self.training_status.config(text="Training completed!")

    def start_recognition(self):
        self.people = self.load_people_list()
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_recognizer.read('facetrained.yml')

        self.cap = cv2.VideoCapture(0)
        self.start_recognition_btn.config(state=tk.DISABLED)
        self.stop_recognition_btn.config(state=tk.NORMAL)
        self.perform_recognition()

    def perform_recognition(self):
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_rect = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y + h, x:x + w]
                label, confidence = self.face_recognizer.predict(faces_roi)

                if confidence < 100:
                    name = self.people[label]
                    cv2.putText(frame, f'{name} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)
                else:
                    name = "Unknown"
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame = Image.fromarray(self.current_frame)
            self.current_frame = ImageTk.PhotoImage(self.current_frame)
            self.recognition_label.config(image=self.current_frame)
            self.recognition_label.image = self.current_frame

        self.master.after(10, self.perform_recognition)

    def stop_recognition(self):
        if self.cap is not None:
            self.cap.release()
        self.start_recognition_btn.config(state=tk.NORMAL)
        self.stop_recognition_btn.config(state=tk.DISABLED)
        self.recognition_label.config(image='')


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionSystem(root)
    root.mainloop()