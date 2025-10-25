import os
import cv2
from deepface import DeepFace
import numpy as np
import json
from datetime import datetime
import shutil


class FaceTrainingSystem:
    def __init__(self, database_path="face_database"):
        """
        Initialize the face training system

        Parameters:
        database_path (str): Path to store face database
        """
        self.database_path = database_path
        self.metadata_file = os.path.join(database_path, "metadata.json")
        self.setup_database()

    def setup_database(self):
        """Create necessary directories and metadata file"""
        if not os.path.exists(self.database_path):
            os.makedirs(self.database_path)

        if not os.path.exists(self.metadata_file):
            self.save_metadata({})

    def save_metadata(self, metadata):
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)

    def load_metadata(self):
        """Load metadata from JSON file"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}

    def capture_faces(self, person_name, num_samples=5):
        """
        Capture face samples using webcam

        Parameters:
        person_name (str): Name of the person being registered
        num_samples (int): Number of face samples to capture
        """
        person_dir = os.path.join(self.database_path, person_name)
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)

        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        samples_captured = 0

        print(f"Starting capture for {person_name}. Press 'c' to capture, 'q' to quit.")

        while samples_captured < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            # Display counter
            display_frame = frame.copy()
            cv2.putText(display_frame, f"Samples: {samples_captured}/{num_samples}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Draw rectangle around detected face
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.imshow('Capture Face Samples', display_frame)
            key = cv2.waitKey(1)

            # Capture image when 'c' is pressed and face is detected
            if key == ord('c') and len(faces) > 0:
                sample_path = os.path.join(person_dir, f"sample_{samples_captured}.jpg")
                cv2.imwrite(sample_path, frame)
                samples_captured += 1
                print(f"Captured sample {samples_captured}/{num_samples}")

            # Quit if 'q' is pressed
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Update metadata
        metadata = self.load_metadata()
        metadata[person_name] = {
            "samples": samples_captured,
            "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.save_metadata(metadata)

        return samples_captured

    def register_face_from_image(self, person_name, image_path):
        """
        Register a face from an existing image file

        Parameters:
        person_name (str): Name of the person
        image_path (str): Path to the image file
        """
        try:
            # Verify that the image contains a face
            DeepFace.detect_face(image_path)

            # Create person directory if it doesn't exist
            person_dir = os.path.join(self.database_path, person_name)
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)

            # Copy image to person's directory
            filename = f"sample_{len(os.listdir(person_dir))}.jpg"
            dest_path = os.path.join(person_dir, filename)
            shutil.copy2(image_path, dest_path)

            # Update metadata
            metadata = self.load_metadata()
            if person_name not in metadata:
                metadata[person_name] = {
                    "samples": 1,
                    "date_added": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                metadata[person_name]["samples"] += 1
            self.save_metadata(metadata)

            return True

        except Exception as e:
            print(f"Error registering face: {str(e)}")
            return False

    def verify_training(self, person_name):
        """
        Verify that the training samples are good quality

        Parameters:
        person_name (str): Name of the person to verify

        Returns:
        bool: True if verification passes
        """
        person_dir = os.path.join(self.database_path, person_name)
        if not os.path.exists(person_dir):
            print(f"No training data found for {person_name}")
            return False

        samples = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if len(samples) < 2:
            print(f"Not enough samples for {person_name}")
            return False

        # Verify that faces in samples match each other
        base_sample = os.path.join(person_dir, samples[0])
        for sample in samples[1:]:
            sample_path = os.path.join(person_dir, sample)
            try:
                result = DeepFace.verify(img1_path=base_sample,
                                         img2_path=sample_path,
                                         enforce_detection=False)
                if not result['verified']:
                    print(f"Sample {sample} doesn't match the base sample")
                    return False
            except Exception as e:
                print(f"Error verifying sample {sample}: {str(e)}")
                return False

        return True

    def list_registered_people(self):
        """
        List all registered people and their details

        Returns:
        dict: Dictionary containing registration details
        """
        return self.load_metadata()


# Example usage
if __name__ == "__main__":
    trainer = FaceTrainingSystem()

    # Example 1: Capture faces using webcam
    person_name = "John_Doe"
    samples = trainer.capture_faces(person_name, num_samples=5)
    print(f"Captured {samples} samples for {person_name}")

    # Example 2: Register face from existing image
    image_path = "path/to/your/image.jpg"
    if trainer.register_face_from_image("Jane_Doe", image_path):
        print("Successfully registered face from image")

    # Verify training
    if trainer.verify_training("John_Doe"):
        print("Training verification passed")

    # List all registered people
    registered_people = trainer.list_registered_people()
    print("\nRegistered people:")
    for person, details in registered_people.items():
        print(f"{person}: {details['samples']} samples, added on {details['date_added']}")