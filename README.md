# Face Recognition Attendance System

This repository contains a collection of Python scripts for experimenting with a camera-based attendance system powered by OpenCV face recognition.  It combines utilities for capturing training images, training an LBPH face recognizer, and running a live attendance check that labels recognized faces in real time.

## Project structure

- `dataset/` – Sample training images grouped per person.
- `features.npy` / `labels.npy` – Feature vectors and labels produced during training.
- `facetrained.yml` – Saved Local Binary Pattern Histogram (LBPH) classifier.
- `haar_face.xml` – Haar cascade used for face detection.
- `LiveVid_FaceRecognition.py` – Run live recognition with the trained model.
- `Combined_TakePic_Train_Livevid*.py`, `Training.py`, `SystemRecognitionForClass*.py` – Additional utilities for capturing images, training the recognizer, and integrating the workflow into attendance tracking scripts.

## Requirements

- Python 3.8+
- [OpenCV](https://opencv.org/) with `opencv-contrib-python` for the LBPH face recognizer.
- NumPy
- (Optional) Pandas for scripts that interact with Excel workbooks.

Install the dependencies into your environment:

```bash
pip install opencv-contrib-python numpy pandas
```

## Usage

1. **Capture training images** – Use one of the `Take Picture` or `Combined_TakePic_Train_Livevid` scripts to add new faces to `dataset/`.
2. **Train the recognizer** – Run `Training.py` to generate `features.npy`, `labels.npy`, and `facetrained.yml`.
3. **Run live recognition** – Execute `LiveVid_FaceRecognition.py` to detect faces from a connected camera.  Press `q` to exit the live feed.

Adjust camera indices (`cv.VideoCapture(0)`/`(1)`) and update the `people` list in the recognition scripts to match your dataset.

## Notes

- The repository includes example attendance workbooks (`attendance.xlsx`, `new_attendance.xlsx`) used by some scripts for recording recognized attendees.
- Ensure your environment has access to a camera when running the live recognition scripts.
- The Haar cascade and LBPH thresholds may need fine tuning for your environment and lighting conditions.
