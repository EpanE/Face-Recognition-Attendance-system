import cv2 as cv
print(cv.__version__)
try:
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    print("Face module is properly installed")
except AttributeError:
    print("Face module is not properly installed")