import cv2 as cv

capture = cv.VideoCapture(0)
# Usually the inside of this function represent number which for camera
# 0 = webcam
# 1 = First Camera that is connected to your device
# It can also display video by giving it the path for the video

# This is where there is difference between reading video and picture
# To run a video we must run the video frame by frame
# So a loop must be use
while True:
    isTrue, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Video', frame)
    cv.imshow('Video Gray', gray)

    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

    print(f'Number of faces found = {len(faces_rect)}')

    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

    cv.imshow('Detected Faces', frame)

    if cv.waitKey(20) & 0xFF==ord('d'): # break out of loop when key 'd' is pressed
        break

capture.release()
cv.destroyAllWindow()

# There is possible error and it is -215 error
# This error occurs when there are no frame to be grab







