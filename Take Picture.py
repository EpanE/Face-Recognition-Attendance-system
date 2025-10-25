import cv2
import os
import re

def get_latest_number(directory):
    pattern = re.compile(r'Azim_(\d+)\.jpg')
    max_number = -1
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)
    return max_number + 1

def get_unique_filename(directory, number):
    filename = f"Azim_{number}.jpg"
    return os.path.join(directory, filename)

save_directory = "dataset/Azim"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

clicknum = get_latest_number(save_directory)

while True:
    ret, frame = cap.read()

    if ret:
        cv2.imshow("Webcam Feed", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            filename = get_unique_filename(save_directory, clicknum)
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            clicknum += 1

        elif key == ord('q'):
            break
    else:
        print("Error: Could not capture image.")
        break

cap.release()
cv2.destroyAllWindows()