from picamera2 import Picamera2
import cv2
import time


def main():
    # Initialize the camera
    picam2 = Picamera2()

    # Configure the camera
    preview_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)})
    picam2.configure(preview_config)

    # Start the camera
    picam2.start()

    # Allow camera to warm up
    time.sleep(2)

    print("Camera is now running. Press 'q' to quit.")

    while True:
        # Capture a frame
        frame = picam2.capture_array()

        # Display the frame
        cv2.imshow("Camera Test", frame)

        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()