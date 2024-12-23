import cv2
import time
import os

# Load the face and smile classifiers
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# Initialize video capture
video_capture = cv2.VideoCapture("C:\\Users\\user\\Desktop\\Edgematrix\\py files\\facedetection.mp4")

# Create output directory for smile photos
output_dir = 'smile_photos'
os.makedirs(output_dir, exist_ok=True)

# Variables to manage smile detection duration
start_time = None
smile_duration = 2  # Duration to capture smiles in seconds

# Function to detect smiles
def detect_smile(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    smile_detected = False

    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y + h, x:x + w]
        roi_color = vid[y:y + h, x:x + w]
        smiles = smile_classifier.detectMultiScale(roi_gray, 1.7, 20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
            smile_detected = True

    return smile_detected

# Main loop for smile detection
while True:
    result, video_frame = video_capture.read()

    if not result:
        print("Error: Could not read frame.")
        break

    smile_detected = detect_smile(video_frame)

    if smile_detected:
        if start_time is None:
            start_time = time.time()

        elapsed_time = time.time() - start_time

        if elapsed_time <= smile_duration:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(output_dir, f"smile_{timestamp}.jpg")
            cv2.imwrite(filename, video_frame)
        else:
            start_time = None  # Reset start_time after duration is reached

    # Display the video frame
    cv2.imshow("Smile Detection Project", video_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
