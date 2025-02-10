import cv2
import dlib
import face_recognition
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.spatial import distance

# Load Dlib face detector and predictor
MODEL_PATH = os.path.join(os.getcwd(), "models", "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(MODEL_PATH):
    print("❌ Error: shape_predictor_68_face_landmarks.dat not found in models folder!")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# Define eye landmarks
LEFT_EYE = list(range(42, 48))   # Right eye
RIGHT_EYE = list(range(36, 42))  # Left eye

# Blink detection settings
EAR_THRESHOLD = 0.21  # Adjusted for better accuracy
BLINK_FRAMES = 2  # Number of frames required to detect a blink
blink_counter = 0

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_EAR(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

# Load known faces
KNOWN_FACES_DIR = "dataset"
known_encodings = []
known_names = []

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_folder = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)

        if encoding:
            known_encodings.append(encoding[0])
            known_names.append(person_name)

print(f"✅ Loaded {len(known_encodings)} known faces.")

# CSV Attendance File
CSV_FILE = "attendance.csv"

# Initialize video capture
cap = cv2.VideoCapture(0)

# Store attendance to avoid duplicate entries in the same session
attendance_records = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Face detection
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare face encoding with known faces
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if face_distances[best_match_index] < 0.4:  # Stricter threshold
            name = known_names[best_match_index]
        else:
            name = "Unknown"

        # Face Landmark Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE]
            left_EAR = calculate_EAR(left_eye)
            right_EAR = calculate_EAR(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # Improved Blink Detection
            if avg_EAR < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_FRAMES:
                    print(f"✅ {name} blinked! (Liveness detected)")
                    blink_counter = 0  # Reset counter after confirming blink

                    # Save attendance only if not already marked today
                    today_date = datetime.now().strftime("%Y-%m-%d")
                    if name != "Unknown" and name not in attendance_records:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        attendance_records[name] = now

                        # Append to CSV file instead of overwriting
                        df = pd.DataFrame([[name, now]], columns=["Name", "Timestamp"])
                        df.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)

                        print(f"📌 Attendance marked for {name} at {now}")

        # Display name on video
        cv2.rectangle(frame, (left*2, top*2), (right*2, bottom*2), (0, 255, 0), 2)
        cv2.putText(frame, name, (left*2, top*2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
