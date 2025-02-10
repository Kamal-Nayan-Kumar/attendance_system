import cv2
import dlib
import os
from scipy.spatial import distance

# Load face detector and landmark predictor from dlib
MODEL_PATH = os.path.join(os.getcwd(), "models", "shape_predictor_68_face_landmarks.dat")

if not os.path.exists(MODEL_PATH):
    print("❌ Error: shape_predictor_68_face_landmarks.dat not found in models folder!")
    exit()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# Define eye landmarks
LEFT_EYE = list(range(42, 48))   # Right eye in the image
RIGHT_EYE = list(range(36, 42))  # Left eye in the image

# Eye Aspect Ratio (EAR) calculation function
def calculate_EAR(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

# Define eye blink threshold
EAR_THRESHOLD = 0.25  # Adjust based on testing
BLINK_CONSEC_FRAMES = 3  # Number of consecutive frames to confirm blink

blink_count = 0
frame_counter = 0

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Extract eye coordinates
        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE]

        # Compute EAR for both eyes
        left_EAR = calculate_EAR(left_eye)
        right_EAR = calculate_EAR(right_eye)

        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Draw eye landmarks
        for (x, y) in left_eye + right_eye:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        # Detect blink
        if avg_EAR < EAR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= BLINK_CONSEC_FRAMES:
                blink_count += 1
                print("👀 Blink Detected!")
            frame_counter = 0  # Reset counter

        # Display EAR on screen
        cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {blink_count}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow("Eye Blink Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
