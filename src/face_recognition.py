import cv2
import face_recognition
import os
import numpy as np

# Define dataset path
dataset_path = "dataset"

# Load registered faces
known_faces = []
known_names = []

# Ensure dataset folder exists
if not os.path.exists(dataset_path):
    print(f"❌ Error: Dataset folder '{dataset_path}' not found!")
    exit()

# Loop through all subfolders (people's names)
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    # Check if it's a folder
    if os.path.isdir(person_folder):
        for filename in os.listdir(person_folder):
            file_path = os.path.join(person_folder, filename)

            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image = face_recognition.load_image_file(file_path)
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        known_faces.append(encodings[0])
                        known_names.append(person_name)  # Assign folder name as person's name
                        print(f"✅ Encoded: {filename} for {person_name}")
                    else:
                        print(f"⚠️ No face found in {filename}, skipping.")

                except Exception as e:
                    print(f"❌ Error loading {filename}: {e}")

# If no faces were found, exit
if not known_faces:
    print("⚠️ No valid faces found in dataset. Please add clear images!")
    exit()

print("\n✅ Dataset Loaded Successfully.")

# Function to recognize faces
def recognize_faces(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        print(f"🎯 Identified: {name}")  # Debugging output

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = recognize_faces(frame)
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
