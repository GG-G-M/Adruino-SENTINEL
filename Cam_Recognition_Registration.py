import cv2
import face_recognition
import os
import numpy as np

# Folders
REGISTERED_FOLDER = "authorized_faces"

if not os.path.exists(REGISTERED_FOLDER):
    os.makedirs(REGISTERED_FOLDER)

def register_face():
    name = input("Enter your name: ").strip()
    cap = cv2.VideoCapture(0)
    print("Press 's' to take a snapshot, 'q' to quit registration")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Save face image
            face_image_path = os.path.join(REGISTERED_FOLDER, f"{name}.jpg")
            cv2.imwrite(face_image_path, frame)
            print(f"Face registered and saved as {face_image_path}")
            break
        elif key == ord('q'):
            print("Registration cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()

def recognize_faces():
    # Load known faces
    known_face_encodings = []
    known_face_names = []

    for filename in os.listdir(REGISTERED_FOLDER):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(REGISTERED_FOLDER, filename))
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])

    # Start webcam
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit recognition")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            top, right, bottom, left = face_location
            top *= 4; right *= 4; bottom *= 4; left *= 4

            # Draw box and name
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Trigger action for authorized person
            if name != "Unknown":
                print(f"Authorized: {name}")
                # Here you can call your codec function or any action
                # e.g., call_codec(name)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Choose mode:")
    print("1. Register face")
    print("2. Recognize faces")
    mode = input("Enter 1 or 2: ").strip()

    if mode == "1":
        register_face()
    elif mode == "2":
        recognize_faces()
    else:
        print("Invalid option")

if __name__ == "__main__":
    main()
