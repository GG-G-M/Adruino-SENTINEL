import cv2
import os
import numpy as np
from PIL import Image

REGISTERED_FOLDER = "authorized_faces"

def simple_face_register():
    """Simple face registration without encoding"""
    name = input("Enter your name: ").strip()
    cap = cv2.VideoCapture(0)
    
    print("Press 's' to capture, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        cv2.imshow("Register Face", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # Save image
            if not os.path.exists(REGISTERED_FOLDER):
                os.makedirs(REGISTERED_FOLDER)
            
            filename = os.path.join(REGISTERED_FOLDER, f"{name}.jpg")
            cv2.imwrite(filename, frame)
            print(f"✅ Saved: {filename}")
            break
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def simple_face_detection():
    """Simple face detection without recognition"""
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face Detected", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Simple menu
print("1. Register Face")
print("2. Detect Faces")
choice = input("Choose: ")

if choice == "1":
    simple_face_register()
elif choice == "2":
    simple_face_detection()