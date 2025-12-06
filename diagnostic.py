# speed_test.py
import cv2
import mediapipe as mp
import time

# Test MediaPipe face detection speed
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
print("Testing optimized face detection...")
print("Press 'q' to quit")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Resize for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    # Detect with MediaPipe
    results = face_detection.process(rgb_small)
    
    # Calculate FPS
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    
    cv2.putText(frame, f"FPS: {fps:.1f} (OPTIMIZED)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Optimized Test', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nAverage FPS: {fps:.1f}")
if fps > 20:
    print("✅ EXCELLENT - Fully optimized!")
elif fps > 15:
    print("✅ GOOD - Well optimized")
else:
    print("⚠️  DECENT - Could optimize more")