import cv2
import os
import numpy as np
from PIL import Image
import pickle
import time
import mediapipe as mp

# ============================================
# CONFIGURATION
# ============================================
REGISTERED_FOLDER = "authorized_faces"
ENCODINGS_FILE = "face_encodings.pkl"
GESTURES_FILE = "hand_gestures.pkl"

# Security thresholds
FACE_CONFIDENCE_THRESHOLD = 0.6
OPENCV_THRESHOLD = 0.85

# Timing settings
FACE_VERIFICATION_TIME = 1.0 # THIS IS THE BEST TIME
GESTURE_TIMEOUT = 6.9 # ALSO THE BEST TIME

# GESTURE RECOGNITION SETTINGS
# ============================================
GESTURE_MATCH_THRESHOLD = 0.69  #  MAIN THRESHOLD (0.60-0.95)
GESTURE_DETECTION_CONFIDENCE = 0.5  # Hand detection sensitivity (0.3-0.9)
GESTURE_TRACKING_CONFIDENCE = 0.3   # Hand tracking sensitivity (0.3-0.7)
# ============================================

# ============================================
# FACE RECOGNITION CLASS
# ============================================
class FaceRecognition:
    """Handle all face detection and recognition"""
    
    def __init__(self):
        self.available_methods = self._test_capabilities()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.known_encodings = []
        self.known_names = []
        self._load_encodings()
    
    def _test_capabilities(self):
        """Test available face recognition methods"""
        methods = []
        try:
            import face_recognition
            methods.append('face_recognition')
            test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            face_recognition.face_encodings(test_img)
            methods.append('face_encodings')
            print("✅ face_recognition library available")
        except (ImportError, Exception):
            print("⚠️  face_recognition unavailable, using OpenCV")
        return methods
    
    def _create_encoding(self, rgb_frame):
        """Create face encoding"""
        if 'face_encodings' in self.available_methods:
            try:
                import face_recognition
                encodings = face_recognition.face_encodings(rgb_frame)
                if encodings:
                    return encodings[0]
            except Exception:
                pass
        
        return self._create_histogram_encoding(rgb_frame)
    
    def _create_histogram_encoding(self, rgb_frame):
        """Create histogram-based encoding"""
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_region = rgb_frame[y:y+h, x:x+w]
            face_std = cv2.resize(face_region, (100, 100))
            
            hsv = cv2.cvtColor(face_std, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(face_std, cv2.COLOR_RGB2LAB)
            
            hist_hsv = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist_lab = cv2.calcHist([lab], [0, 1], None, [50, 60], [0, 255, 0, 255])
            
            hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
            hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
            
            return np.concatenate([hist_hsv, hist_lab])
        return None
    
    def _save_encodings(self):
        """Save encodings to disk"""
        data = {
            name: {'encoding': enc, 'type': type(enc).__name__}
            for name, enc in zip(self.known_names, self.known_encodings)
        }
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(data, f)
    
    def _load_encodings(self):
        """Load encodings from disk"""
        if not os.path.exists(ENCODINGS_FILE):
            return
        
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                data = pickle.load(f)
            
            self.known_names = []
            self.known_encodings = []
            
            for name, info in data.items():
                self.known_names.append(name)
                if isinstance(info, dict):
                    self.known_encodings.append(info['encoding'])
                else:
                    self.known_encodings.append(info)
            
            print(f"✅ Loaded {len(self.known_names)} registered faces")
        except Exception as e:
            print(f"❌ Error loading encodings: {e}")
    
    def register_face(self, name=None):
        """Register a new face"""
        if name is None:
            name = input("Enter name: ").strip()
        
        if not name:
            print("❌ Name cannot be empty")
            return False
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return False
        
        print("Position your face in the frame. Press 's' to capture")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Show detection box
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Press 's' to capture", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Register Face", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encoding = self._create_encoding(rgb_frame)
                
                if encoding is not None:
                    # Save image
                    img_path = os.path.join(REGISTERED_FOLDER, f"{name}.jpg")
                    Image.fromarray(rgb_frame).save(img_path, 'JPEG', quality=95)
                    
                    # Save encoding
                    self.known_names.append(name)
                    self.known_encodings.append(encoding)
                    self._save_encodings()
                    
                    print(f"✅ Face registered: {name}")
                    cap.release()
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("❌ No face detected, try again")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    def verify_face_continuous(self, duration=FACE_VERIFICATION_TIME):
        """Verify face continuously for specified duration"""
        if not self.known_encodings:
            print("❌ No registered faces")
            return None, None
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return None, None
        
        print(f"🔍 Verifying face for {duration} seconds...")
        
        start_time = time.time()
        consistent_name = None
        frame_count = 0
        name_counts = {}
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            name, confidence = self._recognize_single_face(rgb_frame)
            
            if name != "Unknown":
                name_counts[name] = name_counts.get(name, 0) + 1
                frame_count += 1
            
            # Draw feedback
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.putText(frame, f"Verifying: {remaining:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detected: {name}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow("Face Verification", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None, None
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Determine most consistent face
        if name_counts:
            consistent_name = max(name_counts, key=name_counts.get)
            consistency = name_counts[consistent_name] / frame_count if frame_count > 0 else 0
            
            # Require at least 70% consistency
            if consistency >= 0.7:
                print(f"✅ Face verified: {consistent_name} ({consistency:.0%} consistent)")
                return consistent_name, consistency
            else:
                print(f"❌ Inconsistent detection ({consistency:.0%})")
                return None, None
        
        print("❌ No face detected")
        return None, None
    
    def _recognize_single_face(self, rgb_frame):
        """Recognize a single face in frame"""
        # Try face_recognition first
        if 'face_encodings' in self.available_methods:
            try:
                import face_recognition
                locations = face_recognition.face_locations(rgb_frame)
                encodings = face_recognition.face_encodings(rgb_frame, locations)
                
                if encodings:
                    encoding = encodings[0]
                    matches = face_recognition.compare_faces(
                        self.known_encodings, encoding,
                        tolerance=1 - FACE_CONFIDENCE_THRESHOLD
                    )
                    distances = face_recognition.face_distance(self.known_encodings, encoding)
                    
                    if len(distances) > 0:
                        best_idx = np.argmin(distances)
                        confidence = 1 - distances[best_idx]
                        
                        if matches[best_idx] and confidence >= FACE_CONFIDENCE_THRESHOLD:
                            return self.known_names[best_idx], confidence
            except Exception:
                pass
        
        # Fallback to OpenCV
        encoding = self._create_histogram_encoding(rgb_frame)
        if encoding is not None:
            return self._compare_encodings(encoding)
        
        return "Unknown", 0.0
    
    def _compare_encodings(self, current_encoding):
        """Compare encoding with known faces"""
        best_name = "Unknown"
        best_conf = 0.0
        
        for name, known_enc in zip(self.known_names, self.known_encodings):
            if isinstance(current_encoding, np.ndarray) and isinstance(known_enc, np.ndarray):
                min_len = min(len(current_encoding), len(known_enc))
                corr = np.corrcoef(current_encoding[:min_len], known_enc[:min_len])[0, 1]
                
                if not np.isnan(corr) and corr > best_conf and corr >= OPENCV_THRESHOLD:
                    best_conf = corr
                    best_name = name
        
        return best_name, best_conf

# ============================================
# HAND GESTURE RECOGNITION CLASS
# ============================================
class HandGestureRecognition:
    """Handle hand gesture detection and recognition"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=GESTURE_DETECTION_CONFIDENCE,
            min_tracking_confidence=GESTURE_TRACKING_CONFIDENCE     
        )
        self.registered_gestures = {}
        self._load_gestures()
    
    def _load_gestures(self):
        """Load registered gestures from disk"""
        if os.path.exists(GESTURES_FILE):
            try:
                with open(GESTURES_FILE, 'rb') as f:
                    self.registered_gestures = pickle.load(f)
                print(f"✅ Loaded {len(self.registered_gestures)} registered gestures")
            except Exception as e:
                print(f"❌ Error loading gestures: {e}")
    
    def _save_gestures(self):
        """Save gestures to disk"""
        with open(GESTURES_FILE, 'wb') as f:
            pickle.dump(self.registered_gestures, f)
    
    def _extract_landmarks(self, hand_landmarks):
        """Extract normalized landmark positions"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def register_gesture(self, person_name):
        """Register a hand gesture for a person"""
        print(f"\n📷 Registering hand gesture for: {person_name}")
        print("Show your gesture and hold it steady. Press 's' to capture")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                
                cv2.putText(frame, "Hand detected! Press 's' to save", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Show your hand gesture", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Register Gesture", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') and results.multi_hand_landmarks:
                landmarks = self._extract_landmarks(results.multi_hand_landmarks[0])
                self.registered_gestures[person_name] = landmarks
                self._save_gestures()
                print(f"✅ Gesture registered for: {person_name}")
                cap.release()
                cv2.destroyAllWindows()
                return True
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    def verify_gesture(self, person_name, timeout=GESTURE_TIMEOUT):
        """Verify gesture matches registered one"""
        if person_name not in self.registered_gestures:
            print(f"❌ No gesture registered for: {person_name}")
            return False
        
        print(f"\n✋ Show your gesture within {timeout} seconds...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return False
        
        start_time = time.time()
        registered_landmarks = self.registered_gestures[person_name]
        
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)
            
            remaining = timeout - (time.time() - start_time)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                
                current_landmarks = self._extract_landmarks(hand_landmarks)
                similarity = self._compare_gestures(current_landmarks, registered_landmarks)
                
                color = (0, 255, 0) if similarity > GESTURE_MATCH_THRESHOLD else (0, 165, 255)
                cv2.putText(frame, f"Match: {similarity:.0%} (Need: {GESTURE_MATCH_THRESHOLD:.0%})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                if similarity > GESTURE_MATCH_THRESHOLD:
                    cv2.putText(frame, "GESTURE MATCHED!", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Gesture Verification", frame)
                    cv2.waitKey(1000)
                    cap.release()
                    cv2.destroyAllWindows()
                    print("✅ Gesture verified!")
                    return True
            
            cv2.putText(frame, f"Time: {remaining:.1f}s", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Gesture Verification", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("❌ Gesture verification failed")
        return False
    
    def _compare_gestures(self, current, registered):
        """Compare two gesture landmarks"""
        try:
            min_len = min(len(current), len(registered))
            current = current[:min_len]
            registered = registered[:min_len]
            
            # Calculate Euclidean distance
            distance = np.linalg.norm(current - registered)
            # Normalize to similarity score (0-1)
            similarity = 1 / (1 + distance)
            return similarity
        except Exception:
            return 0.0

# ============================================
# MAIN SECURITY SYSTEM
# ============================================
class SecuritySystem:
    """Main security system combining face and gesture recognition"""
    
    def __init__(self):
        self.face_recognition = FaceRecognition()
        self.gesture_recognition = HandGestureRecognition()
        print("\n" + "="*60)
        print("  DUAL AUTHENTICATION SECURITY SYSTEM")
        print("  Face Recognition + Hand Gesture Verification")
        print("="*60)
    
    def register_person(self):
        """Register both face and gesture for a person"""
        print("\n--- REGISTRATION PROCESS ---")
        
        # Step 1: Register Face
        print("\n[1/2] Face Registration")
        if not self.face_recognition.register_face():
            print("❌ Face registration failed")
            return False
        
        name = self.face_recognition.known_names[-1]
        
        # Step 2: Register Gesture
        print("\n[2/2] Gesture Registration")
        if not self.gesture_recognition.register_gesture(name):
            print("❌ Gesture registration failed")
            # Remove face registration if gesture fails
            self.face_recognition.known_names.pop()
            self.face_recognition.known_encodings.pop()
            self.face_recognition._save_encodings()
            return False
        
        print(f"\n✅ {name} registered successfully with face and gesture!")
        return True
    
    def authenticate_person(self):
        """Full authentication: face verification + gesture verification"""
        print("\n" + "="*60)
        print("  AUTHENTICATION PROCESS")
        print("="*60)
        
        # Step 1: Verify Face
        print("\n[1/2] Face Verification")
        person_name, confidence = self.face_recognition.verify_face_continuous()
        
        if person_name is None:
            print("\n❌ AUTHENTICATION FAILED: Face not verified")
            return False
        
        print(f"\n✅ Face verified: {person_name}")
        
        # Step 2: Verify Gesture
        print("\n[2/2] Gesture Verification")
        gesture_verified = self.gesture_recognition.verify_gesture(person_name)
        
        if gesture_verified:
            print("\n" + "="*60)
            print(f"  ✅ AUTHENTICATION SUCCESSFUL")
            print(f"  Welcome, {person_name}!")
            print(f"  🚪 ACCESS GRANTED")
            print("="*60)
            
            # TODO: Send signal to Arduino to unlock door
            # self.send_arduino_signal("UNLOCK")
            
            return True
        else:
            print("\n" + "="*60)
            print(f"  ❌ AUTHENTICATION FAILED")
            print(f"  Face verified but gesture did not match")
            print(f"  🚪 ACCESS DENIED")
            print("="*60)
            return False
    
    def send_arduino_signal(self, command):
        """Send signal to Arduino (to be implemented)"""
        # TODO: Implement Arduino serial communication
        # Example:
        # import serial
        # arduino = serial.Serial('COM3', 9600)
        # arduino.write(command.encode())
        print(f"📡 [PLACEHOLDER] Sending to Arduino: {command}")
    
    def list_registered_users(self):
        """List all registered users"""
        print("\n📋 Registered Users:")
        if not self.face_recognition.known_names:
            print("  No users registered")
            return
        
        for name in self.face_recognition.known_names:
            has_gesture = "✅" if name in self.gesture_recognition.registered_gestures else "❌"
            print(f"  • {name} - Gesture: {has_gesture}")

# ============================================
# MAIN PROGRAM
# ============================================
def main():
    # Create necessary folders
    os.makedirs(REGISTERED_FOLDER, exist_ok=True)
    
    system = SecuritySystem()
    
    while True:
        print("\n" + "-"*60)
        print("MENU:")
        print("1. Register New Person (Face + Gesture)")
        print("2. Authenticate Person (Face + Gesture)")
        print("3. List Registered Users")
        print("4. Test Face Recognition Only")
        print("5. Test Gesture Recognition Only")
        print("6. Exit")
        print("-"*60)
        
        choice = input("Choose option (1-6): ").strip()
        
        if choice == "1":
            system.register_person()
        
        elif choice == "2":
            system.authenticate_person()
        
        elif choice == "3":
            system.list_registered_users()
        
        elif choice == "4":
            name, conf = system.face_recognition.verify_face_continuous()
            if name:
                print(f"✅ Recognized: {name} ({conf:.0%})")
            else:
                print("❌ Face not recognized")
        
        elif choice == "5":
            person_name = input("Enter person name: ").strip()
            system.gesture_recognition.verify_gesture(person_name)
        
        elif choice == "6":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    main()