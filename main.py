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
OPENCV_THRESHOLD = 0.75

# Timing settings
FACE_VERIFICATION_TIME = 5.0
GESTURE_TIMEOUT = 6.9

# GESTURE RECOGNITION SETTINGS
GESTURE_MATCH_THRESHOLD = 0.69
GESTURE_DETECTION_CONFIDENCE = 0.5
GESTURE_TRACKING_CONFIDENCE = 0.3

# Multi-sample settings
SAMPLES_PER_PERSON = 3  # Number of images to capture per person

# ============================================
# IMAGE PREPROCESSING FOR LIGHTING COMPENSATION
# ============================================
def normalize_lighting(image):
    """Apply multiple lighting normalization techniques"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    lab = cv2.merge([l, a, b])
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return normalized

def enhance_contrast(image):
    """Enhanced contrast normalization"""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    y = cv2.equalizeHist(y)
    
    ycrcb = cv2.merge([y, cr, cb])
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    
    return enhanced

def gamma_correction(image, gamma=1.2):
    """Apply gamma correction for brightness adjustment"""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def preprocess_face(image):
    """Complete preprocessing pipeline"""
    normalized = normalize_lighting(image)
    enhanced = enhance_contrast(normalized)
    
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 80:
        enhanced = gamma_correction(enhanced, gamma=0.8)
    elif mean_brightness > 180:
        enhanced = gamma_correction(enhanced, gamma=1.3)
    
    return enhanced

# ============================================
# FACE RECOGNITION CLASS
# ============================================
class FaceRecognition:
    """Handle all face detection and recognition with multiple samples support"""
    
    def __init__(self):
        self.available_methods = self._test_capabilities()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Changed structure: name -> list of encodings
        self.known_encodings = {}  # {name: [encoding1, encoding2, encoding3]}
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
            print("⚠️  face_recognition unavailable, using OpenCV with lighting compensation")
        return methods
    
    def _create_encoding(self, rgb_frame):
        """Create face encoding with lighting normalization"""
        preprocessed = preprocess_face(rgb_frame)
        
        if 'face_encodings' in self.available_methods:
            try:
                import face_recognition
                encodings = face_recognition.face_encodings(preprocessed)
                if encodings:
                    return encodings[0]
            except Exception:
                pass
        
        return self._create_histogram_encoding(preprocessed)
    
    def _create_histogram_encoding(self, rgb_frame):
        """Create enhanced histogram-based encoding with lighting compensation"""
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_region = rgb_frame[y:y+h, x:x+w]
            face_std = cv2.resize(face_region, (100, 100))
            
            hsv = cv2.cvtColor(face_std, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(face_std, cv2.COLOR_RGB2LAB)
            gray_face = cv2.cvtColor(face_std, cv2.COLOR_RGB2GRAY)
            
            hist_hsv = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist_lab = cv2.calcHist([lab], [1, 2], None, [50, 60], [0, 255, 0, 255])
            
            lbp = self._calculate_lbp(gray_face)
            hist_lbp = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            
            hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
            hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
            hist_lbp = cv2.normalize(hist_lbp, hist_lbp).flatten()
            
            return np.concatenate([hist_hsv * 0.3, hist_lab * 0.4, hist_lbp * 0.3])
        return None
    
    def _calculate_lbp(self, gray_image):
        """Calculate Local Binary Pattern"""
        lbp = np.zeros_like(gray_image)
        for i in range(1, gray_image.shape[0] - 1):
            for j in range(1, gray_image.shape[1] - 1):
                center = gray_image[i, j]
                code = 0
                code |= (gray_image[i-1, j-1] >= center) << 7
                code |= (gray_image[i-1, j] >= center) << 6
                code |= (gray_image[i-1, j+1] >= center) << 5
                code |= (gray_image[i, j+1] >= center) << 4
                code |= (gray_image[i+1, j+1] >= center) << 3
                code |= (gray_image[i+1, j] >= center) << 2
                code |= (gray_image[i+1, j-1] >= center) << 1
                code |= (gray_image[i, j-1] >= center) << 0
                lbp[i, j] = code
        return lbp
    
    def _save_encodings(self):
        """Save encodings to disk"""
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(self.known_encodings, f)
    
    def _load_encodings(self):
        """Load encodings from disk"""
        if not os.path.exists(ENCODINGS_FILE):
            return
        
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                self.known_encodings = pickle.load(f)
            
            # Handle old format (list) and convert to new format (dict)
            if isinstance(self.known_encodings, dict):
                # Check if it's old dict format {name: encoding}
                if self.known_encodings and not isinstance(list(self.known_encodings.values())[0], list):
                    # Convert old format to new format
                    old_data = self.known_encodings.copy()
                    self.known_encodings = {}
                    for name, enc_data in old_data.items():
                        if isinstance(enc_data, dict) and 'encoding' in enc_data:
                            self.known_encodings[name] = [enc_data['encoding']]
                        else:
                            self.known_encodings[name] = [enc_data]
                    self._save_encodings()
                    print("✅ Converted old format to multi-sample format")
            
            total_samples = sum(len(encs) for encs in self.known_encodings.values())
            print(f"✅ Loaded {len(self.known_encodings)} people with {total_samples} total samples")
            
        except Exception as e:
            print(f"❌ Error loading encodings: {e}")
            self.known_encodings = {}
    
    def register_face(self, name=None, num_samples=SAMPLES_PER_PERSON):
        """Register multiple face samples for a person"""
        if name is None:
            name = input("Enter name: ").strip()
        
        if not name:
            print("❌ Name cannot be empty")
            return False
        
        # Check if person already exists
        if name in self.known_encodings:
            print(f"⚠️  '{name}' already registered with {len(self.known_encodings[name])} samples")
            choice = input("Add more samples? (y/n): ").strip().lower()
            if choice != 'y':
                return False
        else:
            self.known_encodings[name] = []
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return False
        
        captured_count = 0
        current_samples = len(self.known_encodings[name])
        
        print(f"\n📸 Capturing {num_samples} samples for '{name}'")
        print("💡 TIP: Change your position/angle slightly between captures for better recognition")
        print("Press 's' to capture each sample")
        
        while captured_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preprocessed = preprocess_face(rgb_frame)
            preview = cv2.cvtColor(preprocessed, cv2.COLOR_RGB2BGR)
            
            gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.rectangle(preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            combined = np.hstack([frame, preview])
            
            # Status text
            status = f"Sample {captured_count + 1}/{num_samples} - Press 's' to capture"
            cv2.putText(combined, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "Original", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(combined, "Processed", (frame.shape[1] + 10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if captured_count > 0:
                cv2.putText(combined, f"✓ {captured_count} captured", (10, combined.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Register Face - Multiple Samples", combined)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                encoding = self._create_encoding(rgb_frame)
                
                if encoding is not None:
                    # Save image with sample number
                    img_path = os.path.join(REGISTERED_FOLDER, 
                                          f"{name}_sample{current_samples + captured_count + 1}.jpg")
                    Image.fromarray(rgb_frame).save(img_path, 'JPEG', quality=95)
                    
                    # Add encoding to list
                    self.known_encodings[name].append(encoding)
                    captured_count += 1
                    
                    print(f"  ✅ Captured sample {captured_count}/{num_samples}")
                    time.sleep(0.5)  # Brief pause between captures
                    
                    if captured_count >= num_samples:
                        self._save_encodings()
                        print(f"\n✅ All {num_samples} samples registered for '{name}'!")
                        print(f"   Total samples for {name}: {len(self.known_encodings[name])}")
                        cap.release()
                        cv2.destroyAllWindows()
                        return True
                else:
                    print("  ❌ No face detected, try again")
            
            elif key == ord('q'):
                print("\n❌ Registration cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    def verify_face_continuous(self, duration=FACE_VERIFICATION_TIME):
        """Verify face continuously with multiple sample support"""
        if not self.known_encodings:
            print("❌ No registered faces")
            return None, None
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return None, None
        
        print(f"🔍 Verifying face for {duration} seconds...")
        print("💡 Lighting compensation active - works in various lighting conditions!")
        
        start_time = time.time()
        frame_count = 0
        name_counts = {}
        confidence_scores = {}
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            name, confidence = self._recognize_single_face(rgb_frame)
            
            if name != "Unknown":
                name_counts[name] = name_counts.get(name, 0) + 1
                if name not in confidence_scores:
                    confidence_scores[name] = []
                confidence_scores[name].append(confidence)
                frame_count += 1
            
            elapsed = time.time() - start_time
            remaining = duration - elapsed
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_text = "Dark" if brightness < 80 else "Bright" if brightness > 180 else "Normal"
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.putText(frame, f"Verifying: {remaining:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Detected: {name} ({confidence:.0%})", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Lighting: {brightness_text}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("Face Verification", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None, None
        
        cap.release()
        cv2.destroyAllWindows()
        
        if name_counts:
            consistent_name = max(name_counts, key=name_counts.get)
            consistency = name_counts[consistent_name] / frame_count if frame_count > 0 else 0
            avg_confidence = np.mean(confidence_scores[consistent_name])
            
            if consistency >= 0.6:
                print(f"✅ Face verified: {consistent_name} ({consistency:.0%} consistent, {avg_confidence:.0%} avg confidence)")
                return consistent_name, avg_confidence
            else:
                print(f"❌ Inconsistent detection ({consistency:.0%})")
                return None, None
        
        print("❌ No face detected")
        return None, None
    
    def _recognize_single_face(self, rgb_frame):
        """Recognize face by comparing with all samples"""
        preprocessed = preprocess_face(rgb_frame)
        
        if 'face_encodings' in self.available_methods:
            try:
                import face_recognition
                locations = face_recognition.face_locations(preprocessed)
                encodings = face_recognition.face_encodings(preprocessed, locations)
                
                if encodings:
                    encoding = encodings[0]
                    
                    best_name = "Unknown"
                    best_confidence = 0.0
                    
                    # Compare with all samples of all people
                    for name, sample_encodings in self.known_encodings.items():
                        for sample_enc in sample_encodings:
                            distance = face_recognition.face_distance([sample_enc], encoding)[0]
                            confidence = 1 - distance
                            
                            if confidence > best_confidence and confidence >= FACE_CONFIDENCE_THRESHOLD:
                                best_confidence = confidence
                                best_name = name
                    
                    if best_name != "Unknown":
                        return best_name, best_confidence
                        
            except Exception:
                pass
        
        # Fallback to OpenCV
        encoding = self._create_histogram_encoding(preprocessed)
        if encoding is not None:
            return self._compare_encodings(encoding)
        
        return "Unknown", 0.0
    
    def _compare_encodings(self, current_encoding):
        """Compare with all samples using multiple metrics"""
        best_name = "Unknown"
        best_conf = 0.0
        
        for name, sample_encodings in self.known_encodings.items():
            # Compare with all samples and take the best match
            for sample_enc in sample_encodings:
                if isinstance(current_encoding, np.ndarray) and isinstance(sample_enc, np.ndarray):
                    min_len = min(len(current_encoding), len(sample_enc))
                    
                    corr = np.corrcoef(current_encoding[:min_len], sample_enc[:min_len])[0, 1]
                    
                    dot_product = np.dot(current_encoding[:min_len], sample_enc[:min_len])
                    norm_product = np.linalg.norm(current_encoding[:min_len]) * np.linalg.norm(sample_enc[:min_len])
                    cosine_sim = dot_product / (norm_product + 1e-8)
                    
                    combined_score = (corr * 0.6 + cosine_sim * 0.4) if not np.isnan(corr) else cosine_sim
                    
                    if combined_score > best_conf and combined_score >= OPENCV_THRESHOLD:
                        best_conf = combined_score
                        best_name = name
        
        return best_name, best_conf
    
    def delete_person(self, name):
        """Delete a person's face data"""
        if name in self.known_encodings:
            # Delete images
            for file in os.listdir(REGISTERED_FOLDER):
                if file.startswith(name):
                    os.remove(os.path.join(REGISTERED_FOLDER, file))
            
            # Delete encodings
            del self.known_encodings[name]
            self._save_encodings()
            print(f"✅ Deleted all data for '{name}'")
            return True
        else:
            print(f"❌ Person '{name}' not found")
            return False

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
        
        if person_name in self.registered_gestures:
            print(f"⚠️  Gesture already exists for '{person_name}'")
            choice = input("Replace existing gesture? (y/n): ").strip().lower()
            if choice != 'y':
                return False
        
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
            
            distance = np.linalg.norm(current - registered)
            similarity = 1 / (1 + distance)
            return similarity
        except Exception:
            return 0.0
    
    def delete_gesture(self, name):
        """Delete a person's gesture"""
        if name in self.registered_gestures:
            del self.registered_gestures[name]
            self._save_gestures()
            print(f"✅ Deleted gesture for '{name}'")
            return True
        else:
            print(f"❌ No gesture found for '{name}'")
            return False

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
        print("  🔆 With Multi-Sample & Lighting Compensation 🔆")
        print("="*60)
    
    def register_person(self, num_samples=SAMPLES_PER_PERSON):
        """Register both face and gesture for a person"""
        print("\n--- REGISTRATION PROCESS ---")
        
        # Step 1: Register Face (multiple samples)
        print(f"\n[1/2] Face Registration ({num_samples} samples)")
        if not self.face_recognition.register_face(num_samples=num_samples):
            print("❌ Face registration failed")
            return False
        
        name = list(self.face_recognition.known_encodings.keys())[-1]
        
        # Step 2: Register Gesture
        print("\n[2/2] Gesture Registration")
        if not self.gesture_recognition.register_gesture(name):
            print("❌ Gesture registration failed")
            choice = input("Keep face registration without gesture? (y/n): ").strip().lower()
            if choice != 'y':
                self.face_recognition.delete_person(name)
            return False
        
        print(f"\n✅ {name} registered successfully with {len(self.face_recognition.known_encodings[name])} face samples and gesture!")
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
            return True
        else:
            print("\n" + "="*60)
            print(f"  ❌ AUTHENTICATION FAILED")
            print(f"  Face verified but gesture did not match")
            print(f"  🚪 ACCESS DENIED")
            print("="*60)
            return False
    
    def list_registered_users(self):
        """List all registered users with sample counts"""
        print("\n📋 Registered Users:")
        if not self.face_recognition.known_encodings:
            print("  No users registered")
            return
        
        for name, encodings in self.face_recognition.known_encodings.items():
            has_gesture = "✅" if name in self.gesture_recognition.registered_gestures else "❌"
            sample_count = len(encodings)
            print(f"  • {name} - Face Samples: {sample_count} | Gesture: {has_gesture}")
    
    def delete_user(self):
        """Delete a registered user"""
        self.list_registered_users()
        name = input("\nEnter name to delete: ").strip()
        
        if not name:
            print("❌ Name cannot be empty")
            return
        
        confirm = input(f"⚠️  Delete ALL data for '{name}'? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cancelled")
            return
        
        face_deleted = self.face_recognition.delete_person(name)
        gesture_deleted = self.gesture_recognition.delete_gesture(name)
        
        if face_deleted or gesture_deleted:
            print(f"✅ User '{name}' deleted successfully")
        else:
            print(f"❌ User '{name}' not found")

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
        print(f"1. Register New Person ({SAMPLES_PER_PERSON} face samples + gesture)")
        print("2. Authenticate Person (Face + Gesture)")
        print("3. List Registered Users")
        print("4. Add More Face Samples to Existing User")
        print("5. Test Face Recognition Only")
        print("6. Test Gesture Recognition Only")
        print("7. Delete User")
        print("8. Exit")
        print("-"*60)
        
        choice = input("Choose option (1-8): ").strip()
        
        if choice == "1":
            system.register_person()
        
        elif choice == "2":
            system.authenticate_person()
        
        elif choice == "3":
            system.list_registered_users()
        
        elif choice == "4":
            system.list_registered_users()
            name = input("\nEnter person name: ").strip()
            if name in system.face_recognition.known_encodings:
                num = input(f"How many additional samples? (default {SAMPLES_PER_PERSON}): ").strip()
                num = int(num) if num.isdigit() else SAMPLES_PER_PERSON
                system.face_recognition.register_face(name=name, num_samples=num)
            else:
                print(f"❌ Person '{name}' not found")
        
        elif choice == "5":
            name, conf = system.face_recognition.verify_face_continuous()
            if name:
                print(f"✅ Recognized: {name} ({conf:.0%})")
            else:
                print("❌ Face not recognized")
        
        elif choice == "6":
            system.list_registered_users()
            person_name = input("\nEnter person name: ").strip()
            system.gesture_recognition.verify_gesture(person_name)
        
        elif choice == "7":
            system.delete_user()
        
        elif choice == "8":
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    main()