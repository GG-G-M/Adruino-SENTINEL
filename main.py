import cv2
import os
import numpy as np
import pickle
import time
import mediapipe as mp
import serial
import serial.tools.list_ports
import face_recognition

# ============================================
# CONFIGURATION
# ============================================
REGISTERED_FOLDER = "authorized_faces"
ENCODINGS_FILE = "face_encodings.pkl"
GESTURES_FILE = "hand_gestures.pkl"

# Security thresholds
FACE_CONFIDENCE_THRESHOLD = 0.75
MIN_CONSISTENCY = 0.7

# Timing settings
FACE_VERIFICATION_TIME = 5.0
GESTURE_TIMEOUT = 6.9

# Multi-sample settings
SAMPLES_PER_PERSON = 3

# Visual feedback colors
COLORS = {
    'face_detected': (0, 255, 0),      # Green
    'face_recognized': (0, 200, 255),  # Orange
    'face_unknown': (0, 0, 255),       # Red
    'hand_detected': (255, 255, 0),    # Cyan
    'hand_matched': (255, 0, 255),     # Magenta
    'text': (255, 255, 255),           # White
    'timer': (255, 200, 0)             # Yellow
}

# ============================================
# ARDUINO COMMUNICATION CLASS
# ============================================
class ArduinoController:
    def __init__(self):
        self.arduino = None
        self.connected = self._connect_arduino()
    
    def _connect_arduino(self):
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            if any(id in p.description.upper() for id in ['CH340', 'ARDUINO', 'USB SERIAL']):
                try:
                    self.arduino = serial.Serial(p.device, 9600, timeout=1)
                    time.sleep(2)
                    print(f"✅ Arduino connected: {p.device}")
                    return True
                except:
                    continue
        print("⚠️  Arduino not found - Simulation mode")
        return False
    
    def send_command(self, command):
        if self.connected and self.arduino:
            try:
                self.arduino.write(f"{command}\n".encode())
                return True
            except:
                self.connected = False
        return False
    
    def access_granted(self): self.send_command("GRANTED")
    def access_denied(self): self.send_command("DENIED")
    def system_ready(self): self.send_command("READY")
    def face_verified(self): self.send_command("FACE_VERIFIED")
    def gesture_required(self): self.send_command("GESTURE_REQUIRED")
    
    def close(self):
        if self.connected and self.arduino:
            self.arduino.close()

# ============================================
# FACE RECOGNITION CLASS WITH VISUAL FEEDBACK
# ============================================
class FaceRecognition:
    def __init__(self, arduino_controller):
        self.arduino = arduino_controller
        self.known_encodings = {}
        self._load_encodings()
        self.current_frame = None  # Store current frame for feedback
    
    def _load_encodings(self):
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, 'rb') as f:
                    self.known_encodings = pickle.load(f)
                print(f"✅ Loaded {len(self.known_encodings)} users")
            except Exception as e:
                print(f"❌ Error loading encodings: {e}")
    
    def _save_encodings(self):
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(self.known_encodings, f)
    
    def draw_face_feedback(self, frame, face_locations, face_names, confidences):
        """Draw visual feedback for face detection/recognition"""
        # Draw face rectangles and labels
        for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, confidences):
            # Draw face rectangle
            if name == "Unknown":
                color = COLORS['face_unknown']
                thickness = 2
            else:
                color = COLORS['face_recognized']
                thickness = 3
            
            # Draw main rectangle
            cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
            
            # Draw smaller inner rectangle for depth effect
            cv2.rectangle(frame, 
                         (left + 5, top + 5), 
                         (right - 5, bottom - 5), 
                         color, 1)
            
            # Draw label background
            label = f"{name}: {confidence:.1%}" if name != "Unknown" else "Unknown"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_left = left
            label_top = top - label_size[1] - 10
            
            if label_top < 0:
                label_top = bottom + label_size[1] + 10
            
            # Label background
            cv2.rectangle(frame,
                         (label_left, label_top - label_size[1] - 5),
                         (label_left + label_size[0] + 10, label_top + 5),
                         color, -1)
            
            # Label text
            cv2.putText(frame, label,
                       (label_left + 5, label_top),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 2)
            
            # Draw facial feature points (simplified)
            face_center_x = (left + right) // 2
            face_center_y = (top + bottom) // 2
            
            # Eyes
            cv2.circle(frame, (face_center_x - 20, face_center_y - 10), 3, color, -1)
            cv2.circle(frame, (face_center_x + 20, face_center_y - 10), 3, color, -1)
            
            # Mouth
            cv2.circle(frame, (face_center_x, face_center_y + 15), 3, color, -1)
    
    def register_face(self, name=None, num_samples=SAMPLES_PER_PERSON):
        if name is None:
            name = input("Enter name: ").strip()
        
        if not name:
            print("❌ Name cannot be empty")
            return False
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return False
        
        print(f"\n📸 Registering {num_samples} samples for '{name}'")
        print("Press SPACE to capture, ESC to cancel")
        
        samples = []
        while len(samples) < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Store current frame for feedback
            self.current_frame = frame.copy()
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if face_locations:
                # Get face encodings
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                if face_encodings:
                    # Draw feedback
                    top, right, bottom, left = face_locations[0]
                    
                    # Green pulsing effect for detected face
                    pulse = int((np.sin(time.time() * 5) + 1) * 50)  # Pulsing animation
                    color = (0, 255 - pulse, pulse)
                    
                    # Draw face rectangle with pulsing effect
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
                    
                    # Draw instruction text
                    cv2.putText(frame, f"FACE DETECTED - Sample {len(samples)+1}/{num_samples}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw crosshair on face
                    center_x = (left + right) // 2
                    center_y = (top + bottom) // 2
                    
                    # Crosshair lines
                    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), 
                            (0, 255, 0), 2)
                    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), 
                            (0, 255, 0), 2)
                    
                    # Corner indicators
                    corner_size = 15
                    cv2.line(frame, (left, top), (left + corner_size, top), (0, 255, 0), 3)
                    cv2.line(frame, (left, top), (left, top + corner_size), (0, 255, 0), 3)
                    cv2.line(frame, (right, top), (right - corner_size, top), (0, 255, 0), 3)
                    cv2.line(frame, (right, top), (right, top + corner_size), (0, 255, 0), 3)
                    cv2.line(frame, (left, bottom), (left + corner_size, bottom), (0, 255, 0), 3)
                    cv2.line(frame, (left, bottom), (left, bottom - corner_size), (0, 255, 0), 3)
                    cv2.line(frame, (right, bottom), (right - corner_size, bottom), (0, 255, 0), 3)
                    cv2.line(frame, (right, bottom), (right, bottom - corner_size), (0, 255, 0), 3)
            
            else:
                # No face detected feedback
                h, w = frame.shape[:2]
                cv2.putText(frame, "NO FACE DETECTED - Look at camera", 
                           (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw scanning animation
                scan_y = int((np.sin(time.time() * 3) + 1) / 2 * h)
                cv2.line(frame, (0, scan_y), (w, scan_y), (0, 255, 255), 2)
            
            cv2.imshow("Register Face", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32:  # SPACE key to capture
                if face_encodings:
                    samples.append(face_encodings[0])
                    print(f"  ✅ Captured sample {len(samples)}/{num_samples}")
                    
                    # Visual confirmation flash
                    for _ in range(3):
                        flash_frame = frame.copy()
                        cv2.rectangle(flash_frame, (0, 0), (w, h), (0, 255, 0), -1)
                        cv2.addWeighted(flash_frame, 0.3, frame, 0.7, 0, frame)
                        cv2.imshow("Register Face", frame)
                        cv2.waitKey(50)
                    
                    time.sleep(0.3)
            
            elif key == 27:  # ESC to cancel
                print("\n❌ Registration cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return False
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save encodings
        if name not in self.known_encodings:
            self.known_encodings[name] = []
        self.known_encodings[name].extend(samples)
        self._save_encodings()
        
        print(f"\n✅ Registered {name} with {len(samples)} samples")
        return True
    
    def verify_face_continuous(self, duration=FACE_VERIFICATION_TIME):
        if not self.known_encodings:
            print("❌ No registered faces")
            return None, None
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return None, None
        
        print(f"🔍 Verifying face for {duration} seconds...")
        
        start_time = time.time()
        name_counts = {}
        confidence_scores = {}
        verification_frames = []
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Store for feedback
            self.current_frame = frame.copy()
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_frame)
            face_names = []
            face_confidences = []
            
            if face_locations:
                # Get encodings for detected faces
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for face_encoding in face_encodings:
                    # Compare with all known faces
                    best_match = None
                    best_distance = 1.0
                    
                    for name, known_encs in self.known_encodings.items():
                        for known_enc in known_encs:
                            distance = face_recognition.face_distance([known_enc], face_encoding)[0]
                            if distance < best_distance:
                                best_distance = distance
                                best_match = name
                    
                    confidence = 1 - best_distance
                    if best_match and confidence >= FACE_CONFIDENCE_THRESHOLD:
                        name_counts[best_match] = name_counts.get(best_match, 0) + 1
                        if best_match not in confidence_scores:
                            confidence_scores[best_match] = []
                        confidence_scores[best_match].append(confidence)
                        face_names.append(best_match)
                        face_confidences.append(confidence)
                    else:
                        face_names.append("Unknown")
                        face_confidences.append(confidence)
                
                # Draw face feedback
                self.draw_face_feedback(frame, face_locations, face_names, face_confidences)
            
            # Draw timer and status
            elapsed = time.time() - start_time
            remaining = max(0, duration - elapsed)
            
            # Timer bar
            bar_width = 400
            bar_height = 20
            bar_x = (frame.shape[1] - bar_width) // 2
            bar_y = 50
            
            # Progress
            progress = elapsed / duration
            progress_width = int(bar_width * progress)
            
            # Draw progress bar background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            
            # Draw progress
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                         COLORS['timer'], -1)
            
            # Draw timer text
            timer_text = f"Verifying: {remaining:.1f}s"
            cv2.putText(frame, timer_text, (bar_x, bar_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['timer'], 2)
            
            # Draw status text
            if face_locations:
                status = "FACE DETECTED - Processing..."
                color = (0, 255, 0)
            else:
                status = "NO FACE DETECTED - Look at camera"
                color = (0, 0, 255)
            
            cv2.putText(frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw scan lines animation
            scan_y = int((np.sin(time.time() * 4) + 1) / 2 * frame.shape[0])
            cv2.line(frame, (0, scan_y), (frame.shape[1], scan_y), 
                    (255, 255, 0, 128), 1)
            
            cv2.imshow("Face Verification", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Determine result
        if name_counts:
            best_name = max(name_counts, key=name_counts.get)
            consistency = name_counts[best_name] / sum(name_counts.values())
            avg_confidence = np.mean(confidence_scores[best_name])
            
            if consistency >= MIN_CONSISTENCY and avg_confidence >= FACE_CONFIDENCE_THRESHOLD:
                print(f"✅ Face verified: {best_name} ({avg_confidence:.1%})")
                self.arduino.face_verified()
                return best_name, avg_confidence
        
        print("❌ Face not recognized")
        return None, None
    
    def delete_person(self, name):
        if name in self.known_encodings:
            for file in os.listdir(REGISTERED_FOLDER):
                if file.startswith(name):
                    os.remove(os.path.join(REGISTERED_FOLDER, file))
            
            del self.known_encodings[name]
            self._save_encodings()
            print(f"✅ Deleted {name}")
            return True
        
        print(f"❌ {name} not found")
        return False

# ============================================
# HAND GESTURE RECOGNITION WITH VISUAL FEEDBACK
# ============================================
class HandGestureRecognition:
    def __init__(self, arduino_controller):
        self.arduino = arduino_controller
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.registered_gestures = {}
        self._load_gestures()
    
    def _load_gestures(self):
        if os.path.exists(GESTURES_FILE):
            try:
                with open(GESTURES_FILE, 'rb') as f:
                    self.registered_gestures = pickle.load(f)
                print(f"✅ Loaded {len(self.registered_gestures)} gestures")
            except:
                pass
    
    def _save_gestures(self):
        with open(GESTURES_FILE, 'wb') as f:
            pickle.dump(self.registered_gestures, f)
    
    def draw_hand_feedback(self, frame, hand_landmarks, matched=False, similarity=0):
        """Draw enhanced visual feedback for hand gestures"""
        if hand_landmarks:
            # Draw hand landmarks and connections
            self.mp_drawing.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=COLORS['hand_detected'], thickness=2, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=COLORS['hand_detected'], thickness=2)
            )
            
            # Get hand bounding box
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            
            h, w, _ = frame.shape
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)
            
            # Draw bounding box
            if matched:
                color = COLORS['hand_matched']
                thickness = 3
                
                # Draw pulsing effect for matched gesture
                pulse = int((np.sin(time.time() * 6) + 1) * 30)
                color_with_pulse = (color[0], min(255, color[1] + pulse), color[2])
                
                cv2.rectangle(frame, (x_min - 10, y_min - 10), 
                            (x_max + 10, y_max + 10), color_with_pulse, thickness)
                
                # Draw checkmark for match
                check_x, check_y = x_max + 30, y_min
                cv2.line(frame, (check_x, check_y), (check_x + 10, check_y + 20), 
                        color_with_pulse, 3)
                cv2.line(frame, (check_x + 10, check_y + 20), (check_x + 30, check_y - 10), 
                        color_with_pulse, 3)
                
                # Draw similarity percentage
                sim_text = f"Match: {similarity:.1%}"
                cv2.putText(frame, sim_text, (x_min, y_min - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_with_pulse, 2)
            else:
                color = COLORS['hand_detected']
                thickness = 2
                cv2.rectangle(frame, (x_min - 10, y_min - 10), 
                            (x_max + 10, y_max + 10), color, thickness)
                
                # Draw scanning dots on corners
                dot_radius = 4
                for i in range(4):
                    angle = time.time() * 3 + i * np.pi / 2
                    dot_x = int(x_min + (np.sin(angle) + 1) / 2 * (x_max - x_min))
                    dot_y = int(y_min + (np.cos(angle) + 1) / 2 * (y_max - y_min))
                    cv2.circle(frame, (dot_x, dot_y), dot_radius, color, -1)
            
            # Draw palm center
            palm_x = int(np.mean(x_coords[:5]) * w)
            palm_y = int(np.mean(y_coords[:5]) * h)
            cv2.circle(frame, (palm_x, palm_y), 8, (255, 255, 255), -1)
            cv2.circle(frame, (palm_x, palm_y), 8, color, 2)
    
    def register_gesture(self, person_name):
        print(f"\n✋ Register gesture for {person_name}")
        print("Show gesture and press SPACE to save, ESC to cancel")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Process with MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            if results.multi_hand_landmarks:
                # Draw hand feedback
                self.draw_hand_feedback(frame, results.multi_hand_landmarks[0])
                
                # Extract landmarks
                landmarks = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # Status text
                cv2.putText(frame, "HAND DETECTED - Press SPACE to save", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Registering for: {person_name}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                # No hand detected
                cv2.putText(frame, "SHOW YOUR HAND GESTURE", 
                           (frame.shape[1]//4, frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw hand outline as hint
                h, w = frame.shape[:2]
                center_x, center_y = w//2, h//2
                
                # Draw hand silhouette
                cv2.circle(frame, (center_x, center_y - 50), 30, (100, 100, 255), 2)
                for i in range(5):
                    angle = -np.pi/2 + i * np.pi/6
                    finger_x = int(center_x + np.cos(angle) * 80)
                    finger_y = int(center_y - 50 + np.sin(angle) * 80)
                    cv2.line(frame, (center_x, center_y - 20), 
                            (finger_x, finger_y), (100, 100, 255), 2)
            
            cv2.imshow("Register Gesture", frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32 and results.multi_hand_landmarks:  # SPACE
                self.registered_gestures[person_name] = landmarks
                self._save_gestures()
                
                # Success animation
                for i in range(10):
                    flash_frame = frame.copy()
                    alpha = (i + 1) * 0.1
                    cv2.rectangle(flash_frame, (0, 0), (w, h), (0, 255, 0), -1)
                    cv2.addWeighted(flash_frame, alpha, frame, 1 - alpha, 0, frame)
                    cv2.putText(frame, "GESTURE SAVED!", 
                               (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
                    cv2.imshow("Register Gesture", frame)
                    cv2.waitKey(30)
                
                print(f"✅ Gesture saved for {person_name}")
                cap.release()
                cv2.destroyAllWindows()
                return True
            
            elif key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    def verify_gesture(self, person_name, timeout=GESTURE_TIMEOUT):
        if person_name not in self.registered_gestures:
            print(f"❌ No gesture for {person_name}")
            return False
        
        print(f"\n✋ Show gesture within {timeout}s...")
        self.arduino.gesture_required()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return False
        
        start_time = time.time()
        registered = self.registered_gestures[person_name]
        last_similarity = 0
        
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
            
            remaining = timeout - (time.time() - start_time)
            matched = False
            
            if results.multi_hand_landmarks:
                # Extract current landmarks
                current = []
                for lm in results.multi_hand_landmarks[0].landmark:
                    current.extend([lm.x, lm.y, lm.z])
                
                # Calculate similarity
                distance = np.linalg.norm(np.array(current) - np.array(registered))
                similarity = 1 / (1 + distance)
                last_similarity = similarity
                
                if similarity > 0.7:  # Match threshold
                    matched = True
                
                # Draw hand feedback
                self.draw_hand_feedback(frame, results.multi_hand_landmarks[0], 
                                       matched, similarity)
            
            # Draw timer and progress
            h, w = frame.shape[:2]
            
            # Timer bar
            bar_width = 400
            bar_height = 15
            bar_x = (w - bar_width) // 2
            bar_y = h - 50
            
            progress = (timeout - remaining) / timeout
            progress_width = int(bar_width * progress)
            
            # Progress bar
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                         COLORS['hand_matched'], -1)
            
            # Timer text
            timer_text = f"Time: {remaining:.1f}s"
            cv2.putText(frame, timer_text, (bar_x, bar_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['timer'], 2)
            
            # Status text
            if results.multi_hand_landmarks:
                if matched:
                    status = "GESTURE MATCHED! ✓"
                    color = (0, 255, 0)
                else:
                    status = f"HAND DETECTED - Similarity: {last_similarity:.1%}"
                    color = (255, 200, 0)
            else:
                status = "SHOW YOUR HAND GESTURE"
                color = (0, 0, 255)
            
            cv2.putText(frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Verifying: {person_name}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw target circle (where hand should be)
            if not results.multi_hand_landmarks:
                center_x, center_y = w//2, h//2
                radius = 100
                
                # Pulsing target circle
                pulse_radius = int(radius * (0.8 + 0.2 * np.sin(time.time() * 3)))
                cv2.circle(frame, (center_x, center_y), pulse_radius, 
                          (0, 255, 255, 100), 2)
                
                # Crosshair
                cv2.line(frame, (center_x - 30, center_y), 
                        (center_x + 30, center_y), (0, 255, 255), 2)
                cv2.line(frame, (center_x, center_y - 30), 
                        (center_x, center_y + 30), (0, 255, 255), 2)
            
            cv2.imshow("Gesture Verification", frame)
            
            # Check for match
            if matched:
                time.sleep(0.5)  # Brief success display
                print(f"✅ Gesture verified ({last_similarity:.1%})")
                cap.release()
                cv2.destroyAllWindows()
                return True
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("❌ Gesture not verified")
        return False

# ============================================
# MAIN SECURITY SYSTEM
# ============================================
class SecuritySystem:
    def __init__(self):
        print("\n" + "="*60)
        print("SECURITY SYSTEM INITIALIZING")
        print("="*60)
        
        self.arduino = ArduinoController()
        self.face_recognition = FaceRecognition(self.arduino)
        self.gesture_recognition = HandGestureRecognition(self.arduino)
        
        # Create folders
        os.makedirs(REGISTERED_FOLDER, exist_ok=True)
        
        self.arduino.system_ready()
        print("✅ System Ready")
    
    def register_person(self):
        print("\n--- REGISTRATION ---")
        
        name = input("Enter name: ").strip()
        if not name:
            print("❌ Name required")
            return False
        
        print("\n[1/2] Face Registration")
        if not self.face_recognition.register_face(name=name):
            print("❌ Face registration failed")
            return False
        
        print("\n[2/2] Gesture Registration")
        if not self.gesture_recognition.register_gesture(name):
            print("⚠️  Gesture registration skipped")
        
        print(f"\n✅ {name} registered successfully!")
        return True
    
    def authenticate_person(self):
        print("\n" + "="*60)
        print("AUTHENTICATION STARTED")
        print("="*60)
        
        print("\n[1/2] Face Verification")
        person_name, confidence = self.face_recognition.verify_face_continuous()
        
        if not person_name:
            print("❌ Face verification failed")
            self.arduino.access_denied()
            return False
        
        print(f"✅ Face verified: {person_name}")
        
        print("\n[2/2] Gesture Verification")
        if person_name in self.gesture_recognition.registered_gestures:
            if not self.gesture_recognition.verify_gesture(person_name):
                print("❌ Gesture verification failed")
                self.arduino.access_denied()
                return False
        else:
            print("⚠️  No gesture registered, skipping")
        
        print("\n" + "="*60)
        print(f"✅ ACCESS GRANTED - Welcome {person_name}!")
        print("="*60)
        self.arduino.access_granted()
        return True
    
    def list_users(self):
        print("\n📋 Registered Users:")
        if not self.face_recognition.known_encodings:
            print("  No users registered")
            return
        
        for name, encodings in self.face_recognition.known_encodings.items():
            has_gesture = "✅" if name in self.gesture_recognition.registered_gestures else "❌"
            print(f"  • {name} - Samples: {len(encodings)} | Gesture: {has_gesture}")
    
    def delete_user(self):
        self.list_users()
        name = input("\nEnter name to delete: ").strip()
        
        if not name:
            return
        
        if input(f"Delete '{name}'? (y/n): ").lower() == 'y':
            face_deleted = self.face_recognition.delete_person(name)
            if name in self.gesture_recognition.registered_gestures:
                del self.gesture_recognition.registered_gestures[name]
                self.gesture_recognition._save_gestures()
            
            if face_deleted:
                print(f"✅ {name} deleted")
            else:
                print(f"❌ {name} not found")
    
    def close(self):
        self.arduino.close()

# ============================================
# MAIN PROGRAM
# ============================================
def main():
    system = SecuritySystem()
    
    try:
        while True:
            print("\n" + "-"*60)
            print("MAIN MENU")
            print("-"*60)
            print("1. Register New Person")
            print("2. Authenticate Person")
            print("3. List Users")
            print("4. Delete User")
            print("5. Test Arduino")
            print("6. Exit")
            print("-"*60)
            
            choice = input("Choice (1-6): ").strip()
            
            if choice == "1":
                system.register_person()
            elif choice == "2":
                system.authenticate_person()
            elif choice == "3":
                system.list_users()
            elif choice == "4":
                system.delete_user()
            elif choice == "5":
                if system.arduino.connected:
                    system.arduino.send_command("TEST")
                    print("✅ Test command sent")
                else:
                    print("❌ Arduino not connected")
            elif choice == "6":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Program interrupted")
    finally:
        system.close()

if __name__ == "__main__":
    main()