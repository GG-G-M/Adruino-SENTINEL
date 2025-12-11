import cv2
import os
import numpy as np
from PIL import Image
import pickle
import time
import mediapipe as mp
import serial
import serial.tools.list_ports

# Face recognition library
try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False
    print("⚠️ face_recognition not installed.")

# ============================================
# CONFIGURATION
# ============================================

REGISTERED_FOLDER = "authorized_faces"
ENCODINGS_FILE = "face_encodings.pkl"
GESTURES_FILE = "hand_gestures.pkl"

# Registration Settings
SAMPLES_PER_PERSON = 10
AUTO_CAPTURE_INTERVAL = 1.5
MIN_FACE_CONFIDENCE = 0.80

# Verification Settings
FACE_DISTANCE_THRESHOLD = 0.45  # Lower = stricter matching
MIN_CONSISTENCY = 0.60  # Frame match rate required (How much face is within that frame)
MIN_MATCH_FRAMES = 5  # Minimum frames needed for early exit
MIN_CONFIDENCE_SCORE = 0.10  # Minimum face confidence (IMPORTANT MODIFY BASED ON THE AVERAGE CONSISTENCY) (How much Face is RECOGNIZE within that frame)
FACE_VERIFICATION_TIME = 60  # Max authentication time (seconds)

# Gesture Settings
GESTURE_MATCH_THRESHOLD = 0.69
GESTURE_TIMEOUT = 6.9
GESTURE_TEST_TIMEOUT = 10.0
GESTURE_DETECTION_CONFIDENCE = 0.5
GESTURE_TRACKING_CONFIDENCE = 0.3

# Arduino Settings
DOOR_UNLOCK_DURATION = 2.0

# DOOR CONFIGURATION
# These settings should match your Arduino configuration
DOOR_OPENS_CLOCKWISE = True        # Must match Arduino's DOOR_OPENS_CLOCKWISE setting
DOOR_OPEN_SPEED = 60              # Opening speed percentage (0-100)
DOOR_CLOSE_SPEED = 60             # Closing speed percentage (0-100)
DOOR_OPEN_DURATION = 5.0          # How long door stays open (seconds)
DOOR_CLOSE_DELAY = 2.0            # Delay before closing starts (seconds)
DOOR_ROTATION_TIME = 3.0          # Time for complete door rotation (seconds)

# Calculate total door sequence time for proper waiting
DOOR_TOTAL_SEQUENCE_TIME = DOOR_ROTATION_TIME + DOOR_OPEN_DURATION + DOOR_CLOSE_DELAY + DOOR_ROTATION_TIME

# ============================================
# ARDUINO CONTROLLER
# ============================================
class ArduinoController:
    def __init__(self):
        self.arduino = None
        self.connected = False
        self._connect_arduino()
    
    def _connect_arduino(self):
        print("\n🔌 Scanning for Arduino...")
        ports = list(serial.tools.list_ports.comports())
        
        for p in ports:
            print(f"  Found: {p.device} - {p.description}")
            
            if any(identifier in p.description.upper() for identifier in 
                ['CH340', 'ARDUINO', 'USB SERIAL', 'USB-SERIAL']):
                try:
                    print(f"🔌 Attempting connection to {p.device}...")
                    self.arduino = serial.Serial(p.device, 9600, timeout=1)
                    time.sleep(2)
                    self.connected = True
                    print("✅ Arduino connected successfully!")
                    return
                except Exception as e:
                    print(f"❌ Failed to connect to {p.device}: {e}")
                    continue
        
        print("⚠️ Arduino not found. Running in simulation mode.")
        self.connected = False
    
    def send_command(self, command):
        if self.connected and self.arduino:
            try:
                self.arduino.write(f"{command}\n".encode())
                print(f"📤 Sent to Arduino: {command}")
                return True
            except Exception as e:
                print(f"❌ Failed to send command: {e}")
                self.connected = False
                return False
        else:
            print(f"🔌 [SIMULATION] Command: {command}")
            return True
    
    def access_granted(self):
        print("ACCESS GRANTED - Activating devices...")
        success = self.send_command("granted")
        if success:
            # Wait for door to complete its sequence
            print("🔓 Door opening and closing automatically...")
            # Wait for Arduino to complete its sequence
            time.sleep(DOOR_TOTAL_SEQUENCE_TIME + 2)  # Use configured timing + buffer
            print("✅ Door sequence completed")
        return success
    
    def access_denied(self):
        print("🚫 ACCESS DENIED - Activating alert...")
        self.send_command("denied")
    
    def system_ready(self):
        self.send_command("ready")
    
    def face_verified(self):
        self.send_command("face_verified")
    
    def gesture_required(self):
        self.send_command("gesture_required")
    
    def test_sonar(self):
        """Test sonar sensor for 5 seconds"""
        print("\n📡 Testing Sonar Sensor...")
        print("   Duration: 5 seconds")
        print("   Reading distance every 0.5s")
        print("=" * 60)
        
        if not self.connected:
            print("⚠️ Arduino not connected - simulation mode")
            print("🔌 [SIMULATION] Would send TEST_SONAR command")
            return
        
        try:
            # Clear any pending serial data
            self.arduino.reset_input_buffer()
            
            # Send test command
            self.send_command("test_sonar")
            
            # Read distance measurements for ~5 seconds
            start_time = time.time()
            measurement_count = 0
            
            while time.time() - start_time < 6:  # 6 seconds to ensure we get all data
                if self.arduino.in_waiting > 0:
                    try:
                        line = self.arduino.readline().decode('utf-8').strip()
                        
                        if line.startswith("DISTANCE:"):
                            measurement_count += 1
                            distance_str = line.replace("DISTANCE:", "")
                            
                            if distance_str == "ERROR":
                                print(f"📏 Reading {measurement_count}: No echo detected (sensor error)")
                            else:
                                print(f"📏 Reading {measurement_count}: {distance_str}")
                        
                    except Exception as e:
                        print(f"⚠️ Error reading: {e}")
                        
                time.sleep(0.1)  # Small delay to prevent CPU spinning
            
            print("=" * 60)
            print(f"✅ Sonar test complete - {measurement_count} measurements received")
            
        except Exception as e:
            print(f"❌ Sonar test failed: {e}")
    
    def test_servo(self):
        """Test continuous servo functionality"""
        print("\n⚙️ Testing Continuous Servo...")
        print("   Testing right, left, and stop movements")
        print("=" * 60)
        
        if not self.connected:
            print("⚠️ Arduino not connected - simulation mode")
            print("🔌 [SIMULATION] Would send servo test commands")
            return
        
        try:
            # Test right rotation
            print("Testing RIGHT rotation...")
            self.send_command("right")
            time.sleep(2)
            
            print("Stopping...")
            self.send_command("stop")
            time.sleep(1)
            
            # Test left rotation
            print("Testing LEFT rotation...")
            self.send_command("left")
            time.sleep(2)
            
            print("Stopping...")
            self.send_command("stop")
            time.sleep(1)
            
            # Test speed change
            print("Testing speed change...")
            self.send_command("speed50")
            time.sleep(1)
            self.send_command("right")
            time.sleep(2)
            self.send_command("stop")
            
            print("=" * 60)
            print("✅ Servo test complete")
            
        except Exception as e:
            print(f"❌ Servo test failed: {e}")
    
    def servo_right(self):
        """Rotate servo right (clockwise)"""
        print("🔄 Rotating servo RIGHT...")
        self.send_command("right")
        
    def servo_left(self):
        """Rotate servo left (counter-clockwise)"""
        print("🔄 Rotating servo LEFT...")
        self.send_command("left")
        
    def servo_stop(self):
        """Stop servo rotation"""
        print("🛑 Stopping servo...")
        self.send_command("stop")
        
    def servo_set_speed(self, speed):
        """Set servo speed (0-100)"""
        print(f"⚡ Setting servo speed to {speed}%...")
        self.send_command(f"speed{speed}")
    
    def close(self):
        if self.connected and self.arduino:
            self.arduino.close()
            print("🔌 Arduino connection closed")

# ============================================
# FACE RECOGNITION WITH AUTO-SAMPLING
# ============================================
class FaceRecognition:
    def __init__(self, arduino_controller):
        self.arduino = arduino_controller
        self.known_encodings = {}
        
        if not FACE_REC_AVAILABLE:
            print("❌ face_recognition library not available")
            return
        
        self._load_encodings()
        print("✅ Face Recognition initialized")
        print(f"   Library: face_recognition (dlib)")
        print(f"   Samples per person: {SAMPLES_PER_PERSON}")
        print(f"   Distance threshold: {FACE_DISTANCE_THRESHOLD} (strict)")
    
    def _load_encodings(self):
        if not os.path.exists(ENCODINGS_FILE):
            print("📁 No face database found. Register new faces.")
            return
        
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                self.known_encodings = pickle.load(f)
            
            total_samples = sum(len(encs) for encs in self.known_encodings.values())
            print(f"✅ Loaded {len(self.known_encodings)} people with {total_samples} total samples")
        except Exception as e:
            print(f"❌ Error loading encodings: {e}")
            self.known_encodings = {}
    
    def _save_encodings(self):
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(self.known_encodings, f)
    
    def register_face(self, name=None, num_samples=SAMPLES_PER_PERSON):
        """Auto-capture face samples with quality checks"""
        if not FACE_REC_AVAILABLE:
            print("❌ face_recognition not available")
            return False
        
        if name is None:
            name = input("Enter name: ").strip()
        
        if not name:
            print("❌ Name cannot be empty")
            return False
        
        if name in self.known_encodings:
            print(f"⚠️ '{name}' already registered with {len(self.known_encodings[name])} samples")
            choice = input("Add more samples? (y/n): ").strip().lower()
            if choice != 'y':
                return False
        else:
            self.known_encodings[name] = []
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return False
        
        # Warm up camera
        print("\n📸 Warming up camera...")
        for _ in range(10):
            cap.read()
        
        captured_count = 0
        current_samples = len(self.known_encodings[name])
        last_capture_time = 0
        
        print(f"\n🤖 AUTO-CAPTURE MODE")
        print(f"   Capturing {num_samples} high-quality samples for '{name}'")
        print(f"   Interval: {AUTO_CAPTURE_INTERVAL}s between captures")
        print("="*60)
        print("💡 INSTRUCTIONS:")
        print("   1. Look directly at the camera")
        print("   2. Slowly turn your head left and right")
        print("   3. Move slightly closer/farther")
        print("   4. Change expressions slightly")
        print("   5. System will auto-capture diverse samples")
        print("="*60)
        print("Press 'q' to cancel\n")
        
        time.sleep(2)  # Give user time to read
        
        while captured_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                continue
            
            display_frame = frame.copy()
            current_time = time.time()
            
            # Detect faces with high confidence
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            # Draw rectangles and status
            if face_locations:
                for (top, right, bottom, left) in face_locations:
                    # Calculate face size (quality check)
                    face_width = right - left
                    face_height = bottom - top
                    face_area = face_width * face_height
                    frame_area = frame.shape[0] * frame.shape[1]
                    face_ratio = face_area / frame_area
                    
                    # Get face encodings to check quality
                    face_encodings = face_recognition.face_encodings(rgb_frame, [(top, right, bottom, left)])
                    
                    # Calculate confidence based on encoding quality
                    face_confidence = 1.0  # Default high confidence
                    if face_encodings:
                        # Check encoding strength (non-zero values indicate good quality)
                        encoding_strength = np.mean(np.abs(face_encodings[0]))
                        face_confidence = min(1.0, encoding_strength * 2)  # Normalize
                    
                    # Quality indicators
                    is_good_size = 0.05 < face_ratio < 0.4  # Face is 5-40% of frame
                    is_good_confidence = face_confidence >= MIN_FACE_CONFIDENCE
                    time_ready = (current_time - last_capture_time) >= AUTO_CAPTURE_INTERVAL
                    
                    # Color based on quality
                    if is_good_size and is_good_confidence and time_ready:
                        color = (0, 255, 0)  # Green - ready to capture
                        status = "READY"
                    elif is_good_size and is_good_confidence:
                        color = (0, 255, 255)  # Yellow - waiting
                        status = "WAIT"
                    elif not is_good_confidence:
                        color = (255, 0, 0)  # Red - low confidence
                        status = f"LOW CONF ({face_confidence:.0%})"
                    else:
                        color = (0, 165, 255)  # Orange - adjust distance
                        if face_ratio < 0.05:
                            status = "TOO FAR"
                        else:
                            status = "TOO CLOSE"
                    
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(display_frame, status, (left, top - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Show confidence score
                    cv2.putText(display_frame, f"Conf: {face_confidence:.0%}", (left, bottom + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    
                    # Auto-capture when conditions are met
                    if is_good_size and is_good_confidence and time_ready and captured_count < num_samples:
                        try:
                            # Get face encodings
                            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                            
                            if face_encodings:
                                # Quality check: ensure encoding is unique enough
                                is_unique = True
                                if self.known_encodings[name]:
                                    # Check if this sample is too similar to existing ones
                                    distances = face_recognition.face_distance(
                                        self.known_encodings[name], 
                                        face_encodings[0]
                                    )
                                    # If any distance is too small, sample is too similar
                                    if np.min(distances) < 0.15:
                                        is_unique = False
                                
                                if is_unique:
                                    # Save image
                                    os.makedirs(REGISTERED_FOLDER, exist_ok=True)
                                    img_path = os.path.join(
                                        REGISTERED_FOLDER, 
                                        f"{name}_sample{current_samples + captured_count + 1}.jpg"
                                    )
                                    cv2.imwrite(img_path, frame)
                                    
                                    # Save encoding
                                    self.known_encodings[name].append(face_encodings[0])
                                    captured_count += 1
                                    last_capture_time = current_time
                                    
                                    print(f"✅ Captured sample {captured_count}/{num_samples} - Quality: Good")
                                    
                                    # Visual feedback
                                    cv2.rectangle(display_frame, (0, 0), 
                                                (frame.shape[1], frame.shape[0]), 
                                                (0, 255, 0), 10)
                                else:
                                    print(f"⏭️  Sample too similar to existing, waiting for variation...")
                        
                        except Exception as e:
                            print(f"⚠️ Capture failed: {e}")
            else:
                cv2.putText(display_frame, "NO FACE DETECTED", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Progress bar
            progress = captured_count / num_samples
            bar_width = frame.shape[1] - 40
            bar_height = 30
            cv2.rectangle(display_frame, (20, frame.shape[0] - 60), 
                         (20 + bar_width, frame.shape[0] - 30), (50, 50, 50), -1)
            cv2.rectangle(display_frame, (20, frame.shape[0] - 60), 
                         (20 + int(bar_width * progress), frame.shape[0] - 30), 
                         (0, 255, 0), -1)
            
            # Status text
            status_text = f"Sample {captured_count}/{num_samples}"
            cv2.putText(display_frame, status_text, (30, frame.shape[0] - 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Countdown to next capture
            if face_locations and captured_count < num_samples:
                time_remaining = AUTO_CAPTURE_INTERVAL - (current_time - last_capture_time)
                if time_remaining > 0:
                    cv2.putText(display_frame, f"Next: {time_remaining:.1f}s", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Auto-Capture Registration", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n❌ Registration cancelled")
                cap.release()
                cv2.destroyAllWindows()
                return False
            
            # Check if done
            if captured_count >= num_samples:
                self._save_encodings()
                print(f"\n✅ Registration complete!")
                print(f"   Name: {name}")
                print(f"   Total samples: {len(self.known_encodings[name])}")
                print(f"   Quality: High diversity for accuracy")
                
                # Show success animation
                for _ in range(3):
                    success_frame = display_frame.copy()
                    cv2.putText(success_frame, "REGISTRATION COMPLETE!", 
                               (frame.shape[1]//2 - 200, frame.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.imshow("Auto-Capture Registration", success_frame)
                    cv2.waitKey(300)
                
                cap.release()
                cv2.destroyAllWindows()
                return True
        
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    def verify_face_continuous(self, duration=None):
        """Continuous face verification with strict matching
        
        Args:
            duration: Optional duration in seconds. If None, runs until unknown faces detected
                     or manual exit with 'q' key.
        """
        if not self.known_encodings:
            print("❌ No registered faces")
            return None, None
        
        if not FACE_REC_AVAILABLE:
            print("❌ face_recognition not available")
            return None, None
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera")
            return None, None
        
        if duration:
            print(f"👁️ Verifying face for {duration} seconds...")
        else:
            print(f"👁️ Verifying face (auto-stop on unknown)...")
        print(f"🔒 Distance threshold: {FACE_DISTANCE_THRESHOLD} (strict)")
        print(f"🔒 Required consistency: {MIN_CONSISTENCY:.0%}")
        print(f"🔒 Minimum matches: {MIN_MATCH_FRAMES}")
        print(f"🔒 Min confidence score: {MIN_CONFIDENCE_SCORE:.0%}")
        print("Press 'q' to exit manually")
        
        start_time = time.time()
        total_frames = 0
        name_counts = {}
        confidence_scores = {}
        process_every_n_frames = 2
        consecutive_unknown = 0  # Track consecutive unknown detections
        max_consecutive_unknown = 3  # Stop after 3 consecutive unknowns
        unauthorized_count = 0  # Count unauthorized/unknown faces
        max_unauthorized = 5  # Auto-deny after 5 unauthorized detections
        
        # Warm up
        for _ in range(5):
            cap.read()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            total_frames += 1
            elapsed = time.time() - start_time
            
            # Check if duration limit reached (if set)
            if duration and elapsed >= duration:
                break
            
            # Process every nth frame for speed
            if total_frames % process_every_n_frames == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame, model="hog")
                
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    if face_encodings:
                        best_name, best_distance = self._find_best_match(face_encodings[0])
                        
                        # Convert distance to confidence
                        confidence = max(0, 1 - (best_distance / FACE_DISTANCE_THRESHOLD))
                        
                        if best_name != "Unknown":
                            consecutive_unknown = 0  # Reset counter on known face
                            name_counts[best_name] = name_counts.get(best_name, 0) + 1
                            if best_name not in confidence_scores:
                                confidence_scores[best_name] = []
                            confidence_scores[best_name].append(confidence)
                            
                            # Draw on frame
                            top, right, bottom, left = face_locations[0]
                            color = (0, 255, 0)
                            label = f"{best_name}: {confidence:.0%}"
                        else:
                            consecutive_unknown += 1
                            unauthorized_count += 1
                            top, right, bottom, left = face_locations[0]
                            color = (0, 0, 255)
                            label = f"Unknown ({unauthorized_count}/{max_unauthorized})"
                            
                            # Check if we should auto-deny due to too many unauthorized
                            if duration and unauthorized_count >= max_unauthorized:
                                print(f"\n🚫 AUTO-DENIED: {max_unauthorized} unauthorized face detections")
                                cap.release()
                                cv2.destroyAllWindows()
                                return None, None
                            
                            # Check if we should stop (for no-timer mode)
                            if not duration and consecutive_unknown >= max_consecutive_unknown:
                                print(f"\n⚠️ Stopped: {max_consecutive_unknown} consecutive unknown faces detected")
                                cap.release()
                                cv2.destroyAllWindows()
                                return None, None
                        
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.putText(frame, label, (left, top - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Calculate current average face match confidence (not frame match rate!)
            current_avg_face_confidence = 0
            if name_counts:
                best_current = max(name_counts, key=name_counts.get)
                matches = name_counts[best_current]
                processed = total_frames // process_every_n_frames
                frame_match_rate = matches / processed if processed > 0 else 0
                
                # Calculate average confidence of face matches
                if best_current in confidence_scores and confidence_scores[best_current]:
                    current_avg_face_confidence = np.mean(confidence_scores[best_current])
                
                # Early exit if authorized user is verified (for authentication mode with timer)
                if duration and elapsed >= 3.0:  # Minimum 3 seconds of verification
                    # Check if all verification criteria are met
                    if (frame_match_rate >= MIN_CONSISTENCY and 
                        matches >= MIN_MATCH_FRAMES and 
                        current_avg_face_confidence >= MIN_CONFIDENCE_SCORE):
                        # User is verified! Break early
                        print(f"\n✅ User verified early at {elapsed:.1f}s - proceeding to next step")
                        break
            
            # Display info
            if duration:
                remaining = duration - elapsed
                cv2.putText(frame, f"Time: {remaining:.1f}s | Unauthorized: {unauthorized_count}/{max_unauthorized}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                cv2.putText(frame, f"Scanning... | Press 'q' to exit", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Unknown streak: {consecutive_unknown}/{max_consecutive_unknown}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            fps = total_frames / (elapsed + 0.001)
            y_offset = 90 if not duration else 60
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show average FACE MATCH confidence (not frame match rate)
            cv2.putText(frame, f"Avg Face Confidence: {current_avg_face_confidence:.0%}", (10, y_offset + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if name_counts:
                best_current = max(name_counts, key=name_counts.get)
                matches = name_counts[best_current]
                processed = total_frames // process_every_n_frames
                consistency = matches / processed if processed > 0 else 0
                
                cv2.putText(frame, f"Matches: {matches}/{processed} ({consistency:.0%})", 
                           (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Name: {best_current}", (10, y_offset + 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Face Verification", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return None, None
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate results with strict criteria
        if name_counts:
            consistent_name = max(name_counts, key=name_counts.get)
            match_count = name_counts[consistent_name]
            processed_frames = total_frames // process_every_n_frames
            consistency = match_count / processed_frames if processed_frames > 0 else 0
            avg_confidence = np.mean(confidence_scores[consistent_name])
            
            print(f"\n📊 Verification Results:")
            print(f"   Total frames: {total_frames}")
            print(f"   Processed frames: {processed_frames}")
            print(f"   Matched frames: {match_count}")
            print(f"   Consistency: {consistency:.0%} (need {MIN_CONSISTENCY:.0%})")
            print(f"   Avg confidence: {avg_confidence:.0%} (need {MIN_CONFIDENCE_SCORE:.0%})")
            print(f"   Unauthorized detections: {unauthorized_count}")
            
            # Strict verification checks
            passed_consistency = consistency >= MIN_CONSISTENCY
            passed_match_frames = match_count >= MIN_MATCH_FRAMES
            passed_confidence = avg_confidence >= MIN_CONFIDENCE_SCORE
            
            if passed_consistency and passed_match_frames and passed_confidence:
                print(f"✅ Face verified: {consistent_name}")
                self.arduino.face_verified()
                return consistent_name, avg_confidence
            else:
                print("\n❌ Verification FAILED:")
                if not passed_consistency:
                    print(f"   ❌ Consistency: {consistency:.0%} < {MIN_CONSISTENCY:.0%} (required)")
                else:
                    print(f"   ✅ Consistency: {consistency:.0%}")
                    
                if not passed_match_frames:
                    print(f"   ❌ Match frames: {match_count} < {MIN_MATCH_FRAMES} (required)")
                else:
                    print(f"   ✅ Match frames: {match_count}")
                    
                if not passed_confidence:
                    print(f"   ❌ Confidence: {avg_confidence:.0%} < {MIN_CONFIDENCE_SCORE:.0%} (required)")
                else:
                    print(f"   ✅ Confidence: {avg_confidence:.0%}")
                    
                return None, None
        
        print("❌ No face detected")
        return None, None
    
    def _find_best_match(self, current_encoding):
        """Find best matching face with strict threshold"""
        best_name = "Unknown"
        best_distance = float('inf')
        
        for name, encodings_list in self.known_encodings.items():
            distances = face_recognition.face_distance(encodings_list, current_encoding)
            min_distance = np.min(distances)
            
            if min_distance < best_distance and min_distance <= FACE_DISTANCE_THRESHOLD:
                best_distance = min_distance
                best_name = name
        
        return best_name, best_distance
    
    def delete_person(self, name):
        if name in self.known_encodings:
            # Delete images
            if os.path.exists(REGISTERED_FOLDER):
                for file in os.listdir(REGISTERED_FOLDER):
                    if file.startswith(name):
                        os.remove(os.path.join(REGISTERED_FOLDER, file))
            
            del self.known_encodings[name]
            self._save_encodings()
            print(f"✅ Deleted all data for '{name}'")
            return True
        else:
            print(f"❌ Person '{name}' not found")
            return False

# ============================================
# HAND GESTURE RECOGNITION
# ============================================
class HandGestureRecognition:
    def __init__(self, arduino_controller):
        self.arduino = arduino_controller
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
        if os.path.exists(GESTURES_FILE):
            try:
                with open(GESTURES_FILE, 'rb') as f:
                    self.registered_gestures = pickle.load(f)
                print(f"✅ Loaded {len(self.registered_gestures)} registered gestures")
            except Exception as e:
                print(f"❌ Error loading gestures: {e}")
    
    def _save_gestures(self):
        with open(GESTURES_FILE, 'wb') as f:
            pickle.dump(self.registered_gestures, f)
    
    def _extract_landmarks(self, hand_landmarks):
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def register_gesture(self, person_name):
        print(f"\n📷 Registering hand gesture for: {person_name}")
        
        if person_name in self.registered_gestures:
            print(f"⚠️ Gesture already exists for '{person_name}'")
            choice = input("Replace existing gesture? (y/n): ").strip().lower()
            if choice != 'y':
                return False
        
        print("="*60)
        print("INSTRUCTIONS:")
        print("  • Show your unique hand gesture")
        print("  • Hold it steady when detected")
        print("  • Press 's' to capture")
        print("  • Press 'q' to cancel")
        print("  • Press 'r' to restart camera if frozen")
        print("="*60)
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Initialize camera
                print(f"\n🎥 Starting camera (attempt {retry_count + 1}/{max_retries})...")
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    print("❌ Cannot access camera")
                    retry_count += 1
                    if retry_count < max_retries:
                        print("🔄 Retrying in 2 seconds...")
                        time.sleep(2)
                    continue
                
                # Camera warmup
                print("⏳ Warming up camera...")
                for _ in range(10):
                    cap.read()
                
                print("✅ Camera ready!")
                
                frame_count = 0
                last_frame_time = time.time()
                
                while True:
                    ret, frame = cap.read()
                    
                    # Check for frozen camera
                    current_time = time.time()
                    if current_time - last_frame_time > 5:
                        print("\n⚠️ Camera appears frozen!")
                        print("Press 'r' to restart camera or 'q' to cancel")
                        last_frame_time = current_time
                    
                    if not ret:
                        frame_count += 1
                        if frame_count > 30:  # If no frames after 30 attempts
                            print("\n❌ Camera error - no frames received")
                            break
                        continue
                    
                    frame_count = 0  # Reset counter on successful read
                    last_frame_time = current_time
                    
                    try:
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
                        
                        # Add help text
                        cv2.putText(frame, "s=Save | q=Quit | r=Restart Camera", (10, frame.shape[0] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.imshow("Register Gesture", frame)
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == ord('s') and results.multi_hand_landmarks:
                            landmarks = self._extract_landmarks(results.multi_hand_landmarks[0])
                            self.registered_gestures[person_name] = landmarks
                            self._save_gestures()
                            print(f"\n✅ Gesture registered for: {person_name}")
                            cap.release()
                            cv2.destroyAllWindows()
                            return True
                        
                        elif key == ord('q'):
                            print("\n⚠️ Gesture registration cancelled")
                            cap.release()
                            cv2.destroyAllWindows()
                            return False
                        
                        elif key == ord('r'):
                            print("\n🔄 Restarting camera...")
                            cap.release()
                            cv2.destroyAllWindows()
                            time.sleep(1)
                            break  # Break to retry
                    
                    except Exception as e:
                        print(f"\n⚠️ Frame processing error: {e}")
                        continue
                
                # If we broke out of the loop, clean up and retry
                cap.release()
                cv2.destroyAllWindows()
                retry_count += 1
                
                if retry_count < max_retries:
                    print("🔄 Restarting camera system...")
                    time.sleep(2)
            
            except Exception as e:
                print(f"\n❌ Camera error: {e}")
                if 'cap' in locals():
                    cap.release()
                cv2.destroyAllWindows()
                retry_count += 1
                
                if retry_count < max_retries:
                    print(f"🔄 Retrying ({retry_count}/{max_retries})...")
                    time.sleep(2)
        
        print(f"\n❌ Failed to register gesture after {max_retries} attempts")
        return False
    
    def verify_gesture(self, person_name, timeout=GESTURE_TIMEOUT):
        if person_name not in self.registered_gestures:
            print(f"❌ No gesture registered for: {person_name}")
            return False
        
        print(f"\n✋ Show your gesture within {timeout} seconds...")
        print("Press 'r' to restart camera if frozen, 'q' to cancel")
        self.arduino.gesture_required()
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("❌ Cannot access camera")
                    retry_count += 1
                    if retry_count < max_retries:
                        print("🔄 Retrying...")
                        time.sleep(1)
                    continue
                
                # Warm up
                for _ in range(5):
                    cap.read()
                
                start_time = time.time()
                registered_landmarks = self.registered_gestures[person_name]
                last_frame_time = time.time()
                
                while time.time() - start_time < timeout:
                    ret, frame = cap.read()
                    
                    # Check for frozen camera
                    current_time = time.time()
                    if current_time - last_frame_time > 3:
                        print("\n⚠️ Camera may be frozen - press 'r' to restart")
                        last_frame_time = current_time
                    
                    if not ret:
                        continue
                    
                    last_frame_time = current_time
                    
                    try:
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
                        cv2.putText(frame, "r=Restart | q=Quit", (10, frame.shape[0] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.imshow("Gesture Verification", frame)
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == ord('q'):
                            cap.release()
                            cv2.destroyAllWindows()
                            print("❌ Gesture verification cancelled")
                            return False
                        elif key == ord('r'):
                            print("🔄 Restarting camera...")
                            cap.release()
                            cv2.destroyAllWindows()
                            time.sleep(1)
                            break  # Break to retry
                    
                    except Exception as e:
                        continue
                
                cap.release()
                cv2.destroyAllWindows()
                
                # If we completed the timeout
                if time.time() - start_time >= timeout:
                    print("❌ Gesture verification timeout")
                    return False
                
                # Otherwise we broke out to retry
                retry_count += 1
                
            except Exception as e:
                print(f"❌ Camera error: {e}")
                if 'cap' in locals():
                    cap.release()
                cv2.destroyAllWindows()
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)
        
        print(f"❌ Gesture verification failed after {max_retries} attempts")
        return False
    
    def _compare_gestures(self, current, registered):
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
        if name in self.registered_gestures:
            del self.registered_gestures[name]
            self._save_gestures()
            print(f"✅ Deleted gesture for '{name}'")
            return True
        else:
            print(f"❌ No gesture found for '{name}'")
            return False
    
    def test_gesture(self, person_name, timeout=GESTURE_TEST_TIMEOUT):
        """Test gesture recognition for tweaking (10 second timer by default)
        
        Args:
            person_name: Name of person whose gesture to test
            timeout: Time limit in seconds (default 10s)
        """
        if person_name not in self.registered_gestures:
            print(f"❌ No gesture registered for: {person_name}")
            return False
        
        print(f"\n✋ Testing gesture for {person_name} ({timeout} seconds)")
        print("This is for testing/tweaking only - shows similarity %")
        print("Press 'q' to exit early")
        
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("❌ Cannot access camera")
                    retry_count += 1
                    if retry_count < max_retries:
                        print("🔄 Retrying...")
                        time.sleep(1)
                    continue
                
                # Warm up
                for _ in range(5):
                    cap.read()
                
                start_time = time.time()
                registered_landmarks = self.registered_gestures[person_name]
                best_match = 0.0  # Track best similarity seen
                
                while time.time() - start_time < timeout:
                    ret, frame = cap.read()
                    
                    if not ret:
                        continue
                    
                    remaining = timeout - (time.time() - start_time)
                    
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = self.hands.process(frame_rgb)
                        
                        if results.multi_hand_landmarks:
                            for hand_landmarks in results.multi_hand_landmarks:
                                self.mp_draw.draw_landmarks(
                                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                                )
                            
                            # Calculate similarity
                            current_landmarks = self._extract_landmarks(results.multi_hand_landmarks[0])
                            
                            # Normalize both
                            reg_norm = registered_landmarks / (np.linalg.norm(registered_landmarks) + 1e-6)
                            cur_norm = current_landmarks / (np.linalg.norm(current_landmarks) + 1e-6)
                            
                            similarity = 1 - np.linalg.norm(reg_norm - cur_norm)
                            
                            # Update best match
                            if similarity > best_match:
                                best_match = similarity
                            
                            # Color based on threshold
                            if similarity > GESTURE_MATCH_THRESHOLD:
                                color = (0, 255, 0)  # Green - matches!
                                status = "MATCH!"
                            elif similarity > GESTURE_MATCH_THRESHOLD * 0.8:
                                color = (0, 255, 255)  # Yellow - close
                                status = "CLOSE"
                            else:
                                color = (0, 165, 255)  # Orange - needs work
                                status = "ADJUST"
                            
                            cv2.putText(frame, f"Similarity: {similarity:.0%} - {status}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            cv2.putText(frame, f"Threshold: {GESTURE_MATCH_THRESHOLD:.0%}", 
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(frame, f"Best: {best_match:.0%}", 
                                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, "Show your hand gesture", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Timer display
                        cv2.putText(frame, f"Time: {remaining:.1f}s", (10, frame.shape[0] - 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, "Press 'q' to exit", (10, frame.shape[0] - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.imshow("Test Gesture", frame)
                        key = cv2.waitKey(1) & 0xFF
                        
                        if key == ord('q'):
                            print(f"\n📊 Test Results:")
                            print(f"   Best match: {best_match:.0%}")
                            print(f"   Threshold: {GESTURE_MATCH_THRESHOLD:.0%}")
                            if best_match >= GESTURE_MATCH_THRESHOLD:
                                print("   Status: ✅ Would pass verification")
                            else:
                                print("   Status: ❌ Would fail verification")
                            cap.release()
                            cv2.destroyAllWindows()
                            return True
                    
                    except Exception as e:
                        continue
                
                # Timeout reached
                cap.release()
                cv2.destroyAllWindows()
                print(f"\n📊 Test Results:")
                print(f"   Best match: {best_match:.0%}")
                print(f"   Threshold: {GESTURE_MATCH_THRESHOLD:.0%}")
                if best_match >= GESTURE_MATCH_THRESHOLD:
                    print("   Status: ✅ Would pass verification")
                else:
                    print("   Status: ❌ Would fail verification")
                return True
            
            except Exception as e:
                print(f"❌ Camera error: {e}")
                if 'cap' in locals():
                    cap.release()
                cv2.destroyAllWindows()
                retry_count += 1
                
                if retry_count < max_retries:
                    print(f"🔄 Retrying ({retry_count}/{max_retries})...")
                    time.sleep(2)
        
        print(f"\n❌ Failed to test gesture after {max_retries} attempts")
        return False
    
# ============================================
# MAIN SECURITY SYSTEM
# ============================================
class SecuritySystem:
    def __init__(self):
        print("\n" + "="*60)
        print("  DUAL AUTHENTICATION SECURITY SYSTEM")
        print("  ⚡ Auto-Sampling + Strict Verification ⚡")
        print("  Continuous Servo Control")
        print("="*60)
        print(f"  🔒 Security Configuration:")
        print(f"     Registration:")
        print(f"       • {SAMPLES_PER_PERSON} samples per person")
        print(f"       • {MIN_FACE_CONFIDENCE:.0%} min face confidence")
        print(f"     Verification:")
        print(f"       • {FACE_DISTANCE_THRESHOLD} distance threshold")
        print(f"       • {MIN_CONSISTENCY:.0%} consistency required")
        print(f"       • {MIN_MATCH_FRAMES} min match frames")
        print(f"       • {MIN_CONFIDENCE_SCORE:.0%} min confidence score")
        print("="*60)
        
        self.arduino = ArduinoController()
        self.face_recognition = FaceRecognition(self.arduino)
        self.gesture_recognition = HandGestureRecognition(self.arduino)
        
        self.arduino.system_ready()
    
    def start_system(self):
        """Start system monitoring with sonar sensor integration"""
        print("\n" + "="*60)
        print("  🚀 STARTING SECURITY SYSTEM")
        print("  Sonar Sensor: ACTIVE (30cm detection range)")
        print("  Face Recognition: STAND BY")
        print("="*60)
        print("📡 Monitoring for person detection...")
        print("👤 Step within 30cm of sensor to trigger authentication")
        print("⏹️  Press Ctrl+C to stop monitoring and return to menu")
        print("="*60)
        
        # Send ready command to Arduino to ensure it's monitoring
        self.arduino.system_ready()
        
        # Start monitoring for person detection
        if not self._monitor_person_detection():
            print("\n👋 System monitoring stopped")
            return
        
        # Person detected - start authentication process
        print("\n" + "="*60)
        print("  👤 PERSON DETECTED - STARTING AUTHENTICATION")
        print("="*60)
        
        # Perform authentication
        auth_success = self.authenticate_person()
        
        if auth_success:
            print("\n✅ Authentication completed successfully!")
            print("🔄 Returning to monitoring...")
            time.sleep(3)
            # Continue monitoring after successful auth
            self.start_system()
        else:
            print("\n❌ Authentication failed!")
            print("🔄 Returning to monitoring...")
            time.sleep(3)
            # Continue monitoring after failed auth
            self.start_system()
    
    def _monitor_person_detection(self):
        """Monitor serial port for PERSON_DETECTED signal from Arduino"""
        if not self.arduino.connected:
            print("⚠️ Arduino not connected - cannot monitor sonar sensor")
            print("💡 Please connect Arduino and restart system")
            input("Press Enter to return to menu...")
            return False
        
        try:
            # Clear any pending data
            self.arduino.arduino.reset_input_buffer()
            
            print("\n🔍 Listening for person detection...")
            start_time = time.time()
            
            while True:
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check if Arduino has sent data
                if self.arduino.arduino.in_waiting > 0:
                    try:
                        line = self.arduino.arduino.readline().decode('utf-8').strip()
                        
                        if line == "PERSON_DETECTED":
                            print("\n🎯 PERSON DETECTED! Starting authentication...")
                            return True
                        elif line.startswith("DISTANCE:"):
                            # Optional: Show distance readings for debugging
                            distance_str = line.replace("DISTANCE:", "")
                            if distance_str != "ERROR":
                                try:
                                    distance = float(distance_str.replace("cm", ""))
                                    if distance <= 50:  # Show if within 50cm
                                        print(f"📏 Distance: {distance:.1f}cm", end='\r')
                                except:
                                    pass
                        elif line:
                            print(f"📨 Arduino: {line}")
                            
                    except Exception as e:
                        print(f"⚠️ Error reading from Arduino: {e}")
                        continue
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
                # Show heartbeat indicator every 5 seconds
                if int(elapsed) % 5 == 0 and int(elapsed * 10) % 50 == 0:
                    print(f"💓 Monitoring... ({int(elapsed)}s)", end='\r')
        
        except KeyboardInterrupt:
            print("\n⏹️  Stopping monitoring...")
            return False
        except Exception as e:
            print(f"❌ Error monitoring person detection: {e}")
            input("Press Enter to return to menu...")
            return False
    
    def register_person(self, num_samples=SAMPLES_PER_PERSON):
        print("\n--- REGISTRATION PROCESS ---")
        
        print(f"\n[1/2] Face Registration ({num_samples} auto-captured samples)")
        if not self.face_recognition.register_face(num_samples=num_samples):
            print("❌ Face registration failed")
            return False
        
        name = list(self.face_recognition.known_encodings.keys())[-1]
        
        print("\n[2/2] Gesture Registration")
        print("⚠️ If camera freezes, press 'r' to restart or 'q' to skip")
        
        if not self.gesture_recognition.register_gesture(name):
            print("\n⚠️ Gesture registration incomplete")
            choice = input("Keep face registration without gesture? (y/n): ").strip().lower()
            if choice == 'y':
                print(f"✅ {name} registered with FACE ONLY")
                print("💡 You can add gesture later using Option 4")
                return True
            else:
                self.face_recognition.delete_person(name)
                print("❌ Registration cancelled - face data deleted")
                return False
        
        print(f"\n✅ {name} registered successfully with both face and gesture!")
        return True
    
    def register_gesture_for_user(self):
        """Register or update gesture for existing user"""
        self.list_registered_users()
        
        if not self.face_recognition.known_encodings:
            print("\n❌ No registered users. Register a person first (Option 1)")
            return False
        
        name = input("\nEnter person name to add/update gesture: ").strip()
        
        if name not in self.face_recognition.known_encodings:
            print(f"❌ Person '{name}' not found")
            return False
        
        print(f"\n🎯 Registering gesture for: {name}")
        return self.gesture_recognition.register_gesture(name)
    
    def authenticate_person(self):
        print("\n" + "="*60)
        print("  AUTHENTICATION PROCESS")
        print("="*60)
        
        print("\n[1/2] Face Verification")
        print(f"📌 {FACE_VERIFICATION_TIME}-second timer with auto-deny after 5 unauthorized faces")
        person_name, confidence = self.face_recognition.verify_face_continuous(duration=FACE_VERIFICATION_TIME)
        
        if person_name is None:
            print("\n❌ AUTHENTICATION FAILED: Face not verified")
            self.arduino.access_denied()
            return False
        
        print(f"\n✅ Face verified: {person_name} ({confidence:.0%} confidence)")
        
        print("\n[2/2] Gesture Verification")
        gesture_verified = self.gesture_recognition.verify_gesture(person_name)
        
        if gesture_verified:
            print("\n" + "="*60)
            print(f"  ✅ AUTHENTICATION SUCCESSFUL")
            print(f"  Welcome, {person_name}!")
            print(f"  🚪 ACCESS GRANTED")
            print("="*60)
            
            self.arduino.access_granted()
            return True
        else:
            print("\n" + "="*60)
            print(f"  ❌ AUTHENTICATION FAILED")
            print(f"  Face verified but gesture did not match")
            print(f"  🚪 ACCESS DENIED")
            print("="*60)
            self.arduino.access_denied()
            return False
    
    def list_registered_users(self):
        print("\n📋 Registered Users:")
        if not self.face_recognition.known_encodings:
            print("  No users registered")
            return
        
        for name, encodings in self.face_recognition.known_encodings.items():
            has_gesture = "✅" if name in self.gesture_recognition.registered_gestures else "❌"
            sample_count = len(encodings)
            print(f"  • {name} - Face Samples: {sample_count} | Gesture: {has_gesture}")
    
    def delete_user(self):
        self.list_registered_users()
        name = input("\nEnter name to delete: ").strip()
        
        if not name:
            print("❌ Name cannot be empty")
            return
        
        confirm = input(f"⚠️ Delete ALL data for '{name}'? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Cancelled")
            return
        
        face_deleted = self.face_recognition.delete_person(name)
        gesture_deleted = self.gesture_recognition.delete_gesture(name)
        
        if face_deleted or gesture_deleted:
            print(f"✅ User '{name}' deleted successfully")
        else:
            print(f"❌ User '{name}' not found")
    
    def delete_user_by_name(self, name):
        """Delete user by name (for GUI)"""
        face_deleted = self.face_recognition.delete_person(name)
        gesture_deleted = self.gesture_recognition.delete_gesture(name)
        return face_deleted or gesture_deleted
    
    def add_face_or_gesture(self):
        """Add face samples or gesture for existing user"""
        self.list_registered_users()
        
        if not self.face_recognition.known_encodings:
            print("\n❌ No registered users. Register a person first (Option 1)")
            return False
        
        print("\n" + "-"*40)
        print("What would you like to add?")
        print("1. Add More Face Samples")
        print("2. Add/Update Gesture")
        print("-"*40)
        
        choice = input("Choose (1-2): ").strip()
        
        if choice == "1":
            # Add face samples
            name = input("\nEnter person name: ").strip()
            if name in self.face_recognition.known_encodings:
                num = input(f"How many samples? (default {SAMPLES_PER_PERSON}): ").strip()
                num = int(num) if num.isdigit() else SAMPLES_PER_PERSON
                self.face_recognition.register_face(name=name, num_samples=num)
            else:
                print(f"❌ Person '{name}' not found")
                
        elif choice == "2":
            # Add/update gesture
            name = input("\nEnter person name to add/update gesture: ").strip()
            
            if name not in self.face_recognition.known_encodings:
                print(f"❌ Person '{name}' not found")
                return False
            
            print(f"\n🎯 Registering gesture for: {name}")
            return self.gesture_recognition.register_gesture(name)
        else:
            print("❌ Invalid option")
            return False
    
    def test_face_or_gesture(self):
        """Test face or gesture recognition (for tweaking)"""
        print("\n" + "-"*40)
        print("What would you like to test?")
        print("1. Test Face Recognition")
        print("2. Test Gesture Recognition")
        print("-"*40)
        
        choice = input("Choose (1-2): ").strip()
        
        if choice == "1":
            # Test face recognition - no timer, auto-stop on unknown
            print("\n👁️ Face Recognition Testing Mode")
            print("This will scan continuously until unknown faces detected")
            print("Perfect for tweaking settings and checking recognition quality")
            name, conf = self.face_recognition.verify_face_continuous(duration=None)
            if name:
                print(f"\n✅ Recognition test complete")
                print(f"   Detected: {name}")
                print(f"   Confidence: {conf:.0%}")
            else:
                print("\n❌ Test stopped (unknown faces or manual exit)")
                
        elif choice == "2":
            # Test gesture recognition - 10 second timer
            self.list_registered_users()
            person_name = input("\nEnter person name to test gesture: ").strip()
            
            if person_name not in self.face_recognition.known_encodings:
                print(f"❌ Person '{person_name}' not found")
                return False
            
            self.gesture_recognition.test_gesture(person_name, timeout=GESTURE_TEST_TIMEOUT)
        else:
            print("❌ Invalid option")
            return False
    
    def test_servo_manual(self):
        """Manual servo control for testing"""
        print("\n" + "-"*40)
        print("Manual Servo Control:")
        print("1. Rotate Right")
        print("2. Rotate Left")
        print("3. Stop")
        print("4. Set Speed")
        print("5. Back to Main Menu")
        print("-"*40)
        
        while True:
            choice = input("Choose (1-5): ").strip()
            
            if choice == "1":
                self.arduino.servo_right()
            elif choice == "2":
                self.arduino.servo_left()
            elif choice == "3":
                self.arduino.servo_stop()
            elif choice == "4":
                speed = input("Enter speed (0-100): ").strip()
                if speed.isdigit():
                    speed_val = int(speed)
                    if 0 <= speed_val <= 100:
                        self.arduino.servo_set_speed(speed_val)
                    else:
                        print("❌ Speed must be between 0-100")
                else:
                    print("❌ Invalid speed value")
            elif choice == "5":
                break
            else:
                print("❌ Invalid option")
    
    def close(self):
        self.arduino.close()

# ============================================
# MAIN PROGRAM
# ============================================
def main():
    if not FACE_REC_AVAILABLE:
        print("\n" + "="*60)
        print("⚠️ WARNING: face_recognition not installed!")
        print("="*60)
        print("Run setup.py first or install manually:")
        print("  pip install face-recognition")
        print("="*60)
        return
    
    # Create necessary folders
    os.makedirs(REGISTERED_FOLDER, exist_ok=True)
    
    system = SecuritySystem()
    
    try:
        while True:
            print("\n" + "-"*60)
            print("MENU:")
            print("1. Start System (Sonar + Face + Gesture)")
            print("2. Register New Person (Auto-Capture)")
            print("3. Authenticate Person (Face + Gesture)")
            print("4. List Registered Users")
            print("5. Add More Face or Gesture for Existing User")
            print("6. Test Face or Gesture")
            print("7. Delete User")
            print("8. Test Arduino")
            print("9. Test Sonar Sensor")
            print("10. Test Servo System")
            print("11. Manual Servo Control")
            print("12. Exit")
            print("-"*60)
            
            choice = input("Choose option (1-12): ").strip()
            
            if choice == "1":
                system.start_system()
            
            elif choice == "2":
                system.register_person()
            
            elif choice == "3":
                system.authenticate_person()
            
            elif choice == "4":
                system.list_registered_users()
            
            elif choice == "5":
                system.add_face_or_gesture()
            
            elif choice == "6":
                system.test_face_or_gesture()
            
            elif choice == "7":
                system.delete_user()
            
            elif choice == "8":
                print("\n🔧 Testing Arduino...")
                if system.arduino.connected:
                    system.arduino.send_command("test")
                    print("✅ Test command sent")
                else:
                    print("❌ Arduino not connected")
            
            elif choice == "9":
                system.arduino.test_sonar()
            
            elif choice == "10":
                system.arduino.test_servo()
            
            elif choice == "11":
                system.test_servo_manual()
            
            elif choice == "12":
                print("\n👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid option")
    
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
    finally:
        system.close()

if __name__ == "__main__":
    main()