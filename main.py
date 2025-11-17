import cv2
import os
import numpy as np
from PIL import Image
import pickle

# ============================================
# CONFIGURATION - ADJUST CONFIDENCE HERE
# ============================================
REGISTERED_FOLDER = "authorized_faces"
ENCODINGS_FILE = "face_encodings.pkl"

# ⚠️ IMPORTANT: Adjust this threshold to prevent false positives
# Higher value = stricter matching (fewer false positives)
# Lower value = more lenient matching (may allow unauthorized access)
# Recommended range: 0.5 to 0.7
CONFIDENCE_THRESHOLD = 0.6  # 👈 CHANGE THIS VALUE TO ADJUST SECURITY

# For face_recognition library (if available)
FR_TOLERANCE = 1 - CONFIDENCE_THRESHOLD  # Auto-calculated from threshold

# For OpenCV histogram method (fallback)
OPENCV_THRESHOLD = 0.85  # 👈 For histogram correlation (0.80-0.95 recommended)
# ============================================

def test_face_recognition():
    """Test if face_recognition works and return available methods"""
    available_methods = []
    
    try:
        import face_recognition
        available_methods.append('face_recognition')
        print("✅ face_recognition imported successfully")
        
        # Test face encodings
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        face_recognition.face_encodings(test_image)
        available_methods.append('face_encodings')
        print("✅ face_encodings works")
    except (ImportError, Exception) as e:
        print(f"⚠️  face_recognition unavailable: {e}")
        print("Using OpenCV fallback method")
    
    return available_methods

# Test capabilities
print("=== TESTING FACE RECOGNITION CAPABILITIES ===")
available_methods = test_face_recognition()
print(f"Active confidence threshold: {CONFIDENCE_THRESHOLD}")
print("="*50 + "\n")

if 'face_recognition' in available_methods:
    import face_recognition

def register_face_robust():
    """Register face with multiple fallback methods"""
    name = input("Enter your name: ").strip()
    if not name:
        print("❌ Name cannot be empty")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera")
        return
    
    print("Press 's' to take a snapshot, 'q' to quit registration")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Draw detection box to help user position
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Press 's' to capture", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            face_image_path = os.path.join(REGISTERED_FOLDER, f"{name}.jpg")
            
            # Convert to RGB and save
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image.save(face_image_path, 'JPEG', quality=95)
            
            print(f"✅ Image saved: {face_image_path}")
            
            # Extract face encoding
            encoding = extract_encoding_robust(rgb_frame, face_image_path)
            
            if encoding is not None:
                save_face_encoding(name, encoding)
                print(f"✅ Face encoding saved for: {name}")
            else:
                print("⚠️  Could not extract face encoding")
                
            break
        elif key == ord('q'):
            print("Registration cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()

def extract_encoding_robust(rgb_frame, image_path):
    """Try multiple methods to extract face encoding"""
    
    # Method 1: face_recognition library
    if 'face_encodings' in available_methods:
        try:
            face_encodings = face_recognition.face_encodings(rgb_frame)
            if face_encodings:
                print("✅ Encoding extracted using face_recognition")
                return face_encodings[0]
        except Exception as e:
            print(f"❌ face_recognition encoding failed: {e}")
    
    # Method 2: OpenCV histogram-based encoding
    try:
        encoding = create_basic_encoding(rgb_frame)
        if encoding is not None:
            print("✅ Basic encoding created using OpenCV")
            return encoding
    except Exception as e:
        print(f"❌ Basic encoding failed: {e}")
    
    return None

def create_basic_encoding(rgb_frame):
    """Create a basic face encoding using OpenCV features"""
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_region = rgb_frame[y:y+h, x:x+w]
        face_standard = cv2.resize(face_region, (100, 100))
        
        # Create color histograms
        hsv_face = cv2.cvtColor(face_standard, cv2.COLOR_RGB2HSV)
        lab_face = cv2.cvtColor(face_standard, cv2.COLOR_RGB2LAB)
        
        hist_hsv = cv2.calcHist([hsv_face], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_lab = cv2.calcHist([lab_face], [0, 1], None, [50, 60], [0, 255, 0, 255])
        
        hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
        hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
        
        return np.concatenate([hist_hsv, hist_lab])
    
    return None

def save_face_encoding(name, encoding):
    """Save face encoding to pickle file"""
    encodings_data = {}
    
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, 'rb') as f:
                encodings_data = pickle.load(f)
        except:
            encodings_data = {}
    
    encodings_data[name] = {
        'encoding': encoding,
        'type': type(encoding).__name__
    }
    
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump(encodings_data, f)

def load_face_encodings():
    """Load face encodings from pickle file"""
    if not os.path.exists(ENCODINGS_FILE):
        return [], []
    
    try:
        with open(ENCODINGS_FILE, 'rb') as f:
            encodings_data = pickle.load(f)
        
        names = []
        encodings = []
        
        for name, data in encodings_data.items():
            if isinstance(data, dict) and 'encoding' in data:
                names.append(name)
                encodings.append(data['encoding'])
            else:
                names.append(name)
                encodings.append(data)
        
        return encodings, names
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return [], []

def recognize_faces_robust():
    """Recognize faces with persistent bounding boxes"""
    known_encodings, known_names = load_face_encodings()
    
    if not known_encodings:
        print("No registered faces found. Please register first.")
        return
    
    print(f"Loaded {len(known_encodings)} registered faces: {', '.join(known_names)}")
    print(f"Current confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"OpenCV threshold: {OPENCV_THRESHOLD}")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot access camera")
        return
        
    print("Press 'q' to quit recognition")
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Store last known face positions to keep boxes visible
    last_recognized_faces = []
    frames_without_detection = 0
    MAX_FRAMES_TO_KEEP = 5  # Keep boxes for 5 frames after face disappears
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        recognized_faces = []
        
        # ============================================
        # FACE DETECTION & RECOGNITION
        # ============================================
        
        # Method 1: Try face_recognition if available
        if 'face_encodings' in available_methods:
            try:
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    name = "Unknown"
                    confidence = 0
                    
                    # Compare with known faces using configured threshold
                    matches = face_recognition.compare_faces(
                        known_encodings, 
                        face_encoding, 
                        tolerance=FR_TOLERANCE  # 👈 Uses threshold from config
                    )
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        confidence = 1 - face_distances[best_match_index]
                        
                        # ⚠️ SECURITY CHECK: Only accept if confidence meets threshold
                        if matches[best_match_index] and confidence >= CONFIDENCE_THRESHOLD:
                            name = known_names[best_match_index]
                        else:
                            name = "Unknown"
                    
                    recognized_faces.append({
                        'location': (left, top, right, bottom),
                        'name': name,
                        'confidence': confidence
                    })
                    
            except Exception as e:
                print(f"face_recognition failed: {e}")
        
        # Method 2: Fallback to OpenCV if no faces found
        if not recognized_faces:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    face_region = rgb_frame[y:y+h, x:x+w]
                    current_encoding = create_basic_encoding_from_region(face_region)
                    
                    name = "Unknown"
                    confidence = 0
                    
                    if current_encoding is not None:
                        # Compare using OpenCV threshold
                        name, confidence = compare_basic_encodings(
                            current_encoding, 
                            known_encodings, 
                            known_names, 
                            threshold=OPENCV_THRESHOLD  # 👈 Uses threshold from config
                        )
                    
                    recognized_faces.append({
                        'location': (x, y, x+w, y+h),
                        'name': name,
                        'confidence': confidence
                    })
                    
            except Exception as e:
                print(f"OpenCV face detection failed: {e}")
        
        # ============================================
        # PERSISTENT BOUNDING BOXES
        # ============================================
        
        # Update last known faces or keep previous ones
        if recognized_faces:
            last_recognized_faces = recognized_faces
            frames_without_detection = 0
        else:
            frames_without_detection += 1
            # Keep showing last faces for a few frames
            if frames_without_detection <= MAX_FRAMES_TO_KEEP:
                recognized_faces = last_recognized_faces
        
        # Draw results on frame
        for face in recognized_faces:
            left, top, right, bottom = face['location']
            name = face['name']
            confidence = face['confidence']
            
            # Color coding: Green for authorized, Red for unknown
            if name != "Unknown":
                color = (0, 255, 0)  # GREEN for authorized
                status = "✅ AUTHORIZED"
            else:
                color = (0, 0, 255)  # RED for unauthorized
                status = "❌ UNAUTHORIZED"
            
            # Draw rectangle (thicker for better visibility)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
            
            # Draw label background
            label = f"{name} ({confidence:.2%})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (left, top-35), (left + label_size[0], top), color, -1)
            
            # Draw text
            cv2.putText(frame, label, (left, top-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display threshold info on screen
        info_text = f"Threshold: {CONFIDENCE_THRESHOLD:.2f}"
        cv2.putText(frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def create_basic_encoding_from_region(face_region):
    """Create basic encoding from face region"""
    try:
        if face_region.size == 0:
            return None
            
        face_standard = cv2.resize(face_region, (100, 100))
        
        hsv_face = cv2.cvtColor(face_standard, cv2.COLOR_RGB2HSV)
        lab_face = cv2.cvtColor(face_standard, cv2.COLOR_RGB2LAB)
        
        hist_hsv = cv2.calcHist([hsv_face], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_lab = cv2.calcHist([lab_face], [0, 1], None, [50, 60], [0, 255, 0, 255])
        
        hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
        hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
        
        return np.concatenate([hist_hsv, hist_lab])
        
    except Exception as e:
        return None

def compare_basic_encodings(current_encoding, known_encodings, known_names, threshold=OPENCV_THRESHOLD):
    """Compare basic encodings using histogram correlation"""
    best_match = "Unknown"
    best_confidence = 0
    
    for i, known_encoding in enumerate(known_encodings):
        try:
            if isinstance(current_encoding, np.ndarray) and isinstance(known_encoding, np.ndarray):
                min_len = min(len(current_encoding), len(known_encoding))
                current_vec = current_encoding[:min_len]
                known_vec = known_encoding[:min_len]
                
                correlation = np.corrcoef(current_vec, known_vec)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
                
                # ⚠️ SECURITY CHECK: Must meet threshold
                if correlation > best_confidence and correlation >= threshold:
                    best_confidence = correlation
                    best_match = known_names[i]
                    
        except Exception:
            continue
    
    return best_match, best_confidence

def main():
    print("\n" + "="*50)
    print("   FACE RECOGNITION SYSTEM")
    print("="*50)
    print(f"Security Level: {CONFIDENCE_THRESHOLD:.2f}")
    print(f"Available methods: {available_methods}")
    print("="*50)
    
    while True:
        print("\nOptions:")
        print("1. Register new face")
        print("2. Recognize faces")
        print("3. Check registered faces")
        print("4. Exit")
        
        choice = input("Choose option (1-4): ").strip()
        
        if choice == "1":
            register_face_robust()
        elif choice == "2":
            recognize_faces_robust()
        elif choice == "3":
            known_encodings, known_names = load_face_encodings()
            print(f"\nRegistered faces ({len(known_names)}):")
            for name in known_names:
                print(f"  - {name}")
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid option")

if __name__ == "__main__":
    if not os.path.exists(REGISTERED_FOLDER):
        os.makedirs(REGISTERED_FOLDER)
    
    main()