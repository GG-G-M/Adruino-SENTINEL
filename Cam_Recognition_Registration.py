import cv2
import os
import numpy as np
from PIL import Image
import pickle

# Folders
REGISTERED_FOLDER = "authorized_faces"
ENCODINGS_FILE = "face_encodings.pkl"

def test_face_recognition():
    """Test if face_recognition works and return available methods"""
    available_methods = []
    
    # Test 1: Basic import
    try:
        import face_recognition
        available_methods.append('face_recognition')
        print("✅ face_recognition imported successfully")
    except ImportError as e:
        print(f"❌ face_recognition import failed: {e}")
        return available_methods
    
    # Test 2: Face locations
    try:
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        face_locations = face_recognition.face_locations(test_image)
        available_methods.append('face_locations')
        print("✅ face_locations works")
    except Exception as e:
        print(f"❌ face_locations failed: {e}")
    
    # Test 3: Face encodings
    try:
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        face_encodings = face_recognition.face_encodings(test_image)
        available_methods.append('face_encodings')
        print("✅ face_encodings works")
    except Exception as e:
        print(f"❌ face_encodings failed: {e}")
    
    return available_methods

# Test what works
print("=== TESTING FACE RECOGNITION CAPABILITIES ===")
available_methods = test_face_recognition()

if 'face_recognition' in available_methods:
    import face_recognition

def register_face_robust():
    """Register face with multiple fallback methods"""
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
            # Save the image
            face_image_path = os.path.join(REGISTERED_FOLDER, f"{name}.jpg")
            
            # Convert to RGB and save with PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_image.save(face_image_path, 'JPEG', quality=95)
            
            print(f"✅ Image saved: {face_image_path}")
            
            # Try to extract face encoding
            encoding = extract_encoding_robust(rgb_frame, face_image_path)
            
            if encoding is not None:
                save_face_encoding(name, encoding)
                print(f"✅ Face encoding saved for: {name}")
            else:
                print("⚠️  Could not extract face encoding - using image-based recognition")
                
            break
        elif key == ord('q'):
            print("Registration cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()

def extract_encoding_robust(rgb_frame, image_path):
    """Try multiple methods to extract face encoding"""
    encoding = None
    
    # Method 1: Try face_recognition with face_encodings
    if 'face_encodings' in available_methods:
        try:
            face_encodings = face_recognition.face_encodings(rgb_frame)
            if face_encodings:
                encoding = face_encodings[0]
                print("✅ Encoding extracted using face_recognition")
                return encoding
        except Exception as e:
            print(f"❌ face_recognition encoding failed: {e}")
    
    # Method 2: Try face_recognition with load_image_file
    if 'face_recognition' in available_methods:
        try:
            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)
            if face_encodings:
                encoding = face_encodings[0]
                print("✅ Encoding extracted using load_image_file")
                return encoding
        except Exception as e:
            print(f"❌ load_image_file encoding failed: {e}")
    
    # Method 3: Create basic encoding using OpenCV
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
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        
        # Extract face region
        face_region = rgb_frame[y:y+h, x:x+w]
        
        # Resize to standard size
        face_standard = cv2.resize(face_region, (100, 100))
        
        # Convert to different color spaces and create histograms
        hsv_face = cv2.cvtColor(face_standard, cv2.COLOR_RGB2HSV)
        lab_face = cv2.cvtColor(face_standard, cv2.COLOR_RGB2LAB)
        
        # Create feature vector from histograms
        hist_hsv = cv2.calcHist([hsv_face], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_lab = cv2.calcHist([lab_face], [0, 1], None, [50, 60], [0, 255, 0, 255])
        
        # Normalize and flatten
        hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
        hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
        
        # Combine features
        encoding = np.concatenate([hist_hsv, hist_lab])
        
        return encoding
    
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
                # Handle old format
                names.append(name)
                encodings.append(data)
        
        return encodings, names
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return [], []

def recognize_faces_robust():
    """Recognize faces with multiple fallback methods"""
    known_encodings, known_names = load_face_encodings()
    
    if not known_encodings:
        print("No registered faces found. Please register first.")
        return
    
    print(f"Loaded {len(known_encodings)} registered faces: {', '.join(known_names)}")
    
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit recognition")
    
    # Initialize OpenCV face detector as fallback
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        recognized_faces = []
        
        # Method 1: Try face_recognition if available
        if 'face_encodings' in available_methods:
            try:
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    name = "Unknown"
                    confidence = 0
                    
                    # Compare with known faces
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_names[best_match_index]
                            confidence = 1 - face_distances[best_match_index]
                    
                    recognized_faces.append({
                        'location': (left, top, right, bottom),
                        'name': name,
                        'confidence': confidence
                    })
                    
            except Exception as e:
                print(f"face_recognition failed: {e}")
        
        # Method 2: Fallback to OpenCV if no faces found or face_recognition failed
        if not recognized_faces:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                for (x, y, w, h) in faces:
                    # Extract face region for basic comparison
                    face_region = rgb_frame[y:y+h, x:x+w]
                    current_encoding = create_basic_encoding_from_region(face_region)
                    
                    name = "Unknown"
                    confidence = 0
                    
                    if current_encoding is not None:
                        # Basic comparison with known encodings
                        name, confidence = compare_basic_encodings(current_encoding, known_encodings, known_names)
                    
                    recognized_faces.append({
                        'location': (x, y, x+w, y+h),
                        'name': name,
                        'confidence': confidence
                    })
                    
            except Exception as e:
                print(f"OpenCV face detection failed: {e}")
        
        # Draw results on frame
        for face in recognized_faces:
            left, top, right, bottom = face['location']
            name = face['name']
            confidence = face['confidence']
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            label = f"{name} ({confidence:.2f})" if name != "Unknown" else "Unknown"
            cv2.putText(frame, label, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Print to terminal
            if name != "Unknown":
                print(f"✅ Authorized: {name} (confidence: {confidence:.2f})")
            else:
                print(f"❌ Unknown person detected")
        
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
            
        # Resize to standard size
        face_standard = cv2.resize(face_region, (100, 100))
        
        # Convert color spaces
        hsv_face = cv2.cvtColor(face_standard, cv2.COLOR_RGB2HSV)
        lab_face = cv2.cvtColor(face_standard, cv2.COLOR_RGB2LAB)
        
        # Create histograms
        hist_hsv = cv2.calcHist([hsv_face], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist_lab = cv2.calcHist([lab_face], [0, 1], None, [50, 60], [0, 255, 0, 255])
        
        # Normalize and combine
        hist_hsv = cv2.normalize(hist_hsv, hist_hsv).flatten()
        hist_lab = cv2.normalize(hist_lab, hist_lab).flatten()
        
        return np.concatenate([hist_hsv, hist_lab])
        
    except Exception as e:
        print(f"Basic encoding from region failed: {e}")
        return None

def compare_basic_encodings(current_encoding, known_encodings, known_names, threshold=0.7):
    """Compare basic encodings using histogram correlation"""
    best_match = "Unknown"
    best_confidence = 0
    
    for i, known_encoding in enumerate(known_encodings):
        try:
            if isinstance(current_encoding, np.ndarray) and isinstance(known_encoding, np.ndarray):
                # Ensure same length
                min_len = min(len(current_encoding), len(known_encoding))
                current_vec = current_encoding[:min_len]
                known_vec = known_encoding[:min_len]
                
                # Calculate correlation
                correlation = np.corrcoef(current_vec, known_vec)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
                
                if correlation > best_confidence and correlation > threshold:
                    best_confidence = correlation
                    best_match = known_names[i]
                    
        except Exception as e:
            continue
    
    return best_match, best_confidence

def main():
    print("\n=== FACE RECOGNITION SYSTEM ===")
    print(f"Available methods: {available_methods}")
    
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
    # Create necessary folders
    if not os.path.exists(REGISTERED_FOLDER):
        os.makedirs(REGISTERED_FOLDER)
    
    main()