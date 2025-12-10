import cv2
import os
import numpy as np
from PIL import Image
import sys

def diagnose_environment():
    print("=== FACE RECOGNITION DIAGNOSTIC ===")
    print(f"Python version: {sys.version}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"PIL version: {Image.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    # Test face_recognition import with error handling
    try:
        import face_recognition
        print(f"face_recognition version: {face_recognition.__version__}")
    except ImportError as e:
        print(f"face_recognition: NOT INSTALLED - {e}")
    except Exception as e:
        print(f"face_recognition: Error - {e}")
    print()

def diagnose_image_loading():
    REGISTERED_FOLDER = "authorized_faces"
    
    if not os.path.exists(REGISTERED_FOLDER):
        print(f"Folder '{REGISTERED_FOLDER}' does not exist")
        return
    
    image_files = [f for f in os.listdir(REGISTERED_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    if not image_files:
        print(f"No image files found in '{REGISTERED_FOLDER}'")
        return
    
    print("Testing image loading methods...")
    
    for filename in image_files:
        path = os.path.join(REGISTERED_FOLDER, filename)
        print(f"\n--- Testing {filename} ---")
        
        # Method 1: OpenCV
        try:
            img_cv = cv2.imread(path)
            if img_cv is None:
                print("❌ OpenCV: Failed to load image")
            else:
                print(f"✅ OpenCV: Loaded - Shape: {img_cv.shape}, dtype: {img_cv.dtype}")
        except Exception as e:
            print(f"❌ OpenCV Error: {e}")
        
        # Method 2: PIL
        try:
            img_pil = Image.open(path)
            print(f"✅ PIL: Loaded - Size: {img_pil.size}, Mode: {img_pil.mode}")
            img_pil_array = np.array(img_pil)
            print(f"   Converted to array - Shape: {img_pil_array.shape}, dtype: {img_pil_array.dtype}")
        except Exception as e:
            print(f"❌ PIL Error: {e}")
        
        # Method 3: face_recognition's loader (only if available)
        try:
            import face_recognition
            img_fr = face_recognition.load_image_file(path)
            print(f"✅ face_recognition: Loaded - Shape: {img_fr.shape}, dtype: {img_fr.dtype}")
            
            # Test face encoding with error handling
            try:
                encodings = face_recognition.face_encodings(img_fr)
                print(f"   Face encodings found: {len(encodings)}")
            except Exception as encoding_error:
                print(f"   ❌ Face encoding failed: {encoding_error}")
                
        except ImportError:
            print("❌ face_recognition: Not available for testing")
        except Exception as e:
            print(f"❌ face_recognition Error: {e}")

def create_test_image():
    """Create a simple test image to verify the pipeline"""
    print("\n--- Creating test image ---")
    try:
        # Create a simple RGB image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        test_path = "test_image.jpg"
        
        # Save with PIL
        pil_img = Image.fromarray(test_image)
        pil_img.save(test_path, 'JPEG')
        print(f"✅ Test image saved as {test_path}")
        
        # Try to load with face_recognition if available
        try:
            import face_recognition
            loaded_img = face_recognition.load_image_file(test_path)
            print(f"✅ Test image loaded successfully - Shape: {loaded_img.shape}, dtype: {loaded_img.dtype}")
            
            # Test face encoding on test image
            try:
                encodings = face_recognition.face_encodings(loaded_img)
                print(f"✅ Face encoding test: {len(encodings)} encodings created")
            except Exception as encoding_error:
                print(f"❌ Face encoding test failed: {encoding_error}")
                
        except ImportError:
            print("⚠️  face_recognition not available for loading test")
        except Exception as e:
            print(f"❌ Test image loading failed: {e}")
        
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
            print(f"✅ Test image cleaned up")
            
        return True
        
    except Exception as e:
        print(f"❌ Test image creation failed: {e}")
        return False

def test_face_recognition_functionality():
    """Test specific face_recognition functionality"""
    print("\n--- Testing Face Recognition Functionality ---")
    
    try:
        import face_recognition
        
        # Test 1: Basic functionality with a simple image
        print("1. Testing basic face recognition...")
        test_image = np.random.randint(100, 200, (150, 150, 3), dtype=np.uint8)
        
        # Test face locations
        try:
            face_locations = face_recognition.face_locations(test_image)
            print(f"   ✅ face_locations: Found {len(face_locations)} faces")
        except Exception as e:
            print(f"   ❌ face_locations failed: {e}")
        
        # Test face encodings
        try:
            face_encodings = face_recognition.face_encodings(test_image)
            print(f"   ✅ face_encodings: Created {len(face_encodings)} encodings")
        except Exception as e:
            print(f"   ❌ face_encodings failed: {e}")
        
        # Test 2: Compare faces functionality
        print("2. Testing face comparison...")
        try:
            # Create two different encodings
            encoding1 = np.random.rand(128)
            encoding2 = np.random.rand(128)
            
            # Test compare_faces
            matches = face_recognition.compare_faces([encoding1], encoding2)
            print(f"   ✅ compare_faces: Works (result: {matches[0]})")
            
            # Test face_distance
            distances = face_recognition.face_distance([encoding1], encoding2)
            print(f"   ✅ face_distance: Works (distance: {distances[0]:.3f})")
            
        except Exception as e:
            print(f"   ❌ Face comparison failed: {e}")
            
        return True
        
    except ImportError:
        print("❌ face_recognition not installed")
        return False
    except Exception as e:
        print(f"❌ Face recognition functionality test failed: {e}")
        return False

def check_camera():
    """Test if camera is accessible"""
    print("\n--- Testing Camera ---")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera not accessible")
        return False
    
    ret, frame = cap.read()
    if ret:
        print(f"✅ Camera working - Frame shape: {frame.shape}")
        cap.release()
        return True
    else:
        print("❌ Camera failed to capture frame")
        cap.release()
        return False

if __name__ == "__main__":
    diagnose_environment()
    diagnose_image_loading()
    create_test_image()
    test_face_recognition_functionality()
    check_camera()
    
    print("\n=== DIAGNOSTIC COMPLETE ===")