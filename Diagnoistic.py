import cv2
import face_recognition
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
    print(f"face_recognition version: {face_recognition.__version__}")
    print()

def diagnose_image_loading():
    REGISTERED_FOLDER = "authorized_faces"
    
    if not os.path.exists(REGISTERED_FOLDER):
        print(f"Folder '{REGISTERED_FOLDER}' does not exist")
        return
    
    print("Testing image loading methods...")
    
    for filename in os.listdir(REGISTERED_FOLDER):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
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
            
            # Method 3: face_recognition's loader
            try:
                img_fr = face_recognition.load_image_file(path)
                print(f"✅ face_recognition: Loaded - Shape: {img_fr.shape}, dtype: {img_fr.dtype}")
                
                # Test face encoding
                encodings = face_recognition.face_encodings(img_fr)
                print(f"   Face encodings found: {len(encodings)}")
                
            except Exception as e:
                print(f"❌ face_recognition Error: {e}")

def create_test_image():
    """Create a simple test image to verify the pipeline"""
    print("\n--- Creating test image ---")
    # Create a simple RGB image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_path = "test_image.jpg"
    
    # Save with PIL
    pil_img = Image.fromarray(test_image)
    pil_img.save(test_path, 'JPEG')
    print(f"✅ Test image saved as {test_path}")
    
    # Try to load with face_recognition
    try:
        loaded_img = face_recognition.load_image_file(test_path)
        print(f"✅ Test image loaded successfully - Shape: {loaded_img.shape}, dtype: {loaded_img.dtype}")
        
        # Clean up
        os.remove(test_path)
        return True
    except Exception as e:
        print(f"❌ Failed to load test image: {e}")
        return False

if __name__ == "__main__":
    diagnose_environment()
    diagnose_image_loading()
    create_test_image()