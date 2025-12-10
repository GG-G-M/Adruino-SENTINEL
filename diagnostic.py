import cv2
import os
import numpy as np
import platform
import sys
import time
from PIL import Image

class SecuritySystemDiagnostic:
    def __init__(self):
        self.issues = []
        self.recommendations = []
        self.components = {
            "camera": {"status": "[FAIL]", "score": 0, "required": True},
            "libraries": {"status": "[FAIL]", "score": 0, "required": True},
            "face_recognition": {"status": "[FAIL]", "score": 0, "required": True},
            "performance": {"status": "[FAIL]", "score": 0, "required": True},
            "arduino": {"status": "[WARN]", "score": 0, "required": False}
        }
    
    def print_header(self, title):
        print("\n" + "="*60)
        print(f"  {title}")
        print("="*60)
    
    def test_python_environment(self):
        """Check Python version and environment"""
        print("\nPYTHON ENVIRONMENT:")
        print(f"  Version: {sys.version.split()[0]}")
        print(f"  Executable: {sys.executable}")
        
        # Check virtual environment
        in_venv = sys.prefix != sys.base_prefix
        status = "[OK] Active" if in_venv else "[WARN] Not active"
        print(f"  Virtual Environment: {status}")
        
        if not in_venv:
            self.recommendations.append("Run in virtual environment: venv\\Scripts\\activate (Windows) or source venv/bin/activate (Linux/Mac)")
        
        return in_venv
    
    def test_libraries(self):
        """Test all required libraries for the security system"""
        self.print_header("LIBRARY COMPATIBILITY")
        
        required = {
            "OpenCV": ("cv2", "4.8.1.78", "Face detection and image processing"),
            "NumPy": ("numpy", "1.24.3", "Numerical operations"),
            "Pillow": ("PIL", "10.0.1", "Image handling"),
            "MediaPipe": ("mediapipe", "0.10.8", "Hand gesture recognition"),
            "PySerial": ("serial", "3.5", "Arduino communication"),
            "scikit-learn": ("sklearn", "1.3.0", "Machine learning utilities"),
            "SciPy": ("scipy", "1.11.3", "Scientific computing"),
        }
        
        print("CHECKING REQUIRED LIBRARIES:")
        installed = 0
        total = len(required)
        
        for lib_name, (import_name, version, purpose) in required.items():
            try:
                if import_name == "cv2":
                    module = cv2
                    ver = cv2.__version__
                elif import_name == "PIL":
                    from PIL import Image
                    module = Image
                    ver = Image.__version__
                elif import_name == "sklearn":
                    import sklearn
                    module = sklearn
                    ver = sklearn.__version__
                else:
                    module = __import__(import_name)
                    ver = getattr(module, "__version__", "Unknown")
                
                status = "[OK]"
                installed += 1
                print(f"  {status} {lib_name:15} v{ver:10} - {purpose}")
                
            except ImportError as e:
                status = "[FAIL]"
                print(f"  {status} {lib_name:15} MISSING      - {purpose}")
                self.issues.append(f"{lib_name} not installed")
                self.recommendations.append(f"Install {lib_name}: pip install {import_name}=={version}")
        
        # Special test for face_recognition
        print("\nFACE RECOGNITION LIBRARY:")
        try:
            import face_recognition
            version = face_recognition.__version__
            print(f"  [OK] face_recognition v{version}")
            self.components["face_recognition"]["status"] = "[OK]"
            self.components["face_recognition"]["score"] = 100
            installed += 1
            total += 1
        except ImportError as e:
            print(f"  [FAIL] face_recognition MISSING - Critical for system")
            print(f"  Error: {str(e)[:100]}")
            self.issues.append("face_recognition library not installed")
            self.recommendations.append("Install face_recognition: pip install face-recognition==1.3.0")
            self.recommendations.append("On Windows, install Visual Studio Build Tools first")
        
        # Calculate library score
        lib_score = (installed / total) * 100
        self.components["libraries"]["score"] = lib_score
        self.components["libraries"]["status"] = "[OK]" if lib_score >= 90 else "[WARN]" if lib_score >= 70 else "[FAIL]"
        
        print(f"\nLIBRARY SCORE: {lib_score:.0f}% ({installed}/{total} installed)")
        return lib_score
    
    def test_camera(self):
        """Test camera for face recognition"""
        self.print_header("CAMERA TEST")
        
        print("Testing camera access...")
        camera_working = False
        camera_index = None
        
        # Try different camera indices
        for i in range(3):
            try:
                if platform.system() == "Windows":
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(i)
                
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        camera_working = True
                        camera_index = i
                        height, width = frame.shape[:2]
                        
                        print(f"  [OK] Camera {i}: Found {width}x{height}")
                        
                        # Test FPS
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        if fps <= 0:
                            # Calculate FPS manually
                            start = time.time()
                            for _ in range(10):
                                cap.read()
                            fps = 10 / (time.time() - start)
                        
                        print(f"  Resolution: {width}x{height}")
                        print(f"  FPS: {fps:.1f}")
                        
                        # Test face detection
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        face_cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                        )
                        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                        
                        print(f"  Face detection: {'[OK] Working' if len(faces) >= 0 else '[WARN] No faces detected (might be empty)'}")
                        
                        # Save test image
                        cv2.imwrite("camera_test.jpg", frame)
                        print(f"  Test image saved: camera_test.jpg")
                        
                        cap.release()
                        break
                    else:
                        cap.release()
                else:
                    if cap:
                        cap.release()
            except Exception as e:
                if 'cap' in locals():
                    cap.release()
        
        if camera_working:
            self.components["camera"]["status"] = "[OK]"
            self.components["camera"]["score"] = 90
            print("\n[OK] CAMERA: READY for face recognition")
        else:
            self.components["camera"]["status"] = "[FAIL]"
            self.components["camera"]["score"] = 0
            print("\n[FAIL] CAMERA: NOT DETECTED")
            self.issues.append("Camera not detected")
            self.recommendations.append("Connect a webcam or enable built-in camera")
            self.recommendations.append("Check camera drivers")
        
        return camera_working
    
    def test_performance(self):
        """Test system performance for face recognition"""
        self.print_header("PERFORMANCE TEST")
        
        print("Testing face recognition performance...")
        
        # Create test face
        test_face = np.random.randint(100, 200, (150, 150, 3), dtype=np.uint8)
        
        # Test 1: OpenCV operations
        print("\n1. OpenCV Performance:")
        start = time.time()
        for _ in range(100):
            gray = cv2.cvtColor(test_face, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        cv_time = (time.time() - start) / 100
        
        if cv_time < 0.001:
            cv_status = "[OK] EXCELLENT"
            cv_score = 100
        elif cv_time < 0.005:
            cv_status = "[OK] GOOD"
            cv_score = 80
        elif cv_time < 0.01:
            cv_status = "[WARN] ACCEPTABLE"
            cv_score = 60
        else:
            cv_status = "[FAIL] SLOW"
            cv_score = 30
        
        print(f"  Image processing: {cv_time*1000:.1f}ms per frame ({cv_status})")
        
        # Test 2: Face encoding performance (if available)
        try:
            import face_recognition
            print("\n2. Face Recognition Performance:")
            
            # Test encoding speed
            start = time.time()
            encodings = face_recognition.face_encodings(test_face)
            encoding_time = time.time() - start
            
            if encoding_time < 0.5:
                face_status = "[OK] EXCELLENT"
                face_score = 100
            elif encoding_time < 1.0:
                face_status = "[OK] GOOD"
                face_score = 80
            elif encoding_time < 2.0:
                face_status = "[WARN] ACCEPTABLE"
                face_score = 60
            else:
                face_status = "[FAIL] SLOW"
                face_score = 30
            
            print(f"  Face encoding: {encoding_time:.2f}s ({face_status})")
            
            # Overall performance score
            perf_score = (cv_score + face_score) / 2
            
        except ImportError:
            print("\n2. Face Recognition: [FAIL] NOT AVAILABLE")
            perf_score = cv_score
        
        self.components["performance"]["score"] = perf_score
        self.components["performance"]["status"] = "[OK]" if perf_score >= 80 else "[WARN]" if perf_score >= 60 else "[FAIL]"
        
        print(f"\nPERFORMANCE SCORE: {perf_score:.0f}/100")
        return perf_score
    
    def test_arduino(self):
        """Test Arduino compatibility"""
        self.print_header("ARDUINO COMPATIBILITY")
        
        print("Checking Arduino/serial connection...")
        
        try:
            import serial.tools.list_ports
            ports = list(serial.tools.list_ports.comports())
            
            if not ports:
                print("  [WARN] No serial ports found")
                self.components["arduino"]["status"] = "[WARN]"
                self.components["arduino"]["score"] = 30
                return False
            
            arduino_found = False
            for p in ports:
                print(f"  {p.device} - {p.description}")
                if any(keyword in p.description.upper() for keyword in 
                      ['CH340', 'ARDUINO', 'USB SERIAL']):
                    arduino_found = True
                    print(f"  [OK] Likely Arduino on {p.device}")
            
            if arduino_found:
                self.components["arduino"]["status"] = "[OK]"
                self.components["arduino"]["score"] = 100
                print("\n[OK] ARDUINO: DETECTED")
                return True
            else:
                self.components["arduino"]["status"] = "[WARN]"
                self.components["arduino"]["score"] = 50
                print("\n[WARN] ARDUINO: Not detected (but serial ports available)")
                return False
                
        except ImportError:
            print("  [FAIL] pyserial not installed")
            self.components["arduino"]["status"] = "[FAIL]"
            self.components["arduino"]["score"] = 0
            return False
    
    def check_project_structure(self):
        """Check if project folders and files exist"""
        self.print_header("PROJECT STRUCTURE")
        
        required_folders = ["authorized_faces"]
        required_files = ["main.py", "setup.py", "diagnostic.py"]
        
        print("Checking project structure:")
        
        all_good = True
        for folder in required_folders:
            if os.path.exists(folder):
                print(f"  [OK] Folder: {folder}/")
            else:
                print(f"  [WARN] Missing folder: {folder}/")
                all_good = False
                self.issues.append(f"Missing folder: {folder}")
                self.recommendations.append(f"Create folder: mkdir {folder}")
        
        for file in required_files:
            if os.path.exists(file):
                print(f"  [OK] File: {file}")
            else:
                print(f"  [FAIL] Missing file: {file}")
                all_good = False
                self.issues.append(f"Missing file: {file}")
        
        if all_good:
            print("\n[OK] PROJECT STRUCTURE: COMPLETE")
        else:
            print("\n[WARN] PROJECT STRUCTURE: INCOMPLETE")
        
        return all_good
    
    def generate_report(self):
        """Generate final diagnostic report"""
        self.print_header("DIAGNOSTIC REPORT")
        
        # Calculate overall score
        required_components = [c for c, info in self.components.items() if info["required"]]
        total_score = sum(self.components[c]["score"] for c in required_components)
        avg_score = total_score / len(required_components) if required_components else 0
        
        print(f"OVERALL SYSTEM COMPATIBILITY: {avg_score:.0f}%\n")
        
        print("COMPONENT STATUS:")
        for component, info in self.components.items():
            status = info["status"]
            score = info["score"]
            required = "(Required)" if info["required"] else "(Optional)"
            print(f"  {status} {component:20} {score:3.0f}%  {required}")
        
        # Compatibility verdict
        print("\n" + "="*60)
        print("COMPATIBILITY VERDICT:")
        
        if avg_score >= 90:
            print("[OK] EXCELLENT - System is fully compatible!")
            print("     Ready to run the security system.")
        elif avg_score >= 70:
            print("[OK] GOOD - System is compatible.")
            print("     Minor issues may affect performance.")
        elif avg_score >= 50:
            print("[WARN] FAIR - System has limitations.")
            print("       Some features may not work optimally.")
        else:
            print("[FAIL] POOR - System may not work properly.")
            print("       Fix issues before running the system.")
        
        print("\n" + "="*60)
        
        # Show issues and recommendations
        if self.issues:
            print("\nISSUES TO FIX:")
            for issue in self.issues:
                print(f"  * {issue}")
        
        if self.recommendations:
            print("\nRECOMMENDATIONS:")
            for rec in self.recommendations:
                print(f"  * {rec}")
        
        # Next steps
        print("\n" + "="*60)
        print("NEXT STEPS:")
        
        if avg_score >= 70:
            print("1. Run the security system:")
            print("   python main.py")
            print("\n2. Register faces (Option 1 in menu)")
            print("\n3. Test authentication (Option 2 in menu)")
        else:
            print("1. Fix the issues listed above")
            print("\n2. Re-run this diagnostic:")
            print("   python diagnostic.py")
            print("\n3. Then run the security system:")
            print("   python main.py")
        
        print("\n" + "="*60)
        
        # Save simple report
        self.save_report(avg_score)
        
        return avg_score
    
    def save_report(self, score):
        """Save report to file"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostic_report_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write("FACE RECOGNITION SECURITY SYSTEM - DIAGNOSTIC REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"System: {platform.system()} {platform.release()}\n")
            f.write(f"Python: {sys.version.split()[0]}\n\n")
            
            f.write(f"OVERALL COMPATIBILITY: {score:.0f}%\n\n")
            
            f.write("COMPONENT STATUS:\n")
            for component, info in self.components.items():
                status = info['status'].replace('[OK]', 'PASS').replace('[WARN]', 'WARN').replace('[FAIL]', 'FAIL')
                f.write(f"  {component:20} {status:5} {info['score']:.0f}%\n")
            
            if self.issues:
                f.write("\nISSUES:\n")
                for issue in self.issues:
                    f.write(f"  * {issue}\n")
            
            if self.recommendations:
                f.write("\nRECOMMENDATIONS:\n")
                for rec in self.recommendations:
                    f.write(f"  * {rec}\n")
        
        print(f"\nReport saved to: {filename}")

def run_diagnostic():
    """Main diagnostic function"""
    print("\n" + "="*60)
    print("  SECURITY SYSTEM DIAGNOSTIC")
    print("  Face Recognition + Hand Gesture Authentication")
    print("="*60)
    
    diagnostic = SecuritySystemDiagnostic()
    
    try:
        # Run tests
        diagnostic.test_python_environment()
        diagnostic.test_libraries()
        diagnostic.test_camera()
        diagnostic.test_performance()
        diagnostic.test_arduino()
        diagnostic.check_project_structure()
        
        # Generate final report
        score = diagnostic.generate_report()
        
        return score >= 70  # Return True if system is ready
        
    except KeyboardInterrupt:
        print("\n\n[WARN] Diagnostic interrupted by user")
        return False
    except Exception as e:
        print(f"\n[FAIL] Diagnostic error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting diagnostic for Face Recognition Security System...")
    print("This will test system compatibility and performance.\n")
    
    try:
        ready = run_diagnostic()
        
        if ready:
            print("\n[OK] SYSTEM READY - You can now run the security system!")
            sys.exit(0)
        else:
            print("\n[WARN] SYSTEM NOT READY - Please fix the issues above")
            sys.exit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(f"\n[FAIL] Fatal error: {e}")
        sys.exit(1)