import os
import sys
import subprocess
import venv

def setup_environment():
    """Setup complete Python environment with C++ dependencies"""
    
    print("="*60)
    print("SETTING UP SECURITY SYSTEM ENVIRONMENT")
    print("="*60)
    
    # 1. Create virtual environment
    print("\n[1/3] Creating virtual environment...")
    venv_path = "venv"
    if not os.path.exists(venv_path):
        venv.create(venv_path, with_pip=True)
        print(f"✅ Virtual environment created at: {venv_path}")
    else:
        print("✅ Virtual environment already exists")
    
    # Determine Python path
    if sys.platform == "win32":
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
        pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        python_path = os.path.join(venv_path, "bin", "python")
        pip_path = os.path.join(venv_path, "bin", "pip")
    
    # 2. Install requirements
    print("\n[2/3] Installing dependencies...")
    requirements = [
        "opencv-python==4.8.1.78",
        "numpy==1.24.3",
        "Pillow==10.0.1",
        "mediapipe==0.10.8",
        "pyserial==3.5",
        "face-recognition==1.3.0",
        "face-recognition-models==0.3.0",
        "scikit-learn==1.3.0",
        "scipy==1.11.3",
        "pickle-mixin==1.0.2"
    ]
    
    for package in requirements:
        print(f"Installing {package}...")
        subprocess.run([pip_path, "install", package], check=True)
    
    # 3. Test installation
    print("\n[3/3] Testing installation...")
    test_code = """
import cv2
print(f"✅ OpenCV version: {cv2.__version__}")
import numpy as np
print(f"✅ NumPy version: {np.__version__}")
import mediapipe as mp
print("✅ MediaPipe imported successfully")
try:
    import face_recognition
    print("✅ Face Recognition imported successfully")
except Exception as e:
    print(f"⚠️  Face Recognition warning: {e}")
"""
    
    subprocess.run([python_path, "-c", test_code])
    
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("To activate virtual environment:")
    if sys.platform == "win32":
        print(f"  {venv_path}\\Scripts\\activate")
    else:
        print(f"  source {venv_path}/bin/activate")
    print("Then run: python main.py")
    print("="*60)

if __name__ == "__main__":
    setup_environment()