import os
import sys
import subprocess
import venv

def setup_environment():
    """Setup complete Python environment with C++ dependencies"""
    
    print("="*60)
    print("SETTING UP SECURITY SYSTEM ENVIRONMENT")
    print("Auto-Sampling Face Recognition System")
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
    
    # 2. Upgrade pip
    print("\n[2/3] Upgrading pip...")
    subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # 3. Install requirements
    print("\n[3/3] Installing dependencies...")
    print("This may take 5-10 minutes...")
    
    requirements = [
        "opencv-python==4.8.1.78",
        "numpy==1.24.3",
        "Pillow==10.0.1",
        "mediapipe==0.10.8",
        "pyserial==3.5",
        "scikit-learn==1.3.0",
        "scipy==1.11.3",
        "pickle-mixin==1.0.2",
        # Diagnostic dependencies
        "psutil==5.9.6",        # For hardware monitoring in diagnostic
        "gputil==1.4.0",        # For GPU monitoring in diagnostic
    ]
    
    # Install core packages first
    print("\n📦 Installing core packages...")
    for package in requirements:
        print(f"   Installing {package}...")
        try:
            subprocess.run([pip_path, "install", package], check=True, 
                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ Warning: {package} - {e}")
    
    # Install face-recognition (requires C++ build tools)
    print("\n📦 Installing face-recognition (requires C++ Build Tools)...")
    print("   This may take 5-10 minutes on first install...")
    try:
        subprocess.run([pip_path, "install", "face-recognition==1.3.0"], 
                     check=True, timeout=600)
        print("   ✅ face-recognition installed")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed to install face-recognition")
        print(f"   Error: {e}")
        print("\n⚠️ IMPORTANT: Install Visual Studio Build Tools:")
        print("   https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        print("   Select: 'Desktop development with C++'")
        return False
    except subprocess.TimeoutExpired:
        print("   ⚠️ Installation timeout - this is normal for face-recognition")
        print("   Let it finish in the background...")
    
    # 4. Test installation
    print("\n" + "="*60)
    print("Testing installation...")
    print("="*60)
    
    test_code = """
import sys
success = True

try:
    import cv2
    print(f"✅ OpenCV version: {cv2.__version__}")
except Exception as e:
    print(f"❌ OpenCV: {e}")
    success = False

try:
    import numpy as np
    print(f"✅ NumPy version: {np.__version__}")
except Exception as e:
    print(f"❌ NumPy: {e}")
    success = False

try:
    import mediapipe as mp
    print("✅ MediaPipe imported successfully")
except Exception as e:
    print(f"❌ MediaPipe: {e}")
    success = False

try:
    import face_recognition
    print("✅ face_recognition imported successfully")
except Exception as e:
    print(f"⚠️ face_recognition: {e}")
    print("   Run this command manually:")
    print("   pip install face-recognition")
    success = False

# Test diagnostic dependencies
try:
    import psutil
    print(f"✅ psutil imported successfully (for diagnostics)")
except Exception as e:
    print(f"⚠️ psutil: {e} (optional for diagnostics)")

try:
    import GPUtil
    print("✅ GPUtil imported successfully (for diagnostics)")
except Exception as e:
    print(f"⚠️ GPUtil: {e} (optional for diagnostics)")

if success:
    print("\\n✅ All dependencies installed correctly!")
else:
    print("\\n⚠️ Some dependencies failed to install")

sys.exit(0 if success else 1)
"""
    
    result = subprocess.run([python_path, "-c", test_code])
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE!")
        print("="*60)
        print("To use the system:")
        print("1. Activate virtual environment:")
        if sys.platform == "win32":
            print(f"   {venv_path}\\Scripts\\activate")
        else:
            print(f"   source {venv_path}/bin/activate")
        print("\n2. Run diagnostic to check system compatibility:")
        print("   python diagnostic.py")
        print("\n3. Run the main security system:")
        print("   python main.py")
        print("\n4. Files available:")
        print("   • diagnostic.py - Comprehensive system diagnostic")
        print("   • main.py       - Security system main program")
        print("   • setup.py      - This setup script")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("⚠️ SETUP INCOMPLETE")
        print("="*60)
        print("Please fix the errors above and run setup.py again")
        return False

if __name__ == "__main__":
    success = setup_environment()
    sys.exit(0 if success else 1)