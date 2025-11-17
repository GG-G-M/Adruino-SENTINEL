# check_installation.py
import sys
import pkg_resources

def check_installation():
    print("=== INSTALLATION CHECK ===")
    
    packages = [
        'dlib',
        'face_recognition', 
        'face_recognition_models',
        'opencv-python',
        'numpy',
        'Pillow'
    ]
    
    for package in packages:
        try:
            dist = pkg_resources.get_distribution(package)
            print(f"✅ {package}: {dist.version}")
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package}: NOT INSTALLED")
        except Exception as e:
            print(f"⚠️  {package}: Error - {e}")

if __name__ == "__main__":
    check_installation()