# Adruino-SENTINEL
Run this code:

# 1. Install precompiled dlib wheel
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl

# 2. Install face recognition models
pip install face_recognition_models-0.3.0.tar.gz

# 2. Then install the requirements
pip install -r requirements.txt

# Error Install:
# Go inside Venv
.\.venv\Scripts\activate
# ModuleNotFoundError: No module named 'mediapipe': 
pip install mediapipe
# ModuleNotFoundError: No module named 'face_recognition'
pip install face_recognition
# if still error
python -m pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
then redo the pip face_recog or just use the "pip install -r requirements.txt"


Alternative and WORKING:
# Deactivate current environment
deactivate

# Remove corrupted environment
Remove-Item -Recurse -Force .venv

# Create new environment
python -m venv .venv

# Activate it
.\.venv\Scripts\activate

# First, ensure pip is working
python -m pip install --upgrade pip

# Now install the working versions
pip install dlib-19.22.99-cp310-cp310-win_amd64.whl
pip install opencv-python==4.5.5.64
pip install numpy==1.21.6
pip install pillow==9.5.0
pip install face_recognition_models-0.3.0.tar.gz
pip install face_recognition==1.3.0
pip install mediapipe