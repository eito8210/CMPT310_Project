Smile Duration Tracking for Customer Service Kiosks
CMPT 310 - Summer 2025, Group 38
Jake Sacilotto, Eito Nishikawa, Seungyeop Shin

Overview:
-----------
This application detects customer smiles in real time and calculates an engagement score based on smile duration relative to face presence. It uses a custom-trained Convolutional Neural Network (CNN) with TensorFlow and OpenCV for image processing, and a Python GUI built with Tkinter.

How to Run:
-----------
1. Make sure you have Python 3.8+ installed.
2. Install required packages using the command:

-----------------------------------------------
   pip install -r requirements.txt
------------------------------------------------

3. Place your trained model file (`model.keras`) in the same folder as `smile_tracker_gui.py`.
4. Run the GUI:

---------------------------------------------
   python smile_tracker_gui.py
---------------------------------------------

Features:
-----------
- Real-time face detection and smile classification
- Engagement score calculated based on smile duration
- Tkinter-based GUI with webcam feed and live statistics
- Pop-up summary window after closing

Optional:
-----------
- Run face_detection.py to see the work without the GUI

Notes:
-----------
- All files needed to run are included in this ZIP folder.
