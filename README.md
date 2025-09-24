# Smile Duration Tracking for Customer Service Kiosks  
**CMPT 310 - Summer 2025, Group 38**  
*Jake Sacilotto, Eito Nishikawa, Seungyeop Shin*  

---

## Overview  
This application detects customer smiles in real time and calculates an **engagement score** based on smile duration relative to face presence.  

- Built with a custom-trained **Convolutional Neural Network (CNN)** using TensorFlow  
- Uses **OpenCV** for image processing  
- Includes a **Tkinter-based GUI** for real-time interaction  

---

## Features  
- Real-time **face detection** and **smile classification**  
- Engagement score calculated based on **smile duration vs. face presence**  
- **Tkinter-based GUI** with webcam feed and live statistics  
- Automatic **pop-up summary window** after closing the app
  
<p align="center">
<img width="325" height="289" alt="Image" src="https://github.com/user-attachments/assets/082ff926-d4bc-4b2d-a726-36d8a652897f" />
</p>

<p align="center">
<img width="813" height="337" alt="Image" src="https://github.com/user-attachments/assets/29fac362-de4a-4146-9824-6b1b5a1feffc" />
</p>
---

## How to Run  

### 1. Install Python  
Make sure you have **Python 3.8+** installed.  

### 2. Install Dependencies  
Install the required packages with:  

```bash
pip install -r requirements.txt
```


### 3. Place Model File

Place your trained model file (model.keras) in the same folder as smile_tracker_gui.py.

### 4. Run the GUI

Start the application with:

```bash
python smile_tracker_gui.py
```
