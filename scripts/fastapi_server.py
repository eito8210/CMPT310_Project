from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import asyncio
import json
import base64
import time
from typing import Dict, Any

app = FastAPI()

# より寛容なCORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発環境では全てのオリジンを許可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
face_cascade = None
IMG_SIZE = 64

# Initialize model and cascade
def initialize_models():
    global model, face_cascade
    try:
        model = load_model('model.keras')
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print("Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

# Initialize on startup
@app.on_event("startup")
async def startup_event():
    initialize_models()

class SmileDetectionSession:
    def __init__(self):
        self.face_detected_time = 0.0
        self.smile_detected_time = 0.0
        self.last_frame_time = time.time()
        self.start_time = time.time()
        self.is_active = False
    
    def reset(self):
        self.face_detected_time = 0.0
        self.smile_detected_time = 0.0
        self.last_frame_time = time.time()
        self.start_time = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        total_time = time.time() - self.start_time if self.is_active else 0
        engagement = (self.smile_detected_time / self.face_detected_time * 100) if self.face_detected_time > 0 else 0
        
        return {
            "totalTime": total_time,
            "faceTime": self.face_detected_time,
            "smileTime": self.smile_detected_time,
            "engagement": engagement
        }

def process_frame(frame_data: str, session: SmileDetectionSession) -> Dict[str, Any]:
    global model, face_cascade
    
    if not model or not face_cascade:
        return {"error": "Models not loaded"}
    
    try:
        # Decode base64 image
        img_data = base64.b64decode(frame_data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Invalid frame data"}
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Time calculation
        current_time = time.time()
        frame_time = current_time - session.last_frame_time
        session.last_frame_time = current_time
        
        detections = []
        
        for (x, y, w, h) in faces:
            session.face_detected_time += frame_time
            
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            try:
                # Preprocess for model
                face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                face_normalized = face_resized.astype("float32") / 255.0
                face_input = np.expand_dims(face_normalized, axis=(0, -1))
                
                # Predict smile
                prediction = model.predict(face_input, verbose=0)[0][0]
                
                is_smiling = prediction > 0.10
                if is_smiling:
                    session.smile_detected_time += frame_time
                
                detections.append({
                    "x": int(x),
                    "y": int(y),
                    "width": int(w),
                    "height": int(h),
                    "confidence": float(prediction),
                    "isSmiling": is_smiling,
                    "label": f"Smiling ({prediction:.2f})" if is_smiling else f"Not Smiling ({prediction:.2f})"
                })
                
            except Exception as e:
                print(f"Face processing error: {e}")
        
        return {
            "detections": detections,
            "stats": session.get_stats()
        }
        
    except Exception as e:
        return {"error": f"Frame processing error: {str(e)}"}

@app.websocket("/ws/smile-detection")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = SmileDetectionSession()
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "start":
                session.reset()
                session.is_active = True
                await websocket.send_text(json.dumps({"type": "started"}))
                
            elif message["type"] == "stop":
                session.is_active = False
                final_stats = session.get_stats()
                await websocket.send_text(json.dumps({
                    "type": "stopped",
                    "finalStats": final_stats
                }))
                
            elif message["type"] == "frame" and session.is_active:
                result = process_frame(message["data"], session)
                await websocket.send_text(json.dumps({
                    "type": "detection",
                    "data": result
                }))
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": model is not None and face_cascade is not None,
        "message": "FastAPI server is running with TensorFlow model"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
