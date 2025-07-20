from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import time
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DebugSession:
    def __init__(self):
        self.start_time = time.time()
        self.is_active = False
        self.frame_count = 0
    
    def reset(self):
        self.start_time = time.time()
        self.frame_count = 0
    
    def get_stats(self):
        total_time = time.time() - self.start_time if self.is_active else 0
        # デバッグ用の固定値
        face_time = total_time * 0.8
        smile_time = total_time * 0.4
        engagement = (smile_time / face_time * 100) if face_time > 0 else 0
        
        return {
            "totalTime": total_time,
            "faceTime": face_time,
            "smileTime": smile_time,
            "engagement": engagement
        }

@app.websocket("/ws/smile-detection")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = DebugSession()
    print("🔗 WebSocket client connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            print(f"📨 Received: {message['type']}")
            
            if message["type"] == "start":
                print("🚀 Starting detection session")
                session.reset()
                session.is_active = True
                await websocket.send_text(json.dumps({"type": "started"}))
                
            elif message["type"] == "stop":
                print("⏹️ Stopping detection session")
                session.is_active = False
                final_stats = session.get_stats()
                await websocket.send_text(json.dumps({
                    "type": "stopped",
                    "finalStats": final_stats
                }))
                
            elif message["type"] == "frame" and session.is_active:
                session.frame_count += 1
                print(f"🖼️ Processing frame #{session.frame_count}")
                
                # デバッグ用の模擬検出結果
                mock_detections = [{
                    "x": 100,
                    "y": 100,
                    "width": 200,
                    "height": 200,
                    "confidence": 0.85,
                    "isSmiling": session.frame_count % 3 == 0,  # 3フレームに1回笑顔
                    "label": f"Face #{session.frame_count} - {'Smiling' if session.frame_count % 3 == 0 else 'Not Smiling'}"
                }]
                
                result = {
                    "detections": mock_detections,
                    "stats": session.get_stats()
                }
                
                await websocket.send_text(json.dumps({
                    "type": "detection",
                    "data": result
                }))
                print(f"✅ Sent detection result for frame #{session.frame_count}")
                
    except WebSocketDisconnect:
        print("❌ Client disconnected")
    except Exception as e:
        print(f"💥 WebSocket error: {e}")

@app.get("/health")
async def health_check():
    print("🏥 Health check requested")
    return {
        "status": "healthy",
        "models_loaded": True,
        "message": "Debug FastAPI server is running"
    }

if __name__ == "__main__":
    import uvicorn
    print("🐍 Starting debug FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
