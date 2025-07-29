from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time
import threading
import asyncio
from typing import Dict, Any
import os

# 環境変数からポートを取得（デフォルト: 5001）
API_PORT = int(os.getenv("API_PORT", 5001))
API_HOST = os.getenv("API_HOST", "0.0.0.0")

app = FastAPI(title="Smile Detection API", version="1.0.0")

# CORS設定 - Next.jsのポートに対応
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js開発サーバー
        "http://127.0.0.1:3000",
        "http://localhost:3001",  # 代替ポート
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydanticモデル
class InitResponse(BaseModel):
    success: bool
    message: str = ""

class DetectionResponse(BaseModel):
    faceDetected: bool
    smiling: bool
    confidence: float

class StatsResponse(BaseModel):
    totalTime: float
    faceTime: float
    smileTime: float
    engagement: float

class StatusResponse(BaseModel):
    modelsLoaded: bool
    detecting: bool

# グローバル変数
model = None
face_cascade = None
detection_active = False
detection_thread = None
current_stats = {
    'totalTime': 0.0,
    'faceTime': 0.0,
    'smileTime': 0.0,
    'engagement': 0.0
}
current_detection = {
    'faceDetected': False,
    'smiling': False,
    'confidence': 0.0
}

# 検出変数
face_detected_time = 0.0
smile_detected_time = 0.0
start_time = 0.0
last_frame_time = 0.0
IMG_SIZE = 64

def initialize_models():
    """モデルを初期化する関数"""
    global model, face_cascade
    try:
        # 訓練済みモデル "model.keras" を読み込み
        model = load_model('model.keras')
        # HAAR cascadesを使用した顔検出器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return True
    except Exception as e:
        print(f"モデル読み込みエラー: {e}")
        return False

def detection_loop():
    """検出ループ関数（あなたのオリジナルコードをベースに）"""
    global detection_active, face_detected_time, smile_detected_time, start_time, last_frame_time
    global current_stats, current_detection
    
    # ウェブカメラを開く
    cap = cv2.VideoCapture(0)
    
    # カウンターをリセット
    face_detected_time = 0.0
    smile_detected_time = 0.0
    start_time = time.time()
    last_frame_time = time.time()
    
    while detection_active:
        # ウェブカメラからフレームを取得
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # グレースケールに変換（モデルが白黒で訓練されているため）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 顔を検出
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # フレーム時間を計算
        current_time = time.time()
        frame_time = current_time - last_frame_time
        last_frame_time = current_time
        
        face_detected = len(faces) > 0
        smiling = False
        confidence = 0.0
        
        if face_detected:
            face_detected_time += frame_time
            
            for (x, y, w, h) in faces:
                # 顔領域を抽出
                face_roi = gray[y:y+h, x:x+w]
                try:
                    # モデルが期待するサイズにリサイズ
                    face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                    # 値を正規化して(0,1)の範囲にする
                    face_normalized = face_resized.astype("float32") / 255.0
                    face_input = np.expand_dims(face_normalized, axis=(0, -1))
                    
                    # 笑顔の確率を取得
                    prediction = model.predict(face_input, verbose=0)[0][0]
                    confidence = float(prediction)
                    
                    if prediction > 0.10:  # 笑顔の閾値
                        smile_detected_time += frame_time
                        smiling = True
                        
                except Exception as e:
                    print("顔の前処理に失敗:", e)
                    
                break  # 最初の顔のみ処理
        
        # 現在の検出結果を更新
        current_detection = {
            'faceDetected': face_detected,
            'smiling': smiling,
            'confidence': confidence
        }
        
        # 統計を更新
        total_time = current_time - start_time
        engagement = (smile_detected_time / face_detected_time * 100) if face_detected_time > 0 else 0
        
        current_stats = {
            'totalTime': total_time,
            'faceTime': face_detected_time,
            'smileTime': smile_detected_time,
            'engagement': engagement
        }
        
        # CPU使用量を抑えるための小さな遅延
        time.sleep(0.1)
    
    cap.release()

@app.post("/api/initialize", response_model=InitResponse)
async def initialize():
    """モデルを初期化"""
    success = initialize_models()
    if success:
        return InitResponse(success=True, message="モデルが正常に初期化されました")
    else:
        raise HTTPException(status_code=500, detail="モデルの初期化に失敗しました")

@app.post("/api/start-detection", response_model=Dict[str, bool])
async def start_detection():
    """検出を開始"""
    global detection_active, detection_thread
    
    if model is None or face_cascade is None:
        raise HTTPException(status_code=400, detail="モデルが初期化されていません")
    
    if not detection_active:
        detection_active = True
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.start()
        return {"success": True}
    
    raise HTTPException(status_code=400, detail="検出は既にアクティブです")

@app.post("/api/stop-detection", response_model=Dict[str, bool])
async def stop_detection():
    """検出を停止"""
    global detection_active, detection_thread
    
    detection_active = False
    if detection_thread:
        detection_thread.join()
    
    return {"success": True}

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """統計を取得"""
    return StatsResponse(**current_stats)

@app.get("/api/detection", response_model=DetectionResponse)
async def get_detection():
    """現在の検出結果を取得"""
    return DetectionResponse(**current_detection)

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """ステータスを取得"""
    return StatusResponse(
        modelsLoaded=model is not None and face_cascade is not None,
        detecting=detection_active
    )

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "Smile Detection API", "status": "running", "port": API_PORT}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Smile Detection API を起動中...")
    print(f"📱 API: http://localhost:{API_PORT}")
    print(f"📖 Docs: http://localhost:{API_PORT}/docs")
    print("⏹️  停止するには Ctrl+C を押してください")
    print("-" * 50)
    
    uvicorn.run(
        "face_detection_api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info"
    )
