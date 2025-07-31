import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time

# ===== 元のface_detection.pyのコード部分 =====
# Loading our trained model "model.keras"
model = load_model('model.keras')

# This is our face detector using HAAR cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# initliizing variables for counting time
face_detected_time = 0.0
smile_detected_time = 0.0

# I had a problem where I would try to add the seconds everytime the face showed uo, but it would overwrite it
# so now i get the total time and subtract between the last frame a face was encountered
# since we cant do += 0.5 or whatever cause we are not sure how many seconds per frame that it checks
last_frame_time = time.time()

start_of_code_time = time.time()

IMG_SIZE = 64

# ===== FastAPI用の追加部分 =====
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import threading
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

# API用の制御変数
detection_active = False
detection_thread = None
cap = None

# 現在の検出結果と統計（API用）
current_detection = {
    'faceDetected': False,
    'smiling': False,
    'confidence': 0.0
}

def detection_loop_api():
    """元のface_detection.pyのwhile Trueループを関数化したもの"""
    global detection_active, face_detected_time, smile_detected_time, last_frame_time, start_of_code_time
    global current_detection, cap
    
    # Open up the webcam
    cap = cv2.VideoCapture(0)
    
    while detection_active:
        #This gets every frame from the webcam
        ret, frame = cap.read()

        if not ret:
            continue

        #here we convert to gray colour since our model was trained in black and white
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #This is where we attempt to detect the face
        # the first number (the lower the number the more accuracte but slower)
        # The first number is called the scale factor
        #second number is called min neuighbours,the higher the numebr aka 7 it might not detect face
        # if its at like 3 it might give like random boxes (its more sensitive)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        
        # JAKES FACE SCORE CODE: this is where I am setting up how I will subtract the time where no face is detected
              #time.time() will not add any time, it just retunrs the time it is during the day
        current_time= time.time()
        frame_time = (current_time) - (last_frame_time)

        # this will update the frame to the last time we saw a face and we use it in computation above
        last_frame_time = current_time

        # API用の検出結果初期化
        face_detected = len(faces) > 0
        smiling = False
        confidence = 0.0

        for (x, y, w, h) in faces:
            #This is our visual rectangle on the face (currently set to blue (255, 0, 0))
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)  # API版では画面表示しない

            #counter for face on screen
            face_detected_time += frame_time

            # roi = region of interest
            face_roi = gray[y:y+h, x:x+w]
            try:

                # this is where we have to resize our ROI to the size that the model expects
                face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                # normalize our values so that we can get values from (0,1)
                face_nomalized = face_resized.astype("float32") / 255.0
                face_input = np.expand_dims(face_nomalized, axis = (0, -1))

                #this is where we get our smile probability and use it when outputting smiling/not smiling
                prediction = model.predict(face_input, verbose=0)[0][0]
                confidence = float(prediction)

                # open cv uses BlueGreenRed instead of RGB for some reason
                if prediction > 0.10:     #THIS IS OUR SMILING THRESHOLD, CAN PUT IT UP OR DOWN BASED ON RESULTS
                    #add smile time
                    smile_detected_time += frame_time
                    smiling = True
                    label = f"Smiling ({prediction:.2f})" 
                    color = (0, 255, 0) #green
                else:
                    label = f"Not Smiling ({prediction:.2f})"
                    color = (0, 0, 255) #red

                # API版では画面表示をしないのでコメントアウト
                # cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            #if anything goes bad it fails
            except Exception as e:
                print("Face preprocessing failed:", e)

        # API用の現在の検出結果を更新
        current_detection = {
            'faceDetected': face_detected,
            'smiling': smiling,
            'confidence': confidence
        }

        # API版では画面表示をしないのでコメントアウト
        # cv2.imshow('Smile Detection', frame)

        # API版では'q'キーチェックをしないのでコメントアウト
        # #this will exit our while loop if the user presses q
        # if cv2.waitKey(1) == ord('q'):
        #     break

        # CPU使用量を抑えるための小さな遅延
        time.sleep(0.1)

    #this closes the webcam and the open cv stuff
    if cap:
        cap.release()
    # cv2.destroyAllWindows()  # API版では不要

def calculate_final_stats():
    """元のface_detection.pyの最終統計計算部分"""
    end_of_code_time = time.time()
    total_runtime = (end_of_code_time) - (start_of_code_time)

    # python they always print on seperate lines so no need for \n like in C++
    print(f"------------------------------------------------------------------")
    print(f"Total Time Code Ran For: {total_runtime:.2f} seconds".center(66))
    print(f"Total Time a Face Was Detected: {face_detected_time:.2f} seconds".center(66))
    print(f"Total Time a Smile Was Detected: {smile_detected_time:.2f} seconds".center(66))

    if face_detected_time > 0:
        engagement_score = (smile_detected_time) / (face_detected_time)

        # make sure this prints in percentage
        print(f"Engagement Score: {engagement_score:.2%}".center(66))
        print(f"------------------------------------------------------------------")
        return engagement_score * 100
    else:
        print(f"No face was detected, therefore engagement score doesn't apply".center(66))
        print(f"------------------------------------------------------------------")
        return 0

# ===== FastAPI エンドポイント =====

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "Smile Detection API (based on face_detection.py)", "status": "running"}

@app.post("/api/initialize", response_model=InitResponse)
async def initialize():
    """モデルを初期化"""
    # モデルは既に読み込み済み（元のコードの通り）
    if model is not None and face_cascade is not None:
        return InitResponse(success=True, message="モデルが正常に初期化されました")
    else:
        raise HTTPException(status_code=500, detail="モデルの初期化に失敗しました")

@app.post("/api/start-detection", response_model=Dict[str, Any])
async def start_detection():
    """検出を開始"""
    global detection_active, detection_thread
    global face_detected_time, smile_detected_time, start_of_code_time, last_frame_time
    global current_detection
    
    if model is None or face_cascade is None:
        raise HTTPException(status_code=400, detail="モデルが初期化されていません")
    
    if detection_active:
        return {"success": False, "detail": "検出は既にアクティブです"}
    
    try:
        # 元のface_detection.pyの初期化部分を再現
        face_detected_time = 0.0
        smile_detected_time = 0.0
        last_frame_time = time.time()
        start_of_code_time = time.time()
        
        current_detection = {
            'faceDetected': False,
            'smiling': False,
            'confidence': 0.0
        }
        
        detection_active = True
        detection_thread = threading.Thread(target=detection_loop_api, daemon=True)
        detection_thread.start()
        return {"success": True}
    
    except Exception as e:
        detection_active = False
        raise HTTPException(status_code=500, detail=f"検出開始エラー: {str(e)}")

@app.post("/api/stop-detection", response_model=Dict[str, Any])
async def stop_detection():
    """検出を停止"""
    global detection_active, detection_thread
    
    try:
        if detection_active:
            print("Stopping detection...")
            detection_active = False
            
            if detection_thread and detection_thread.is_alive():
                detection_thread.join(timeout=3.0)
                
            detection_thread = None
            
            # 元のface_detection.pyの最終統計を出力
            calculate_final_stats()
            
            print("Detection stopped successfully")
            return {"success": True}
        else:
            return {"success": True, "message": "検出は既に停止されています"}
            
    except Exception as e:
        print(f"Stop detection error: {e}")
        detection_active = False
        detection_thread = None
        return {"success": False, "error": str(e)}

@app.post("/api/reset-stats", response_model=Dict[str, bool])
async def reset_stats():
    """統計データをリセット"""
    global face_detected_time, smile_detected_time, start_of_code_time, last_frame_time
    global current_detection
    
    try:
        # 元のface_detection.pyの初期化部分を再現
        face_detected_time = 0.0
        smile_detected_time = 0.0
        last_frame_time = time.time()
        start_of_code_time = time.time()
        
        current_detection = {
            'faceDetected': False,
            'smiling': False,
            'confidence': 0.0
        }
        
        return {"success": True}
    except Exception as e:
        print(f"Reset stats error: {e}")
        return {"success": False}

@app.get("/api/detection", response_model=DetectionResponse)
async def get_detection():
    """現在の検出結果を取得"""
    return DetectionResponse(
        faceDetected=current_detection['faceDetected'],
        smiling=current_detection['smiling'],
        confidence=current_detection['confidence']
    )

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """現在の統計を取得"""
    current_time = time.time()
    total_time = current_time - start_of_code_time
    
    # 元のface_detection.pyの計算方法を使用
    engagement = 0.0
    if face_detected_time > 0:
        engagement = (smile_detected_time / face_detected_time) * 100
    
    return StatsResponse(
        totalTime=total_time,
        faceTime=face_detected_time,
        smileTime=smile_detected_time,
        engagement=engagement
    )

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """システムステータスを取得"""
    return StatusResponse(
        modelsLoaded=(model is not None and face_cascade is not None),
        detecting=detection_active
    )

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Smile Detection API (based on face_detection.py) を起動中...")
    print(f"📱 API: http://localhost:{API_PORT}")
    print(f"📖 Docs: http://localhost:{API_PORT}/docs")
    print("⏹️  停止するには Ctrl+C を押してください")
    print("-" * 50)
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        log_level="info"
    )