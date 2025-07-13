import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import torch
import time
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch import nn

# === 1. Load pretrained model ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("smile_resnet18.pt", map_location=DEVICE))
model.to(DEVICE).eval()

# === 2. Image transform (expects RGB)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# === 3. Face detector (OpenCV DNN)
net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

# === 4. Video setup
cap = cv2.VideoCapture(0)  # Webcam
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
FRAME_TIME = 1.0 / FPS

face_time = 0.0
smile_time = 0.0
SMILE_THRESH = 0.10  # Set lower if needed

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    dets = net.forward()

    best = None
    best_area = 0
    for i in range(dets.shape[2]):
        conf = dets[0, 0, i, 2]
        if conf < 0.5:
            continue
        x1, y1, x2, y2 = (dets[0,0,i,3:7] * [W, H, W, H]).astype(int)
        area = (x2 - x1)*(y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)

    if best:
        x1, y1, x2, y2 = best
        face_crop = frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        inp = transform(rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(inp)
            probs = F.softmax(logits, dim=1)
        smile_prob = probs[0, 1].item()
        smiling = smile_prob > SMILE_THRESH

        color = (0,255,255) if smiling else (255,255,255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{'Smiling' if smiling else 'Not Smiling'} {smile_prob:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )

        face_time += FRAME_TIME
        if smiling:
            smile_time += FRAME_TIME

    cv2.putText(
        frame,
        "Press 'Q' to quit",
        (10, H - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,                # ⬅️ Larger font scale
        (0, 255, 0),        # ⬅️ Lime green in BGR
        2                   # ⬅️ Thicker line
    )
    cv2.imshow("Smile Webcam Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n✅ Face duration:  {face_time:.3f} s")
print(f"✅ Smile duration: {smile_time:.3f} s")
if face_time > 0:
    print(f"✅ Engagement:     {100 * smile_time / face_time:.1f}%")
else:
    print("⚠️ No face detected.")
