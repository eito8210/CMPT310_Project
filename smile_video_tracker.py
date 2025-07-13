import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import torch
import mediapipe as mp
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

# === 2. Transform (still expects 3 channels) ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225],
    ),
])

# === 3. MediaPipe ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# === 4. Video and face detector ===
cap = cv2.VideoCapture("Smiling_Video.mp4")
if not cap.isOpened():
    print("❌ Could not open video file.")
    exit()

net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 30.0
FRAME_TIME = 1.0 / FPS

out = cv2.VideoWriter(
    "output.avi",
    cv2.VideoWriter_fourcc(*"XVID"),
    FPS,
    (W, H)
)

face_time = 0.0
smile_time = 0.0
SMILE_THRESH = 0.16  # adjust as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_input = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)  # match model input

        # === Predict ===
        inp = transform(face_input).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(inp)
            probs = F.softmax(logits, dim=1)
        smile_prob = probs[0, 1].item()
        smiling = smile_prob > SMILE_THRESH

        # === Annotations ===
        color = (0,255,255) if smiling else (255,255,255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{'Smiling' if smiling else 'Not Smiling'} {smile_prob:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, color, 2
        )

        # === Timers ===
        face_time += FRAME_TIME
        if smiling:
            smile_time += FRAME_TIME

        # === Ellipse ===
        mesh = face_mesh.process(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))  # still use RGB for landmarks
        if mesh.multi_face_landmarks:
            lm = mesh.multi_face_landmarks[0]
            w_rel, h_rel = x2 - x1, y2 - y1
            cx = int(lm.landmark[13].x * w_rel) + x1
            cy = int(lm.landmark[13].y * h_rel) + y1
            ax = int(0.2 * w_rel)
            ay = int(0.1 * h_rel)
            cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360, (0,128,255), 2)

    out.write(frame)
    cv2.imshow("Smile Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Face duration:  {face_time:.3f} s")
print(f"✅ Smile duration: {smile_time:.3f} s")
if face_time > 0:
    print(f"✅ Engagement:     {100 * smile_time / face_time:.1f}%")
else:
    print("⚠️ No face detected.")
