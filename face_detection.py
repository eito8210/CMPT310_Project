import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('model.keras')
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

IMG_SIZE = 64

net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel"
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            face_roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)

            try:
                face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
                face_nomalized = face_resized.astype("float32") / 255.0
                face_input = np.expand_dims(face_nomalized, axis=(0, -1))

                prediction = model.predict(face_input)[0][0]
                label = "Smiling" if prediction > 0.6 else "Not Smiling"
                color = (0, 255, 0) if label == "Smiling" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            except Exception as e:
                print("Face preprocessing failed:", e)


    cv2.imshow('Smile Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()