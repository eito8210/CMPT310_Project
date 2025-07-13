import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model('model.keras')
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

IMG_SIZE = 64

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        face_roi = gray[y:y+h, x:x+w]
        try:
            face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            face_nomalized = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_nomalized, axis = (0, -1))

            prediction = model.predict(face_input)[0][0]
            label = "Smiling" if prediction > 0.6 else "Not Smiling"
            color = (0, 255, 0) if label == "Smiling" else (0, 0, 255)

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        except Exception as e:
            print("Face preprocessing failed:", e)

    cv2.imshow('Smile Detection', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()