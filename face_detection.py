import numpy as np
import cv2
from tensorflow.keras.models import load_model


# Loading our trained model "model.keras"
model = load_model('model.keras')

# Open up the webcam
cap = cv2.VideoCapture(0)

# This is our face detector using HAAR cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

IMG_SIZE = 64

while True:
    #This gets every frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    #here we convert to gray colour since our model was trained in black and white
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #This is where we attempt to detect the face
    # the first number (the lower the number the more accuracte but slower)
    # The first number is called the scale factor

    #second number is called min neuighbours,the higher the numebr aka 7 it might not detect face
    # if its at like 3 it might give like random boxes (its more sensitive)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    for (x, y, w, h) in faces:
        #This is our visual rectangle on the face (currently set to blue (255, 0, 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

        # roi = region of interest
        face_roi = gray[y:y+h, x:x+w]
        try:

            # this is where we have to resize our ROI to the size that the model expects
            face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
            # normalize our values so that we can get values from (0,1)
            face_nomalized = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_nomalized, axis = (0, -1))

            #this is where we get our smile probability and use it when outputting smiling/not smiling
            prediction = model.predict(face_input)[0][0]


            # open cv uses BlueGreenRed instead of RGB for some reason
            if prediction > 0.6:     #THIS IS OUR SMILING THRESHOLD, CAN PUT IT UP OR DOWN BASED ON RESULTS
                label = "Smiling" 
                color = (0, 255, 0) #green
            else:
                label = "Not Smiling"
                color = (0, 0, 255) #red

            #
           
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        #if anything goes bad it fails
        except Exception as e:
            print("Face preprocessing failed:", e)

    cv2.imshow('Smile Detection', frame)


    #this will exit our while loop if the user presses q
    if cv2.waitKey(1) == ord('q'):
        break

#this closes the webcam and the open cv stuff
cap.release()
cv2.destroyAllWindows()