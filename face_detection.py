import numpy as np
import cv2
from tensorflow.keras.models import load_model
import time


# Loading our trained model "model.keras"
model = load_model('model.keras')

# Open up the webcam
cap = cv2.VideoCapture(0)

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

    
    # JAKES FACE SCORE CODE: this is where I am setting up how I will subtract the time where no face is detected
          #time.time() will not add any time, it just retunrs the time it is during the day
    current_time= time.time()
    frame_time = (current_time) - (last_frame_time)

    # this will update the frame to the last time we saw a face and we use it in computation above
    last_frame_time = current_time

    for (x, y, w, h) in faces:
        #This is our visual rectangle on the face (currently set to blue (255, 0, 0))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

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
            prediction = model.predict(face_input)[0][0]


            # open cv uses BlueGreenRed instead of RGB for some reason
            if prediction > 0.10:     #THIS IS OUR SMILING THRESHOLD, CAN PUT IT UP OR DOWN BASED ON RESULTS
                #add smile time
                smile_detected_time += frame_time
                label = f"Smiling ({prediction:.2f})" 
                color = (0, 255, 0) #green
            else:
                label = f"Not Smiling ({prediction:.2f})"
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
else:
    print(f"No face was detected, therefore engagement score doesn't apply".center(66))
    print(f"------------------------------------------------------------------")
