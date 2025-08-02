from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import os

valid_formats = [".jpg", ".jpeg", ".png"]

def image_paths(root):
    image_paths = []

    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            extension = os.path.splitext(filename)[1].lower()

            if extension in valid_formats:
                image_path = os.path.join(dirpath, filename)
                image_paths.append(image_path)

    return image_paths

image_paths = image_paths("SMILEs")
IMG_SIZE = [64, 64]

def load_dataset(image_paths, target_size = IMG_SIZE):
    data =[]
    labels = []

    for image_path in image_paths:
        image = load_img(image_path, color_mode = "grayscale", target_size = target_size)
        image = img_to_array(image)
        data.append(image)
    
        label = image_path.split(os.path.sep)[-3]
        label = 1 if label == "positives" else 0
        labels.append(label)

    return np.array(data) / 255.0, np.array(labels)

data, labels = load_dataset(image_paths)

def build_model(input_shape = IMG_SIZE + [1]):
    model = Sequential([
        Conv2D(filters = 32,
               kernel_size = (3, 3),
               activation = "relu",
               padding = "same",
               input_shape = input_shape),
        MaxPool2D(2, 2),
        Conv2D(filters = 64,
               kernel_size = (3, 3),
               activation = "relu",
               padding = "same",
               input_shape = input_shape),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(256, activation = "relu"),
        Dense(1, activation = "sigmoid")
    ])

    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    return model

label, counts = np.unique(labels, return_counts = True)

counts = max(counts) / counts
class_weights = dict(zip(label, counts))

(X_train, X_test, y_train, y_test) = train_test_split(data, labels,
                                                      test_size = 0.2,
                                                      stratify = labels,
                                                      random_state = 42)

(X_train, X_valid, y_train, y_valid) = train_test_split(X_train, y_train,
                                                      test_size = 0.2,
                                                      stratify = y_train,
                                                      random_state = 42)

if len(X_train.shape) == 3:
    X_train = np.expand_dims(X_train, axis = -1)
    X_valid = np.expand_dims(X_valid, axis = -1)
    X_test = np.expand_dims(X_test, axis = -1)

train_datagen = ImageDataGenerator (
    rotation_range = 15,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'
)

train_generator = train_datagen.flow(X_train, y_train, batch_size = 64)

model = build_model()
EPOCHS = 20
history = model.fit(
                    train_generator,
                    validation_data = (X_valid, y_valid),
                    class_weight = class_weights,
                    epochs = EPOCHS)
model.save("model.keras")

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(range(EPOCHS), acc, "b", label = "Training Accuracy")
plt.plot(range(EPOCHS), val_acc, "r", label = "Validation Accuracy")
plt.legend()

plt.figure()

plt.plot(range(EPOCHS), loss, "g", label = "Training Loss")
plt.plot(range(EPOCHS), val_loss, "orange", label = "Validation Loss")
plt.legend()

plt.show()

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("test loss: ", test_loss)
print("test accuracy: ", test_accuracy)