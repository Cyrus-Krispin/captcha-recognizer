import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_data(data_dir="extracted_letter_images"):
    images = []
    labels = []

    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (28, 28))  # Resize to a fixed size (e.g., 28x28)
                img = img / 255.0  # Normalize the image to [0, 1]
                images.append(img)
                labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # Convert labels to integers (since folders are the characters)
    label_to_int = {char: idx for idx, char in enumerate(sorted(set(labels)))}
    labels = np.array([label_to_int[label] for label in labels])

    # Reshape to include a channel dimension
    images = images.reshape(images.shape[0], 28, 28, 1)

    # One-hot encode labels
    labels = to_categorical(labels)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_to_int




def build_model(input_shape=(28, 28, 1), num_classes=36):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the data
X_train, X_test, y_train, y_test, label_to_int = load_data()

# Build the model
model = build_model(input_shape=(28, 28, 1), num_classes=len(label_to_int))

# Train the model
model.fit(X_train, y_train, epochs=60, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

model.save("captcha_recognition_model.h5")