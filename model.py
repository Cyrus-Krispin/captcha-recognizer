import cv2
import pickle
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_recognition_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

# Initialize the data and labels
data = []
labels = []

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size without using imutils.
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """
    (h, w) = image.shape[:2]

    # Calculate the aspect ratio and resize accordingly
    if w > h:
        new_w = width
        new_h = int((height / float(w)) * h)
        image = cv2.resize(image, (new_w, new_h))
    else:
        new_h = height
        new_w = int((width / float(h)) * w)
        image = cv2.resize(image, (new_w, new_h))

    padW = (width - image.shape[1]) // 2
    padH = (height - image.shape[0]) // 2

    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    return image

# Loop over the input images
for label in os.listdir(LETTER_IMAGES_FOLDER):
    #print("[INFO] Processing images for letter: {}".format(label))
    label_folder = os.path.join(LETTER_IMAGES_FOLDER, label)
    #print(label_folder)
    if os.path.isdir(label_folder):  # Check if it is a directory
        for image_file in os.listdir(label_folder):
            image_path = os.path.join(label_folder, image_file)
            #print(image_path)

            # Load the image and convert it to grayscale
            image = cv2.imread(image_path)
            if image is None:
                continue  # Skip if the image cannot be loaded
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize the letter so it fits in a 20x20 pixel box
            image = resize_to_fit(image, 20, 20)

            # Add a third channel dimension to the image to make Keras happy
            image = np.expand_dims(image, axis=2)

            # Add the letter image and its label to our training data
            data.append(image)
            labels.append(label)

# Scale the raw pixel intensities to the range [0, 1] (this improves training)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data shape: {}".format(data.shape))
print("[INFO] labels shape: {}".format(labels.shape))

# Split the training data into separate train and test sets
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=0)

# Convert the labels (letters) into one-hot encodings that Keras can work with
# Define all possible labels: lowercase letters + digits
all_labels = [chr(i) for i in range(97, 123)] + [str(i) for i in range(10)]  # 'a' to 'z' + '0' to '9'
lb = LabelBinarizer().fit(all_labels)  # Fit on the expected labels
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save the mapping from labels to one-hot encodings
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Build the neural network!
model = Sequential()

# First convolutional layer with max pooling
model.add(Conv2D(20, (5, 5), padding="same", input_shape=(20, 20, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Second convolutional layer with max pooling
model.add(Conv2D(50, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# Hidden layer with 500 nodes
model.add(Flatten())
model.add(Dense(500, activation="relu"))

# Output layer with 36 nodes (26 letters + 10 digits)
model.add(Dense(36, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the neural network
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=32, epochs=10, verbose=1)

# Save the trained model to disk
model.save(MODEL_FILENAME)
