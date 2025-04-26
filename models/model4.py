import cv2
import pickle
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.layers import LeakyReLU, GlobalAveragePooling2D, Add, Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

LETTER_IMAGES_FOLDER = "extracted_letter_images"
MODEL_FILENAME = "captcha_recognition_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"

def create_residual_block(input_tensor, filters, kernel_size=(3, 3)):
    """Create a residual block with skip connection"""
    x = Conv2D(filters, kernel_size, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Skip connection
    if input_tensor.shape[-1] != filters:
        input_tensor = Conv2D(filters, (1, 1), padding='same')(input_tensor)
    
    x = Add()([x, input_tensor])
    x = LeakyReLU(alpha=0.1)(x)
    return x

def build_enhanced_model(input_shape=(40, 40, 1), num_classes=36):
    """Build an enhanced model with residual connections and modern architecture"""
    inputs = Input(shape=input_shape)
    
    # Initial convolution block
    x = Conv2D(64, (7, 7), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Residual blocks with increasing filters
    x = create_residual_block(x, 64)
    x = Dropout(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = create_residual_block(x, 128)
    x = Dropout(0.2)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = create_residual_block(x, 256)
    x = Dropout(0.3)(x)
    
    # Global pooling instead of flattening
    x = GlobalAveragePooling2D()(x)
    
    # Dense layers with skip connections
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Data loading and preprocessing (same as before)
data = []
labels = []

for label in os.listdir(LETTER_IMAGES_FOLDER):
    label_folder = os.path.join(LETTER_IMAGES_FOLDER, label)
    if os.path.isdir(label_folder):
        for image_file in os.listdir(label_folder):
            image_path = os.path.join(label_folder, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (40, 40))
            image = np.expand_dims(image, axis=2)
            data.append(image)
            labels.append(label)

# Convert to numpy arrays and normalize
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

# Split data
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert labels
all_labels = [chr(i) for i in range(97, 123)] + [str(i) for i in range(10)]
lb = LabelBinarizer().fit(all_labels)
Y_train = lb.transform(Y_train)
Y_test = lb.transform(Y_test)

# Save label binarizer
with open(MODEL_LABELS_FILENAME, "wb") as f:
    pickle.dump(lb, f)

# Enhanced data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    fill_mode="nearest",
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,
    vertical_flip=False
)

datagen.fit(X_train)

# Build and compile model
model = build_enhanced_model()
optimizer = Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", 
              optimizer=optimizer,
              metrics=["accuracy"])

# Enhanced callbacks
checkpoint = ModelCheckpoint(
    MODEL_FILENAME,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    mode='max'
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# Train with enhanced parameters
model.fit(
    datagen.flow(X_train, Y_train, batch_size=32),
    validation_data=(X_test, Y_test),
    epochs=150,
    callbacks=[checkpoint, early_stopping, lr_scheduler],
    verbose=1
)