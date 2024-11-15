import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

# ====================
# 1. Load and Preprocess Dataset
# ====================
def load_dataset(dataset_path, image_size=(224, 224), batch_size=32):
    # Load training dataset
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )
    # Load validation dataset
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )
    # Normalize pixel values
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
    return train_dataset, val_dataset

# ====================
# 2. Define the VGG16 Model
# ====================
def create_model(num_classes):
    # Load the VGG16 model without the top classification layers
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    # Add custom classification layers
    x = layers.Flatten()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Regularization
    output = layers.Dense(num_classes, activation='softmax')(x)
    # Create the final model
    model = models.Model(inputs=base_model.input, outputs=output)
    return model

# ====================
# 3. Train the Model
# ====================
def train_model(model, train_dataset, val_dataset, epochs=10):
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )
    return history

# ====================
# 4. Main Execution
# ====================
if __name__ == "__main__":
    # Path to your dataset (update this path)
    dataset_path = "extracted_letter_images"
    
    # Load dataset
    train_dataset, val_dataset = load_dataset(dataset_path)

   

    # Create the model
    model = create_model(36)

    # Train the model
    print("Training the model...")
    history = train_model(model, train_dataset, val_dataset, epochs=10)

    # Evaluate the model
    loss, accuracy = model.evaluate(val_dataset)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")

    # Save the trained model
    model.save("vgg16_character_classifier.h5")
    print("Model saved as 'vgg16_character_classifier.h5'")
