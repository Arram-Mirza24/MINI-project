# save_model.py
# Run this script to save your trained model

import tensorflow as tf
from app import build_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Build the model architecture
model = build_model()

# Define parameters
train_path = "path/to/your/training/data"  # Update this path
batch_size = 64
epochs = 10

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical")

# Train the model (if you don't have the weights)
hist = model.fit(
    train_generator,
    epochs=epochs
)

# Save the model weights
model.save_weights('waste_classification_model.h5')
print("Model weights saved successfully!")
