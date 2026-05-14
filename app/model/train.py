import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training images shape:", x_train.shape)
print("Training labels shape:", y_train.shape)

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Save raw digit image properly
raw_image = (x_train[0] * 255).astype(np.uint8)

pil_image = Image.fromarray(raw_image)

pil_image.save("outputs/raw_digit.png")

print("Raw digit image saved!")

# Build neural network
model = tf.keras.models.Sequential([

    # Add channel dimension
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),

    # First convolution layer
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation='relu'
    ),

    # Pooling layer
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second convolution layer
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    # Second pooling
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Flatten for dense layer
    tf.keras.layers.Flatten(),

    # Dense layer
    tf.keras.layers.Dense(64, activation='relu'),

    # Output layer
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(x_train, y_train, epochs=5)

# Evaluate model
test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("\nTest Accuracy:", test_accuracy)

model.save("outputs/mnist_model.keras")

print("Model saved successfully!")