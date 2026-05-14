import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

# Load trained model
model = tf.keras.models.load_model("outputs/mnist_model.keras")

print("Model loaded successfully!")

# Load MNIST dataset again
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_test = x_test / 255.0

# Select one test image
sample_image = x_test[0]

# Add batch dimension
sample_image = np.expand_dims(sample_image, axis=0)

# Predict
prediction = model.predict(sample_image)

# Get predicted digit
predicted_digit = np.argmax(prediction)

print("\nPredicted Digit:", predicted_digit)
print("Actual Digit:", y_test[0])