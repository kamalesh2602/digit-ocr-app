import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training images shape:", x_train.shape)
print("Training labels shape:", y_train.shape)

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build neural network
model = tf.keras.models.Sequential([
    
    # Flatten 28x28 image into 784 values
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # Hidden layer
    tf.keras.layers.Dense(128, activation='relu'),
    
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

model.save("app/model/mnist_model.keras")

print("Model saved successfully!")