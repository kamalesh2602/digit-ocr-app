import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import os
# Create output directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)
# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training images shape:", x_train.shape)
print("Training labels shape:", y_train.shape)

print("\nBefore normalization:")
print("Min pixel value:", x_train[0].min())
print("Max pixel value:", x_train[0].max())

# Normalize dataset
x_train = x_train / 255.0
x_test = x_test / 255.0

print("\nAfter normalization:")
print("Min pixel value:", x_train[0].min())
print("Max pixel value:", x_train[0].max())

# Display first image
plt.imshow(x_train[0], cmap="gray")

# Show title
plt.title(f"Label: {y_train[0]}")

# Remove axis numbers
plt.axis("off")

# Save image
plt.tight_layout()

plt.savefig(
    "outputs/digit.png",
    format="png",
    dpi=300
)

plt.close()

print("Image saved as digit.png")