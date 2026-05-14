import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("outputs/mnist_model.keras")

print("Model loaded successfully!")

# Load image
image = Image.open("outputs/raw_digit.png")

# Convert to grayscale
image = image.convert("L")

# Resize to 28x28
image = image.resize((28, 28))

# Convert image to numpy array
image_array = np.array(image)

# Normalize
image_array = image_array / 255.0

# Invert colors
image_array = 1 - image_array

# Display processed image
plt.imshow(image_array, cmap="gray")
plt.title("Processed Image")
plt.axis("off")
plt.show()

processed_image = (image_array * 255).astype(np.uint8)

Image.fromarray(processed_image).save("outputs/processed_digit.png")

print("Processed image saved!")
# Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

print("Image shape:", image_array.shape)
print("Min value:", image_array.min())
print("Max value:", image_array.max())

# Predict
prediction = model.predict(image_array)

# Get predicted digit
predicted_digit = np.argmax(prediction)

print("\nPredicted Digit:", predicted_digit)

