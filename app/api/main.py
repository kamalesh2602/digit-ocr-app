from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/frontend"), name="static")

# Load trained model once
model = tf.keras.models.load_model(
    "outputs/mnist_model.keras"
)

print("Model loaded successfully!")

@app.get("/")
def home():
    return {
        "message": "Digit OCR API Running"
    }

@app.get("/app")
def serve_frontend():
    return FileResponse("app/frontend/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    image_data = await file.read()

    # Open image using PIL
    image = Image.open(BytesIO(image_data))

    # Convert to grayscale
    image = image.convert("L")

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array
    image_array = np.array(image)

    # Normalize
    image_array = image_array / 255.0

    # Invert colors
    image_array = 1 - image_array

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)

    # Get digit
    predicted_digit = int(np.argmax(prediction))

    # Confidence
    confidence = float(np.max(prediction))

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(confidence * 100, 2)
    }