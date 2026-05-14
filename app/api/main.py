from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import cv2

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

    # Open image
    image = Image.open(BytesIO(image_data))

    # Convert to grayscale
    image = image.convert("L")

    # Convert PIL image to numpy array
    image_array = np.array(image)

    # Threshold image
    _, thresh = cv2.threshold(
        image_array,
        50,
        255,
        cv2.THRESH_BINARY
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # No digit detected
    if len(contours) == 0:
        return {
            "error": "No digit detected"
        }

    # Get largest contour
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    # Crop digit
    digit = thresh[y:y+h, x:x+w]

    # Get digit dimensions
    h, w = digit.shape

    # Create square canvas with padding
    size = max(h, w) + 40

    square = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    # Center digit inside square
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2

    square[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = digit

    # Resize to 28x28
    digit = cv2.resize(
        square,
        (28, 28)
    )

    # Normalize
    digit = digit / 255.0

    # Save processed image for debugging
    processed_image = (
        digit * 255
    ).astype(np.uint8)

    Image.fromarray(
        processed_image
    ).save(
        "outputs/processed_digit.png"
    )

    # Add batch dimension
    digit = np.expand_dims(
        digit,
        axis=0
    )

    # Predict
    prediction = model.predict(digit)

    # Get predicted digit
    predicted_digit = int(
        np.argmax(prediction)
    )

    # Confidence
    confidence = float(
        np.max(prediction)
    )

    return {
        "predicted_digit": predicted_digit,
        "confidence": round(
            confidence * 100,
            2
        )
    }