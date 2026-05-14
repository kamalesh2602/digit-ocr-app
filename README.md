# Digit OCR Learning Project

A beginner-friendly OCR and Deep Learning learning project built using TensorFlow, CNNs, FastAPI and OpenCV.

This is **NOT a production-ready OCR system**.
The main purpose of this project was to learn:

* Deep Learning basics
* CNNs (Convolutional Neural Networks)
* OCR preprocessing
* TensorFlow inference
* FastAPI model serving
* Frontend ↔ Backend integration

The focus was understanding the complete AI pipeline rather than achieving perfect real-world accuracy.

---

# Features

* Draw digits directly in browser
* CNN-based digit prediction
* FastAPI backend inference API
* OpenCV preprocessing pipeline
* Contour detection and cropping
* Confidence score prediction
* Model save/load support

---

# Tech Stack

* TensorFlow
* FastAPI
* OpenCV
* Pillow
* NumPy
* HTML/CSS/JavaScript
* UV Package Manager

---

# Project Structure

```text
digit-ocr-app/

├── app/
│
├── api/
│   └── main.py
│
├── frontend/
│   └── index.html
│
├── model/
│   ├── train.py
│   ├── predict.py
│   └── predict_custom.py
│
├── outputs/
│   ├── mnist_model.keras
│   ├── processed_digit.png
│   └── raw_digit.png
│
├── pyproject.toml
└── README.md
```

---

# Important Learning Note

The model is trained using the MNIST dataset.

MNIST digits are:

* clean
* centered
* standardized

Browser-drawn digits are very different from MNIST digits, so prediction accuracy may vary.

This project helped understand an important OCR concept:

> Preprocessing and input quality matter a lot in AI systems.

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone <your-repo-link>
cd digit-ocr-app
```

---

## 2. Install UV

### Linux / Mac

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:

```bash
uv --version
```

---

## 3. Create Virtual Environment

```bash
uv venv
```

Activate environment:

### Linux / Mac

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

---

## 4. Install Dependencies

```bash
uv add tensorflow fastapi uvicorn pillow numpy matplotlib python-multipart opencv-python
```

---

# Running The Project

## Step 1 — Train Model

```bash
uv run app/model/train.py
```

This will:

* train CNN model
* evaluate accuracy
* save trained model inside `outputs/`

Expected accuracy:
~98%+

---

## Step 2 — Start FastAPI Server

```bash
uv run uvicorn app.api.main:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

---

## Step 3 — Open Frontend

Open in browser:

```text
http://127.0.0.1:8000/app
```

---

## Step 4 — Test OCR App

* Draw a digit
* Click Predict
* Observe prediction and confidence score

---

# API Documentation

Swagger UI available at:

```text
http://127.0.0.1:8000/docs
```

---

# Current Limitations

This is a learning project and still has limitations:

* Accuracy varies for browser-drawn digits
* No advanced OCR preprocessing
* No data augmentation
* No deployment setup
* No authentication system
* No database integration

---

# Future Improvements

Possible future improvements:

* Better preprocessing pipeline
* Data augmentation
* Improved CNN architecture
* Docker deployment
* Real OCR for words/documents
* Tesseract/EasyOCR integration
* Better frontend UI

---

# Main Learning Outcome

The biggest learning from this project:

```text
AI systems are not only about models.
Preprocessing and inference pipelines are equally important.
```
