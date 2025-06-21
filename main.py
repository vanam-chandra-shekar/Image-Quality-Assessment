
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import shutil
import os
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Create FastAPI app
app = FastAPI()

# Load trained CNN model
model_path = "model.h5"  # Change if your model name is different
model = load_model(model_path, compile=False)


# Create a temp directory for uploaded images
os.makedirs("temp", exist_ok=True)

def preprocess_image(image_path):
    """Convert image to numpy array and resize"""
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Match your model's input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.get("/")
def root():
    return {"message": "Image Quality Assessment API is up!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Preprocess image
    try:
        img_array = preprocess_image(file_location)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

    # Predict MOS
    prediction = model.predict(img_array)[0][0]
    classification = "Good" if prediction > 3 else "Bad"

    return {
        "filename": file.filename,
        "MOS_Prediction": round(float(prediction), 2),
        "Quality": classification
    }

