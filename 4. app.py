from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = FastAPI()

# Load the model
model = load_model("nested_unet_model.h5")

def preprocess_uploaded_image(img):
    # Convert and preprocess the uploaded image
    img = cv2.imdecode(np.fromstring(img, np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))  # Adjust based on model input size
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = await file.read()
    img_array = preprocess_uploaded_image(img)

    # Use the model for prediction
    prediction = model.predict(img_array)
    prediction = (prediction[0] > 0.5).astype(np.uint8)  # Binarize the output

    return JSONResponse(content={"prediction": prediction.tolist()})
