from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from PIL import Image
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

model = tf.keras.models.load_model("malaria_cnn_model.h5")
IMG_SIZE = 128

app.mount("/static", StaticFiles(directory="static"), name="static")

def prepare_image(uploaded_file):
    image = Image.open(uploaded_file.file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    img_array = prepare_image(file)
    prediction = model.predict(img_array)[0][0]
    label = "🦠 Parasitic" if prediction > 0.5 else "✅ Uninfected  "
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": label,
        "confidence": f"{confidence*100:.2f}%",
        "filename": file.filename
    })
