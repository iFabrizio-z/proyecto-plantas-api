import io
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile

BASE = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE.parent / "modelos" / "mejor_modelo.keras")))
CLASSES_PATH = Path(os.getenv("CLASSES_PATH", str(BASE.parent / "modelos" / "clases.json")))

with open(CLASSES_PATH, encoding="utf-8") as f:
    info = json.load(f)

CLASSES = info["class_names"]
IMG_SIZE = tuple(info.get("image_size", [224, 224]))
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

app = FastAPI(title="API de detección de enfermedades de plantas")

@app.get("/")
def home():
    return {"mensaje": "API de plantas funcionando"}

@app.post("/predecir")
async def predecir(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(400, "Envía una imagen JPEG, PNG o WEBP")

    contenido = await file.read()
    if len(contenido) > 8 * 1024 * 1024:
        raise HTTPException(413, "La imagen supera el límite de 8 MB")

    try:
        imagen = Image.open(io.BytesIO(contenido)).convert("RGB")
        imagen = imagen.resize((IMG_SIZE[1], IMG_SIZE[0]))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "No se pudo leer la imagen")

    # El modelo ya incluye Rescaling: aquí se mantienen valores de 0 a 255.
    arreglo = np.asarray(imagen, dtype=np.float32)
    arreglo = np.expand_dims(arreglo, axis=0)

    probabilidades = model.predict(arreglo, verbose=0)[0]
    indices = np.argsort(probabilidades)[::-1][:5]
    mejor = int(indices[0])

    return {
        "prediccion": CLASSES[mejor],
        "confianza": round(float(probabilidades[mejor]), 4),
        "top_5": [
            {
                "clase": CLASSES[int(i)],
                "confianza": round(float(probabilidades[i]), 4)
            }
            for i in indices
        ],
        "aviso": "Predicción orientativa; no garantiza un diagnóstico correcto."
    }
