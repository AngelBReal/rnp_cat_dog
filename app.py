from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

# Cargar ambos modelos
cnn_model = tf.keras.models.load_model("model/cnn_model.h5")
fc_model = tf.keras.models.load_model("model/fc_model.h5")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data["image"]
    model_type = data.get("model", "cnn")  # cnn o fc

    # Decodificar imagen base64
    _, encoded = image_data.split(",", 1)
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded)).convert("RGB")
    image = image.resize((100, 100))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    # Seleccionar modelo
    model = cnn_model if model_type == "cnn" else fc_model
    pred = model.predict(arr)[0][0]

    if pred < 0.4:
        label = "Gato"
    elif pred > 0.6:
        label = "Perro"
    else:
        label = "No claro"

    return jsonify({"resultado": label, "confianza": round(float(pred), 3)})

if __name__ == "__main__":
    app.run(debug=True)
