from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

# Cargar modelos TFLite
interpreter_cnn = tf.lite.Interpreter(model_path="models/h5/cnn_model.tflite")
interpreter_fc = tf.lite.Interpreter(model_path="models/h5/fc_model.tflite")

interpreter_cnn.allocate_tensors()
interpreter_fc.allocate_tensors()

# Obtener info de entrada/salida
input_cnn = interpreter_cnn.get_input_details()
output_cnn = interpreter_cnn.get_output_details()

input_fc = interpreter_fc.get_input_details()
output_fc = interpreter_fc.get_output_details()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    image_data = data["image"]
    model_type = data.get("model", "cnn")

    _, encoded = image_data.split(",", 1)
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded)).convert("RGB")
    image = image.resize((100, 100))
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    if model_type == "cnn":
        interpreter_cnn.set_tensor(input_cnn[0]['index'], arr)
        interpreter_cnn.invoke()
        output = interpreter_cnn.get_tensor(output_cnn[0]['index'])[0][0]
    else:
        interpreter_fc.set_tensor(input_fc[0]['index'], arr)
        interpreter_fc.invoke()
        output = interpreter_fc.get_tensor(output_fc[0]['index'])[0][0]

    if output < 0.4:
        label = "Gato"
    elif output > 0.6:
        label = "Perro"
    else:
        label = "No claro"

    return jsonify({"resultado": label, "confianza": round(float(output), 3)})

if __name__ == "__main__":
    app.run(debug=True)
