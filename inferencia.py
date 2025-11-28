import os
import numpy as np
from PIL import Image
import tensorflow as tf

model = "model.tflite"
img_size = (224, 224)
classes = ["bus", "car", "motorcycle", "truck"]

interpreter = tf.lite.Interpreter(model_path=model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Modelo carregado com sucesso!\n")


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(img_size)
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


img_paths = [
    "test/truck.jpg",
    "test/delorean.jpg",
    "test/fusca.jpg",
    "test/moto.jpg",
    "test/buss.jpg",
    "test/onix.jpg",
    "test/1113.jpg",
    "test/skyline.jpg",
    "test/fire.jpg",
    "test/audi.jpg",
    "test/ambulance.jpg",
]


for img_path in img_paths:
    img = preprocess_image(img_path)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])[0]

    pred_idx = int(np.argmax(output))
    confidences = np.round(output, 2)

    print(f"Imagem: {img_path}")
    print("Probabilidades:", confidences)
    print("Predicao:", classes[pred_idx])
    print("-" * 30)
