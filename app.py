from flask import Flask, request, jsonify
import numpy as np
import cv2
import detectar_personas

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def recibir_imagen():
    image_data = request.data
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Imagen no válida"}), 400

    personas = detectar_personas.contar_personas(img)
    return jsonify({"personas": personas})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
