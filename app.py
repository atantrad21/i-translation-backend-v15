import os
import base64
import gdown
import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# V16 CHAMPION MODELS ONLY (F & G)
# ==========================================
MODEL_LINKS = {
    'F': '1NTBlkD3MQPfjoAN2rRoySoaCNqsTkELZ', # MRI to CT Champion
    'G': '15YPfERDoVbTWHPzzAn54OKpRVpvFOyRe'  # CT to MRI Champion
}

generators = {}

def load_models():
    print("======================================================================")
    print("LOADING V16 CHAMPION GENERATORS (F & G)")
    print("======================================================================")
    for name, file_id in MODEL_LINKS.items():
        model_path = f"/tmp/generator_{name.lower()}.h5"
        if not os.path.exists(model_path):
            print(f"Downloading Generator {name}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
        
        print(f"Loading Generator {name} into memory...")
        generators[name] = tf.keras.models.load_model(model_path, compile=False)
        print(f"✓ Generator {name} LOADED!")

load_models()

def preprocess_image(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    img = (img / 127.5) - 1.0 # Normalize to [-1, 1]
    img = np.expand_dims(img, axis=0)
    return img

def postprocess_tensor(tensor):
    img = tensor[0].numpy()
    img = (img + 1.0) * 127.5 # Denormalize
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC) # High-Res 512x512
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/convert', methods=['POST'])
def convert():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    conversion_type = request.form.get('type')
    file = request.files['image']
    
    try:
        input_tensor = preprocess_image(file.read())
        
        # Route perfectly to the winning models
        if conversion_type == 'ct_to_mri':
            result_tensor = generators['G'](input_tensor, training=False)
            return jsonify({'image_G': postprocess_tensor(result_tensor)})
            
        elif conversion_type == 'mri_to_ct':
            result_tensor = generators['F'](input_tensor, training=False)
            return jsonify({'image_F': postprocess_tensor(result_tensor)})
            
        else:
            return jsonify({'error': 'Invalid conversion type'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
