import os
import io
import base64
import gdown
import pydicom
import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ==========================================
# CYCLEGAN CUSTOM LAYER
# ==========================================
class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-5, center=True, scale=True, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of input tensor should have a defined dimension')
        if self.scale:
            self.gamma = self.add_weight(shape=(dim,), name='gamma', initializer='ones')
        if self.center:
            self.beta = self.add_weight(shape=(dim,), name='beta', initializer='zeros')
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        outputs = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        if self.scale:
            outputs = outputs * self.gamma
        if self.center:
            outputs = outputs + self.beta
        return outputs

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({'axis': self.axis, 'epsilon': self.epsilon, 'center': self.center, 'scale': self.scale})
        return config

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
        generators[name] = tf.keras.models.load_model(
            model_path, 
            compile=False, 
            custom_objects={'InstanceNormalization': InstanceNormalization}
        )
        print(f"✓ Generator {name} LOADED!")

load_models()

# === THE SMART IMAGE PROCESSOR ===
def preprocess_image(image_bytes, filename, target_shape):
    # Extract exactly what the AI expects (e.g. 64x64x3 or 64x64x1)
    _, target_h, target_w, target_c = target_shape
    
    if filename.endswith('.dcm'):
        dicom = pydicom.dcmread(io.BytesIO(image_bytes))
        img = dicom.pixel_array
        img = img - np.min(img)
        if np.max(img) != 0:
            img = (img / np.max(img) * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    else:
        np_img = np.frombuffer(image_bytes, np.uint8)
        # Read the upload in Color by default
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    # Match the AI's required color channels perfectly
    if len(img.shape) == 2: # If image is Grayscale
        if target_c == 3:   # But AI wants Color
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3: # If image is Color
        if target_c == 1:     # But AI wants Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
    # Resize to AI's required dimensions
    img = cv2.resize(img, (target_w, target_h))
    img = (img / 127.5) - 1.0 # Normalize
    
    if target_c == 1 and len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1) 
        
    img = np.expand_dims(img, axis=0) # Add batch dimension
    return img

def postprocess_tensor(tensor):
    img = tensor[0].numpy()
    img = (img + 1.0) * 127.5 
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # Scale back up to 512x512 High Res
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=-1)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC) 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC) 
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
        # Determine which model we are using
        model_key = 'G' if conversion_type == 'ct_to_mri' else 'F'
        model = generators[model_key]
        
        # Dynamically fetch what shape this specific model wants!
        expected_shape = model.input_shape
        
        # Preprocess the image to perfectly match the model
        input_tensor = preprocess_image(file.read(), file.filename.lower(), expected_shape)
        
        # Run translation
        result_tensor = model(input_tensor, training=False)
        
        return jsonify({f'image_{model_key}': postprocess_tensor(result_tensor)})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
