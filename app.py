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
        # We pass the custom layer into the loader here!
        generators[name] = tf.keras.models.load_model(
            model_path, 
            compile=False, 
            custom_objects={'InstanceNormalization': InstanceNormalization}
        )
        print(f"✓ Generator {name} LOADED!")

load_models()

def preprocess_image(image_bytes, filename):
    if filename.endswith('.dcm'):
        # 1a. Read the DICOM file directly from bytes
        dicom = pydicom.dcmread(io.BytesIO(image_bytes))
        img = dicom.pixel_array
        
        # DICOM files have intense 16-bit brightness. Normalize down to standard 8-bit.
        img = img - np.min(img)
        if np.max(img) != 0:
            img = (img / np.max(img) * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    else:
        # 1b. Read standard PNG/JPG files
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
        
    # 2. Resize to exactly what the AI expects (64x64)
    img = cv2.resize(img, (64, 64))
    
    # 3. Normalize pixel values to [-1, 1]
    img = (img / 127.5) - 1.0 
    
    # 4. Add the channel and batch dimensions: (1, 64, 64, 1)
    img = np.expand_dims(img, axis=-1) 
    img = np.expand_dims(img, axis=0)  
    return img
def postprocess_tensor(tensor):
    # 1. Extract the image from the AI's output batch (now 64x64x1)
    img = tensor[0].numpy()
    
    # 2. Denormalize pixel values back to [0, 255]
    img = (img + 1.0) * 127.5 
    img = np.clip(img, 0, 255).astype(np.uint8)
    
    # 3. Drop the single channel dimension so OpenCV can process it as a flat image
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=-1)
        
    # 4. Magically upscale the 64x64 output to High-Res 512x512 using Cubic Interpolation
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC) 
    
    # 5. Convert back to standard BGR so the web browser can read it perfectly
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 6. Encode and send back to frontend
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/convert', methods=['POST'])
def convert():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    conversion_type = request.form.get('type')
    file = request.files['image']
    
    try:
       try:
        # Pass both the file bytes and the filename so it knows if it's a DICOM!
        input_tensor = preprocess_image(file.read(), file.filename.lower())
        
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
