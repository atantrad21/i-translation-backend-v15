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
    'F': '1NTBlkD3MQPfjoAN2rRoySoaCNqsTkELZ',  # MRI to CT Champion
    'G': '15YPfERDoVbTWHPzzAn54OKpRVpvFOyRe'   # CT to MRI Champion
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


# === STRICT 1-CHANNEL GRAYSCALE PROCESSOR ===
import imageio.v2 as imageio # Add this to your imports at the top!

# === STRICT 1-CHANNEL GRAYSCALE PROCESSOR (IMAGEIO MATCH) ===
# === HYBRID PROCESSOR (PYDICOM SAFETY + IMAGEIO MATH) ===
# === HYBRID PROCESSOR (STRICT 217x181 GEOMETRY) ===
def preprocess_image(image_bytes, filename, expected_shape):
    
    # --- YOUR CUSTOM TRAINING GEOMETRY ---
    # If the image looks "squashed" horizontally, swap these two numbers!
    target_h = 217 
    target_w = 181 

    if filename.endswith('.dcm'):
        # 1. PYDICOM SAFETY (Bypasses the 156826 padding crash)
        dicom = pydicom.dcmread(io.BytesIO(image_bytes))
        img = dicom.pixel_array.astype(np.float32)

        # Handle 3D scans (grab middle slice)
        if len(img.shape) == 3 and img.shape[2] not in [1, 3, 4]:
            img = img[img.shape[0] // 2]
        elif len(img.shape) == 4:
            img = img[0, img.shape[1] // 2]

        if len(img.shape) == 3:
            img = np.mean(img, axis=-1)

        # 2. IMAGEIO MATH (Match Training Contrast)
        p_low, p_high = np.percentile(img, (0.1, 99.9)) # Safely ignore bright artifacts
        img = np.clip(img, p_low, p_high)

        if p_high - p_low > 0:
            img = (img - p_low) / (p_high - p_low) * 255.0
        else:
            img = np.zeros_like(img)
        
        img = img.astype(np.uint8)

    else:
        # REGULAR PNG/JPG UPLOADS
        temp_path = f"/tmp/temp_upload_{filename}"
        with open(temp_path, "wb") as f:
            f.write(image_bytes)

        try:
            img = imageio.imread(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        img = np.array(img).astype(np.float32)
        if len(img.shape) == 3:
            img = np.mean(img, axis=-1)

    # --- 3. STRICT GEOMETRY RESIZE ---
    # We forcefully resize to your specific training dimensions
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # --- 4. NORMALIZE TO [-1, 1] & FORCE SHAPE ---
    img = (img.astype(np.float32) / 127.5) - 1.0
    img = img.reshape((1, target_h, target_w, 1))
    
    return img
def postprocess_tensor(tensor):
    if hasattr(tensor, 'numpy'):
        img = tensor[0].numpy()
    else:
        img = tensor[0]
        
    # Denormalize from [-1.0, 1.0] to [0, 255]
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)

    # --- HIGH-QUALITY UPSCALING ---
    # INTER_CUBIC is best for upscaling from 256x256 to 512x512
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=-1)
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # BGR for cv2 encoding
    else:
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # --- GAN ARTIFACT REDUCTION ---
    # A subtle bilateral filter smooths out artificial pixel noise while keeping edges (like the skull/tissues) sharp
    img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)

    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


@app.route('/convert', methods=['POST'])
def convert():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    conversion_type = request.form.get('type')
    file = request.files['image']

    try:
        model_key = 'G' if conversion_type == 'ct_to_mri' else 'F'
        model = generators[model_key]

        expected_shape = model.input_shape
        input_tensor = preprocess_image(file.read(), file.filename.lower(), expected_shape)

        result_tensor = model(input_tensor, training=False)

        return jsonify({f'image_{model_key}': postprocess_tensor(result_tensor)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
