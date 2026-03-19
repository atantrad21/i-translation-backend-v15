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


# === THE SMART IMAGE PROCESSOR ===
def preprocess_image(image_bytes, filename, expected_shape):
    # 1. SAFELY EXTRACT SHAPE (Force standard Python integers to prevent OpenCV C++ errors)
    if isinstance(expected_shape, list):
        target_shape = expected_shape[0]
    else:
        target_shape = expected_shape
        
    target_h = int(target_shape[1]) if target_shape[1] is not None else 256
    target_w = int(target_shape[2]) if target_shape[2] is not None else 256
    target_c = int(target_shape[3]) if target_shape[3] is not None else 3

    if filename.endswith('.dcm'):
        dicom = pydicom.dcmread(io.BytesIO(image_bytes))
        img = dicom.pixel_array.astype(np.float32)

        # --- HANDLE 3D SCANS (GRAB THE MIDDLE SLICE) ---
        if len(img.shape) == 3 and img.shape[2] not in [1, 3, 4]:
            mid_index = img.shape[0] // 2
            img = img[mid_index]
        elif len(img.shape) == 4:
            mid_index = img.shape[1] // 2
            img = img[0, mid_index]

        # --- FIX INVERTED COLORS ---
        if getattr(dicom, 'PhotometricInterpretation', '') == 'MONOCHROME1':
            img = np.max(img) - img

        # --- STANDARD LINEAR NORMALIZATION (GAN-Safe) ---
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img = np.zeros_like(img)
        
        img = img.astype(np.uint8)

        # Match AI Color Channels (Keep as 2D for now if Grayscale)
        if target_c == 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    else:
        # --- HANDLE STANDARD PNG/JPG UPLOADS ---
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not read image data.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Match AI Color Channels (Keep as 2D for now if Grayscale)
        if target_c == 1:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # --- 2. HIGH-QUALITY RESIZE ---
    # OpenCV safely resizes the pure (H,W) or (H,W,3) array here without crashing
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # --- 3. EXPAND DIMS & NORMALIZE TO [-1, 1] ---
    # Now we safely add the channel dimension back if the model expects 1 channel
    if target_c == 1 and len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    # Scale strictly to [-1.0, 1.0]
    img = (img.astype(np.float32) / 127.5) - 1.0
    
    # Add the batch dimension: (1, H, W, C)
    img = img.reshape((1, target_h, target_w, target_c))
    
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
