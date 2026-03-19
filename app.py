import os
import io
import base64
import gdown
import pydicom
import tensorflow as tf
import numpy as np
import cv2
import imageio.v2 as imageio # Required for matching the training PNG contrast
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


# ==========================================
# === THE STRICT 64x64 TRANSLATOR ===
# ==========================================
def preprocess_image(image_bytes, filename, expected_shape):
    # 1. OBEY THE MODEL'S BAKED-IN SHAPE (64x64)
    target_h = 64
    target_w = 64

    if filename.endswith('.dcm'):
        # --- THE PNG SIMULATION ---
        dicom = pydicom.dcmread(io.BytesIO(image_bytes))
        img = dicom.pixel_array.astype(np.float32)

        # Handle 3D scans
        if len(img.shape) == 3 and img.shape[2] not in [1, 3, 4]:
            img = img[img.shape[0] // 2]
        elif len(img.shape) == 4:
            img = img[0, img.shape[1] // 2]

        if len(img.shape) == 3:
            img = np.mean(img, axis=-1)

        # Emulate the 8-bit image scaling
        img_min = np.min(img)
        img_max = np.max(img)
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min) * 255.0
        else:
            img = np.zeros_like(img)
        
        img = img.astype(np.uint8)
        
        # Bake the 8-bit PNG compression artifacts into the array
        _, encoded_png = cv2.imencode('.png', img)
        img = cv2.imdecode(encoded_png, cv2.IMREAD_GRAYSCALE)

    else:
        # --- REGULAR PNG/JPG UPLOADS ---
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read image data.")

    # --- 2. SQUASH TO 64x64 ---
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # --- 3. NORMALIZE TO [-1, 1] & FORMAT FOR KERAS ---
    img = (img.astype(np.float32) / 127.5) - 1.0
    
    # Safely reshape to (1, 64, 64, 1) so functional_563 accepts it perfectly
    img = img.reshape((1, target_h, target_w, 1))
    
    return img


# ==========================================
# === SMART UPSCALER (BAND-AID SUPER RESOLUTION) ===
# ==========================================
def postprocess_tensor(tensor):
    if hasattr(tensor, 'numpy'):
        img = tensor[0].numpy()
    else:
        img = tensor[0]
        
    # --- 1. DENORMALIZE FROM AI TENSOR ---
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert strictly to 2D Grayscale array
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=-1)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # --- 2. HIGH-FIDELITY UPSCALING ---
    # Lanczos4 is mathematically far superior to Cubic when stretching tiny 64x64 images
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)

    # --- 3. UNSHARP MASKING (Fakes Super-Resolution) ---
    # This specifically detects blurry edges and artificially sharpens them
    gaussian_blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)

    # --- 4. GAN ARTIFACT REDUCTION ---
    # Smooths out the tiny "checkerboard" static left behind by the CycleGAN
    img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)

    # Convert back to BGR so the web browser can read the colors properly
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


# ==========================================
# FLASK ROUTING
# ==========================================
@app.route('/convert', methods=['POST'])
def convert():
