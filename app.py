"""
I-Translation Medical Image Converter - Backend API
Version: v15.0 - Maximum Denoising (512x512)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os
import sys
import io
import base64
import cv2
import gdown
import pydicom
import tensorflow as tf
from tensorflow.keras import layers
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print(f"[INFO] Python version: {sys.version}")
print(f"[INFO] TensorFlow version: {tf.__version__}")
print("[INFO] V15.0 - Advanced Denoising Pipeline")

class InstanceNormalization(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale', shape=(input_shape[-1],),
            initializer='ones', trainable=True
        )
        self.offset = self.add_weight(
            name='offset', shape=(input_shape[-1],),
            initializer='zeros', trainable=True
        )
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config

def download_from_gdrive(file_id, destination):
    try:
        logger.info(f"Downloading model from Google Drive...")
        gdown.download(id=file_id, output=destination, quiet=False)
        if os.path.exists(destination) and os.path.getsize(destination) > 10000:
            return True
        return False
    except Exception as e:
        logger.error(f"gdown failed: {str(e)}")
        return False

def load_all_generators():
    logger.info("="*70)
    logger.info("LOADING ALL 4 GENERATORS")
    logger.info("="*70)
    
    file_ids = {
        'F': '1NTBlkD3MQPfjoAN2rRoySoaCNqsTkELZ',
        'G': '15YPfERDoVbTWHPzzAn54OKpRVpvFOyRe',
        'I': '1K2DTtrsYpeB4XILn8eZAU4G6a3lty065',
        'J': '1Reo76L5CCybAplmj_pPNWZCWFLK6n8Zp'
    }
    
    generators = {}
    
    for name, file_id in file_ids.items():
        try:
            logger.info(f"Processing Generator {name}...")
            model_file = f"/tmp/generator_{name.lower()}.h5"
            
            if not download_from_gdrive(file_id, model_file):
                logger.error(f"Download failed for Generator {name}.")
                continue
            
            model = tf.keras.models.load_model(
                model_file,
                custom_objects={'InstanceNormalization': InstanceNormalization},
                compile=False
            )
            
            logger.info(f"✓ Generator {name} LOADED!")
            generators[name] = model
            os.remove(model_file)
                
        except Exception as e:
            logger.error(f"Generator {name} failed: {str(e)}")
    
    return generators

generators = load_all_generators()

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

def preprocess_image(image_bytes, filename, conversion_type):
    """Preprocess grayscale image strictly to 64x64 for the models"""
    if filename.lower().endswith('.dcm'):
        dicom_data = pydicom.dcmread(io.BytesIO(image_bytes))
        img_array = dicom_data.pixel_array.astype(float)
        if img_array.max() > 0:
            img_array = (np.maximum(img_array, 0) / img_array.max()) * 255.0
        img_array = np.uint8(img_array)
    else:
        img = Image.open(io.BytesIO(image_bytes)).convert('L') # Must be Grayscale ('L')
        img_array = np.array(img, dtype=np.uint8)
        
    # Pre-processing contrast
    if conversion_type == 'ct_to_mri':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    else:
        clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(8, 8))
        
    img_array = clahe.apply(img_array)
    
    # Must be 64x64 for the AI
    img = Image.fromarray(img_array)
    img = img.resize((64, 64), Image.LANCZOS)
    
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def advanced_denoise_grayscale(img_array):
    """Advanced multi-stage denoising pipeline for grayscale medical images"""
    # Stage 1: Non-Local Means Denoising (Grayscale version)
    img_denoised = cv2.fastNlMeansDenoising(
        img_array,
        None,
        h=12,
        templateWindowSize=7,
        searchWindowSize=21
    )
    
    # Stage 2: Bilateral Filter (edge-preserving smoothing)
    img_bilateral = cv2.bilateralFilter(img_denoised, d=9, sigmaColor=75, sigmaSpace=75)
    
    # Stage 3: Morphological Operations (remove salt-and-pepper noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_opened = cv2.morphologyEx(img_bilateral, cv2.MORPH_OPEN, kernel)
    img_closed = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, kernel)
    
    # Stage 4: Gaussian Blur
    img_final = cv2.GaussianBlur(img_closed, (3, 3), 0.5)
    
    return img_final

def postprocess_image(prediction, conversion_type, target_size=(512, 512)):
    """Upscale and apply maximum denoising to the AI output"""
    # Strip batch/channel dimensions (Fixes the HTML <sup>0</sup> error)
    prediction = np.squeeze(prediction, axis=0)
    prediction = np.squeeze(prediction, axis=-1)
    
    # Convert back to 0-255 pixels
    img_array = ((prediction + 1.0) * 127.5).astype(np.uint8)
    
    # Upscale first
    img_upscaled = cv2.resize(img_array, target_size, interpolation=cv2.INTER_CUBIC)
    
    # Apply advanced denoising ONLY to MRI->CT (where the noise happens)
    if conversion_type == 'mri_to_ct':
        img_upscaled = advanced_denoise_grayscale(img_upscaled)
    
    # Gentle sharpening
    sharpening_kernel = np.array([
        [0, -0.2, 0],
        [-0.2, 1.8, -0.2],
        [0, -0.2, 0]
    ])
    img_sharpened = cv2.filter2D(img_upscaled, -1, sharpening_kernel)
    
    # Convert to PIL
    img_pil = Image.fromarray(img_sharpened, mode='L')
    
    # Subtle PIL Enhancements
    enhancer_sharpness = ImageEnhance.Sharpness(img_pil)
    img_pil = enhancer_sharpness.enhance(1.2)
    
    enhancer_contrast = ImageEnhance.Contrast(img_pil)
    img_pil = enhancer_contrast.enhance(1.1)
    
    # Final smoothing filter
    if conversion_type == 'mri_to_ct':
        img_pil = img_pil.filter(ImageFilter.SMOOTH)
        
    return img_pil

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'online',
        'version': 'v15.0-maximum-denoising-512x512',
        'models_loaded': len(generators) == 4,
        'generators': list(generators.keys())
    })

@app.route('/convert', methods=['POST'])
def convert_image():
    if len(generators) != 4:
        return jsonify({'error': 'Models not fully loaded.'}), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
        
    conversion_type = request.form.get('type', 'ct_to_mri')
    
    try:
        image_file = request.files['image']
        filename = image_file.filename
        image_bytes = image_file.read()
        
        # Preprocess
        input_tensor = preprocess_image(image_bytes, filename, conversion_type)
        
        # Determine generators
        predictions = {}
        if conversion_type == 'ct_to_mri':
            predictions['G'] = generators['G'](input_tensor, training=False)
            predictions['I'] = generators['I'](input_tensor, training=False)
        else:
            predictions['F'] = generators['F'](input_tensor, training=False)
            predictions['J'] = generators['J'](input_tensor, training=False)
        
        # Postprocess
        results = {}
        for gen_name, pred_tensor in predictions.items():
            output_img = postprocess_image(pred_tensor.numpy(), conversion_type)
            
            img_byte_arr = io.BytesIO()
            output_img.save(img_byte_arr, format='PNG', optimize=True)
            img_byte_arr.seek(0)
            
            img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
            results[f'image_{gen_name}'] = img_base64
            
        return jsonify(results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
