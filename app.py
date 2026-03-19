import os
import io
import base64
import datetime
import gdown
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
import pydicom.uid
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

        # --- THE MISSING FIX: PERCENTILE CLIPPING ---
        # This cuts out scanner artifacts and forces the DICOM 
        # to have the exact same clean contrast as your PNGs!
        p_low, p_high = np.percentile(img, (1.0, 99.0))
        img = np.clip(img, p_low, p_high)

        if p_high - p_low > 0:
            img = (img - p_low) / (p_high - p_low) * 255.0
        else:
            img = np.zeros_like(img)
        
        img = img.astype(np.uint8)
        
        # Bake the 8-bit PNG compression artifacts into the array
        _, encoded_png = cv2.imencode('.png', img)
        img = cv2.imdecode(encoded_png, cv2.IMREAD_GRAYSCALE)

    else:
        # --- REGULAR PNG/JPG UPLOADS (These work perfectly!) ---
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read image data.")

    # --- 2. SQUASH TO 64x64 ---
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # --- 3. NORMALIZE TO [-1, 1] & FORMAT FOR KERAS ---
    img = (img.astype(np.float32) / 127.5) - 1.0
    img = img.reshape((1, target_h, target_w, 1))
    
    return img


# ==========================================
# === POST-PROCESSING & FILE GENERATION ===
# ==========================================
def get_sharpened_grayscale(tensor):
    """Takes the raw AI tensor and applies the Band-Aid upscaling to a crisp 512x512."""
    if hasattr(tensor, 'numpy'):
        img = tensor[0].numpy()
    else:
        img = tensor[0]
        
    # Denormalize
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Force Grayscale
    if len(img.shape) == 3 and img.shape[2] == 1:
        img = np.squeeze(img, axis=-1)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Upscale and Sharpen
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    gaussian_blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    img = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)
    img = cv2.bilateralFilter(img, d=5, sigmaColor=25, sigmaSpace=25)
    return img

def convert_to_png_base64(gray_img):
    """Converts the grayscale array to a PNG for the website UI."""
    bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode('.png', bgr_img)
    return base64.b64encode(buffer).decode('utf-8')

def convert_to_dicom_base64(gray_img, modality):
    """Packages the grayscale array into a strict, valid DICOM file."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.2') # CT/MRI Storage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    # Inject standard medical metadata
    ds.PatientName = "AI^Generated^Patient"
    ds.PatientID = "ITRANS-V16"
    ds.Modality = modality.upper()
    ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SecondaryCaptureDeviceManufacturer = "I-Translation AI"

    # Image specs (8-bit grayscale)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 7
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.Rows, ds.Columns = gray_img.shape
    ds.PixelData = gray_img.tobytes()

    # Save to memory and encode
    with io.BytesIO() as buffer:
        ds.save_as(buffer, write_like_original=False)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


# ==========================================
# FLASK ROUTING
# ==========================================
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

        # Get the AI translation
        result_tensor = model(input_tensor, training=False)

        # Process the image once
        sharpened_image = get_sharpened_grayscale(result_tensor)
        
        # Generate both file formats!
        modality_string = "MR" if model_key == 'G' else "CT"
        
        return jsonify({
            f'image_{model_key}': convert_to_png_base64(sharpened_image),
            f'dicom_{model_key}': convert_to_dicom_base64(sharpened_image, modality_string)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
