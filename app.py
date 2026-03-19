import os
import io
import base64
import datetime
import zipfile
import glob
import gdown
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
import pydicom.uid
import tensorflow as tf
import numpy as np
import cv2
import imageio.v2 as imageio 
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
# HIGH-RES 990 EPOCH CHAMPION MODELS (ZIP)
# ==========================================
ZIP_FILE_ID = '113RSDAYjY-dNw9D9bd1tiRAGkzzDO8HL'
generators = {}

def load_models():
    print("======================================================================")
    print("LOADING 990-EPOCH GENERATORS (F & G) FROM ZIP")
    print("======================================================================")
    
    zip_path = "/tmp/models.zip"
    extract_dir = "/tmp/extracted_models"
    
    # 1. Download and Extract if we haven't already
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        print("Downloading massive ZIP file... (This may take a minute)")
        # Using id= bypasses the Google Drive virus warning!
        gdown.download(id=ZIP_FILE_ID, output=zip_path, quiet=False)
        
        print("Extracting ZIP file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print("Extraction complete!")

    # 2. Find the F and G models inside the extracted folder
    model_paths = glob.glob(f"{extract_dir}/**/*.h5", recursive=True)
    
    path_F = None
    path_G = None
    
    for path in model_paths:
        filename = os.path.basename(path).lower()
        if 'f' in filename:
            path_F = path
        elif 'g' in filename:
            path_G = path

    if not path_F or not path_G:
        raise ValueError(f"Could not find both F and G models in the ZIP! Found: {model_paths}")

    # 3. Load them into memory
    print(f"Loading Generator F from: {path_F}")
    generators['F'] = tf.keras.models.load_model(
        path_F, compile=False, custom_objects={'InstanceNormalization': InstanceNormalization}
    )
    print("✓ Generator F LOADED!")

    print(f"Loading Generator G from: {path_G}")
    generators['G'] = tf.keras.models.load_model(
        path_G, compile=False, custom_objects={'InstanceNormalization': InstanceNormalization}
    )
    print("✓ Generator G LOADED!")

load_models()


# ==========================================
# === TRUE HIGH-DEFINITION TRANSLATOR ===
# ==========================================
def preprocess_image(image_bytes, filename, expected_shape):
    # 1. OBEY THE NEW HIGH-RES SHAPE (256x256)
    target_h = 256
    target_w = 256

    if filename.endswith('.dcm'):
        # --- THE PNG SIMULATION ---
        dicom = pydicom.dcmread(io.BytesIO(image_bytes))
        img = dicom.pixel_array.astype(np.float32)

        if len(img.shape) == 3 and img.shape[2] not in [1, 3, 4]:
            img = img[img.shape[0] // 2]
        elif len(img.shape) == 4:
            img = img[0, img.shape[1] // 2]

        if len(img.shape) == 3:
            img = np.mean(img, axis=-1)

        # --- PERCENTILE CLIPPING FIX ---
        p_low, p_high = np.percentile(img, (1.0, 99.0))
        img = np.clip(img, p_low, p_high)

        if p_high - p_low > 0:
            img = (img - p_low) / (p_high - p_low) * 255.0
        else:
            img = np.zeros_like(img)
        
        img = img.astype(np.uint8)
        
        _, encoded_png = cv2.imencode('.png', img)
        img = cv2.imdecode(encoded_png, cv2.IMREAD_GRAYSCALE)

    else:
        # --- REGULAR PNG/JPG UPLOADS ---
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not read image data.")

    # --- 2. RESIZE TO 256x256 ---
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # --- 3. NORMALIZE TO [-1, 1] & FORMAT FOR KERAS ---
    img = (img.astype(np.float32) / 127.5) - 1.0
    img = img.reshape((1, target_h, target_w, 1))
    
    return img


# ==========================================
# === POST-PROCESSING & FILE GENERATION ===
# ==========================================
def postprocess_tensor(tensor):
    """Cleanly processes the true 256x256 AI output without artificial blurring."""
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

    # Soft resize to 512x512 strictly for UI viewing consistency (No more fake sharpening!)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)
    
    return img

def convert_to_png_base64(gray_img):
    bgr_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode('.png', bgr_img)
    return base64.b64encode(buffer).decode('utf-8')

def convert_to_dicom_base64(gray_img, modality):
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.2') 
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    ds.PatientName = "AI^Generated^Patient"
    ds.PatientID = "ITRANS-990-EPOCH"
    ds.Modality = modality.upper()
    ds.StudyDate = datetime.datetime.now().strftime('%Y%m%d')
    ds.StudyTime = datetime.datetime.now().strftime('%H%M%S')
    ds.StudyInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SecondaryCaptureDeviceManufacturer = "I-Translation AI"

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 7
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.Rows, ds.Columns = gray_img.shape
    ds.PixelData = gray_img.tobytes()

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

        result_tensor = model(input_tensor, training=False)

        final_image = postprocess_tensor(result_tensor)
        modality_string = "MR" if model_key == 'G' else "CT"
        
        return jsonify({
            f'image_{model_key}': convert_to_png_base64(final_image),
            f'dicom_{model_key}': convert_to_dicom_base64(final_image, modality_string)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
