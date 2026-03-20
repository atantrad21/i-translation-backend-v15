def convert_to_dicom_base64(gray_img, modality):
    import numpy as np
    import cv2
    import io
    import base64
    import datetime
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    
    # 1. Convert whatever the AI spit out into a rigid Numpy array
    img_array = np.array(gray_img)

    # 2. Resize to Standard Medical Resolution (512x512)
    # This prevents the "black box" by stretching the AI output to fit a standard viewer canvas
    if img_array.shape != (512, 512):
        img_array = cv2.resize(img_array, (512, 512), interpolation=cv2.INTER_CUBIC)

    # 3. Normalize the contrast mathematically (forces values safely between 0 and 255)
    img_min = img_array.min()
    img_max = img_array.max()
    if img_max > img_min:
        img_array = (img_array - img_min) / (img_max - img_min) * 255.0
    
    # 4. CRITICAL FIX: Cast to 8-bit unsigned integer to match 'BitsAllocated = 8'
    img_array = img_array.astype(np.uint8)

    # --- Start standard DICOM metadata generation ---
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.2') 
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\0" * 128)
    
    ds.PatientName = "AI^Generated^Patient"
    ds.PatientID = "ITRANS-STABLE-EPOCH"
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
    
    # Use the exact dimensions of our safely resized array (512x512)
    ds.Rows, ds.Columns = img_array.shape
    
    # Pack the clean 8-bit image data into the DICOM structure
    ds.PixelData = img_array.tobytes()

    # --- Base64 Conversion for Frontend Download ---
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    # Save to a temporary memory buffer instead of writing a physical file to the server
    buffer = io.BytesIO()
    pydicom.filewriter.dcmwrite(buffer, ds, write_like_original=False)
    
    # Encode the raw DICOM bytes into base64 text so the web app can receive it
    dicom_bytes = buffer.getvalue()
    dicom_base64 = base64.b64encode(dicom_bytes).decode('utf-8')
    
    return dicom_base64
