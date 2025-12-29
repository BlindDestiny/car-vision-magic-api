import fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from typing import Dict, Any, List
import logging
import pycocotools.mask as mask_utils
import tempfile
import os
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import threading
from PIL import Image
from io import BytesIO
import httpx
from zeep import AsyncClient
from zeep.transports import AsyncTransport
import requests
import base64
# from pydantic import BaseModel
# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO)

# ----------------------
# App & device
# ----------------------
app = FastAPI(title="YOLO Video + Image Inference API")

# Allow all origins for development (change for production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {DEVICE}")

# ----------------------
# Models
# ----------------------
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "parts": {"path": "models/parts_seg.pt", "type": "segmentation", "imgsz": (640, 480), "model": None},
    "damage": {"path": "models/damage_seg.pt", "type": "segmentation", "imgsz": (1280, 960), "model": None},
    "license": {"path": "models/license_plate.pt", "type": "detection", "imgsz": 640, "model": None},
}

CONF_THRESH = 0.15
IOU_THRESH = 0.30
FRAME_SKIP = 2  # skip frames for speed
interpreter_lock = threading.Lock()
CLASS_NAMES = ["frontLeft", "frontRight", "rearLeft", "rearRight"]
OPENALPR_URL = "https://api.openalpr.com/v2/recognize_bytes"
SECRET_KEY = "sk_f71502be1175ed8ccd71bd7b"
# Settings
TFLITE_MODEL_NAME = "models/car_model_quant_dynamic_batch.tflite"
EUROTAX_WSDL = "http://services.eurotax.pt/MatVinPTWS/MatVinPT.asmx?wsdl"
EUROTAX_DATA_WS_URL = "http://services.eurotax.pt/VINGREENAPI/VGWSAPI.asmx?wsdl"
USERKEY = "4cedce7a-ef44-461c-93f6-b94deac23531"
USERNAME = "itcTest"
PASSWORD = "itcPass"
COMPUTER_KEY = "4CB4-25D1-B2F7-6253-EC9B-E053-30B4-E759"

print("✅ Loading TFLite interpreter...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_NAME)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def regular_error_response():
    return JSONResponse({"ok": False, "message": "Plate recognition failed"}, status_code=500)

def run_inference(input_data):
    with interpreter_lock:
        interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = np.copy(interpreter.get_tensor(output_details[0]['index']))
    return output_data
# ----------------------
# Load models on startup
# ----------------------
@app.on_event("startup")
async def load_models():
    for name, cfg in MODEL_CONFIG.items():
        logging.info(f"Loading {name}")
        cfg['model'] = YOLO(cfg['path'])
        logging.info(f"{name} loaded")

# ----------------------
# RLE compression for masks
# ----------------------
def mask_to_rle(mask: np.ndarray) -> str:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')  # convert bytes -> str
    return rle

# ----------------------
# Letterbox
# ----------------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # (h, w)
    h0, w0 = shape

    r = min(new_shape[0] / h0, new_shape[1] / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )

    return img_padded, r, left, top

def results_to_dict_rle(
    results,
    model_type: str,          # "parts", "damage", or "license"
    orig_shape=None,          # original image or ROI shape (h, w)
    gain=1.0,
    pad_x=0,
    pad_y=0,
    roi_offset=(0, 0),
    imgsz=None                # model input size
):
    """
    Convert YOLO results to dicts with optional RLE masks.
    Undo letterbox and ROI offsets. Preserves tiny objects.
    Adds model input size for frontend scaling.
    """
    import math
    detections = []
    ox, oy = roi_offset
    h0, w0 = orig_shape if orig_shape else (None, None)

    # Determine model input size
    if imgsz is None:
        model_w, model_h = w0, h0
    elif isinstance(imgsz, int):
        model_w = model_h = imgsz
    else:
        model_w, model_h = imgsz

    for r in results:
        if not hasattr(r, "boxes") or r.boxes is None:
            continue

        for i, box in enumerate(r.boxes):
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = r.names[cls]

            # Undo letterbox & ROI offsets
            if orig_shape:
                x1 = (x1 - pad_x) / gain + ox
                y1 = (y1 - pad_y) / gain + oy
                x2 = (x2 - pad_x) / gain + ox
                y2 = (y2 - pad_y) / gain + oy

                # Clip to image
                x1 = float(max(0, min(x1, w0)))
                y1 = float(max(0, min(y1, h0)))
                x2 = float(max(0, min(x2, w0)))
                y2 = float(max(0, min(y2, h0)))

            det = {
                "label": label,
                "confidence": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "type": model_type,
                "model_width": model_w,
                "model_height": model_h,
            }

            # Add mask if available
            if hasattr(r, "masks") and r.masks is not None:
                mask = r.masks.data[i].cpu().numpy().astype(np.uint8)

                # Resize mask to original ROI size (avoid collapsing tiny masks)
                if orig_shape:
                    new_w = max(1, math.ceil(w0))
                    new_h = max(1, math.ceil(h0))
                    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

                # Apply ROI offset
                if roi_offset != (0, 0):
                    full_mask = np.zeros((h0 + oy, w0 + ox), dtype=np.uint8)
                    full_mask[oy:oy+h0, ox:ox+w0] = mask
                    mask = full_mask

                det["mask_rle"] = mask_to_rle(mask)
                det["mask_width"] = mask.shape[1]
                det["mask_height"] = mask.shape[0]

            detections.append(det)

    return detections



# ----------------------
# ROI crop: crop image to union of bounding boxes
# ----------------------
def crop_to_roi(img: np.ndarray, boxes: List[Dict[str, float]]):
    if not boxes:
        return img, (0, 0)
    x1 = min(b['x1'] for b in boxes)
    y1 = min(b['y1'] for b in boxes)
    x2 = max(b['x2'] for b in boxes)
    y2 = max(b['y2'] for b in boxes)
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
    cropped = img[y1:y2, x1:x2]
    return cropped, (x1, y1)

# ----------------------
# Video frame generator
# ----------------------
def video_stream_generator(video_path: str, model_name: str):
    cfg = MODEL_CONFIG[model_name]
    model = cfg['model']
    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    license_boxes = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue

        # Optional ROI cropping for damage/parts
        if model_name in ['parts', 'damage'] and license_boxes:
            roi_img, (ox, oy) = crop_to_roi(frame, license_boxes)
        else:
            roi_img = frame
            ox, oy = 0, 0

        imgsz = cfg['imgsz']
        if isinstance(imgsz, int):
            imgsz = (imgsz, imgsz)
        h0, w0 = roi_img.shape[:2]

        roi_img_lb, gain, pad_x, pad_y = letterbox(roi_img, imgsz)

        results = model.predict(
            roi_img_lb,
            imgsz=imgsz,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            device=DEVICE,
            stream=False,
        )

        detections = results_to_dict_rle(
            results,
            cfg["type"],
            orig_shape=(h0, w0),
            gain=gain,
            pad_x=pad_x,
            pad_y=pad_y,
        )

        # shift back ROI offset
        for det in detections:
            det["x1"] += ox
            det["x2"] += ox
            det["y1"] += oy
            det["y2"] += oy


        # Save license boxes for next frame ROI
        if model_name == 'license':
            license_boxes = detections

        yield f"data: {detections}\n\n"
        frame_id += 1
    cap.release()
    os.remove(video_path)

# ----------------------
# Video inference endpoint
# ----------------------
@app.post("/infer/video/{model_name}")
async def infer_video(model_name: str, file: UploadFile = File(...)):
    if model_name not in MODEL_CONFIG:
        return JSONResponse({"error": "Invalid model name"}, status_code=400)
    
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_file.write(await file.read())
    tmp_file.close()

    return StreamingResponse(video_stream_generator(tmp_file.name, model_name),
                             media_type="text/event-stream")

# ----------------------
# Image inference endpoint with masks + ROI optional
# ----------------------
@app.post("/infer/image/{model_name}")
async def infer_image(model_name: str, file: UploadFile = File(...)):
    if model_name not in MODEL_CONFIG:
        return JSONResponse({"error": "Invalid model name"}, status_code=400)

    cfg = MODEL_CONFIG[model_name]
    model = cfg["model"]

    imgsz = cfg["imgsz"]
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)

    bytes_data = await file.read()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    img0 = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    h0, w0 = img0.shape[:2]

    img, gain, pad_x, pad_y = letterbox(img0, imgsz)

    results = model.predict(
        img,
        imgsz=imgsz,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        device=DEVICE,
        stream=False,
    )

    detections = results_to_dict_rle(
        results,
        cfg["type"],
        orig_shape=(h0, w0),
        gain=gain,
        pad_x=pad_x,
        pad_y=pad_y,
    )

    return {"detections": detections}


@app.post("/predict")
async def predict_single(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    try:
        # read image bytes
        bytes_data = await file.read()
        img = Image.open(BytesIO(bytes_data)).convert("RGB")
        img = img.resize((299, 299))

        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # inference
        predictions = run_inference(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return {
            "prediction": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    

@app.post("/plate_recognize")
async def plate_recognition(file: UploadFile = File(...)):
   try:
        # Read file bytes
        image_bytes = await file.read()

        # Encode to base64
        img_base64 = base64.b64encode(image_bytes)

        # Send to OpenALPR
        url = f"https://api.openalpr.com/v2/recognize_bytes?recognize_vehicle=1&country=eu&secret_key={SECRET_KEY}"
        response = requests.post(url, data=img_base64)
        result = response.json().get("results")
        
        # Extract first plate if available
        
        if len(result) > 0:
            return await decode_plate(result[0].get("plate"))
        else:
            return {"plate": None, "raw": result, "message": "No number plate found"}

   except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/decode_plate")
async def decode_plate(plate: str):
    print(plate)
    async with httpx.AsyncClient() as session:
        transport = AsyncTransport(client=session)
        client = AsyncClient(wsdl=EUROTAX_WSDL, transport=transport)

        response = await client.service.GetVIN(
            userKey=USERKEY,
            licensePlate=plate
        )
        
        # response is usually a Zeep object or dict-like
        vin_result = response.GetVINResult if hasattr(response, "GetVINResult") else response
       
        client = AsyncClient(wsdl=EUROTAX_DATA_WS_URL, transport=transport)

        # Call the SOAP method
        responseData = await client.service.GetVehicleFromVIN(
            userName=USERNAME,
            userPassword=PASSWORD,
            computerUniqueKey=COMPUTER_KEY,
            vin=vin_result
        )
        print(responseData)
        return {
            'vin': vin_result ,
            'plate': plate,
            'brand': responseData['makeDescription'],
            'model': responseData['modelDescription'],
            'version': responseData['typeDescription'],
            'bodyType': responseData['bodyType'],
            'numberOfDoors': responseData['numberDoors'],
            'engineKW': responseData['engineKW'],
            'enginePS': responseData['enginePS'],
            'fuelType': responseData['fuelType'],
            'gearType': responseData['gearType'],
            'engineSize': responseData['engineSize'],
            'numberOfGears': responseData['numberGears'],
            'color': responseData['colourCode'],
            'registrationDate': responseData['registrationDate']
        }

# ----------------------
# Health check
# ----------------------
@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE, "models_loaded": {k: v['model'] is not None for k,v in MODEL_CONFIG.items()}}
