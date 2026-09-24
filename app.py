import io
import json
import os

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

APP_VERSION = "1.2"
MODEL_PATH = os.getenv("MODEL_PATH", "face_hybrid_int8.onnx")
META_PATH = os.getenv("META_PATH", "deployment_metadata.json")

try:
    with open(META_PATH, "r", encoding="utf-8") as f:
        META = json.load(f)
except Exception:
    META = {}

DEFAULT_THRESHOLD = float(
    os.getenv(
        "VERIFY_THRESHOLD",
        META.get("verification", {}).get(
            "recommended_deployment_threshold", 0.45
        ),
    )
)

app = FastAPI(title="LightFace Hybrid API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lightface-hybrid-ai.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ONNX Runtime CPU session: one thread is appropriate for Render Free.
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
session = ort.InferenceSession(
    MODEL_PATH,
    so,
    providers=["CPUExecutionProvider"],
)

# EXACT normalization used by the supplied training pipeline.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 112
FACE_MARGIN = 0.35
DETECT_MAX_SIDE = 640

# Lightweight offline detector; no additional model file is required.
FACE_DETECTOR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
if FACE_DETECTOR.empty():
    raise RuntimeError("OpenCV Haar face detector could not be loaded.")


def detect_face(rgb: np.ndarray):
    """Return the largest face box in original-image coordinates."""
    h, w = rgb.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))

    if scale < 1.0:
        small = cv2.resize(
            rgb,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = rgb

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    boxes = FACE_DETECTOR.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(boxes) == 0:
        return None, 0

    largest = max(boxes, key=lambda b: int(b[2]) * int(b[3]))
    x, y, bw, bh = largest
    return (
        float(x / scale),
        float(y / scale),
        float((x + bw) / scale),
        float((y + bh) / scale),
    ), len(boxes)


def crop_with_margin(rgb: np.ndarray, box):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    mx = bw * FACE_MARGIN
    my = bh * FACE_MARGIN

    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))

    if x2 <= x1 or y2 <= y1:
        raise HTTPException(422, "Detected face crop is invalid.")

    return rgb[y1:y2, x1:x2]


def prep(data: bytes) -> np.ndarray:
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    rgb = np.asarray(im)
    box, face_count = detect_face(rgb)

    if box is None:
        raise HTTPException(
            422,
            "No face detected. Use a clear photo with one visible face, "
            "good lighting, and the face reasonably large in the image.",
        )

    face = crop_with_margin(rgb, box)
    face = cv2.resize(
        face,
        (INPUT_SIZE, INPUT_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    # Match training exactly: RGB -> [0,1] -> ImageNet normalization -> CHW.
    x = face.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))[None]
    return np.ascontiguousarray(x, dtype=np.float32)


def embed(data: bytes) -> np.ndarray:
    z = session.run(
        ["embedding"],
        {"image": prep(data)},
    )[0][0].astype(np.float32)
    return z / (np.linalg.norm(z) + 1e-12)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": os.path.basename(MODEL_PATH),
        "runtime": "onnxruntime-cpu",
        "embedding_dim": 256,
        "input_size": [112, 112],
        "preprocessing": "face-detect -> margin-crop -> resize -> ImageNet-normalize",
        "face_detection": "opencv-haar-cascade",
        "default_threshold": DEFAULT_THRESHOLD,
    }


@app.post("/embedding")
async def embedding(file: UploadFile = File(...)):
    z = embed(await file.read())
    return {
        "embedding": z.tolist(),
        "dimension": int(z.shape[0]),
    }


@app.post("/verify")
async def verify(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: float | None = None,
):
    z1 = embed(await file1.read())
    z2 = embed(await file2.read())
    score = float(np.dot(z1, z2))
    t = DEFAULT_THRESHOLD if threshold is None else float(threshold)

    return {
        "similarity": score,
        "threshold": t,
        "verified": bool(score >= t),
        "preprocessing": "face-detect -> margin-crop -> resize -> ImageNet-normalize",
    }
