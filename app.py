import io
import os
import hashlib
import urllib.request

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

APP_VERSION = "2.0"
REC_MODEL = os.getenv("REC_MODEL", "w600k_mbf.onnx")
DET_MODEL = os.getenv("DET_MODEL", "face_detection_yunet_2023mar.onnx")
REC_URL = "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx?download=true"
DET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
REC_SHA256 = "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f"
DET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def ensure_model(path, url, expected_sha):
    if os.path.exists(path) and os.path.getsize(path) > 1024:
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        if digest == expected_sha:
            return
        os.remove(path)
    print(f"Downloading {path} ...")
    urllib.request.urlretrieve(url, path)
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != expected_sha:
        raise RuntimeError(f"SHA256 mismatch for {path}: {digest}")


ensure_model(REC_MODEL, REC_URL, REC_SHA256)
ensure_model(DET_MODEL, DET_URL, DET_SHA256)

app = FastAPI(title="LightFace MobileFaceNet API", version=APP_VERSION)
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

# Recognition: InsightFace MobileFaceNet / ArcFace ONNX.
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
recognizer = ort.InferenceSession(
    REC_MODEL, so, providers=["CPUExecutionProvider"]
)
REC_INPUT = recognizer.get_inputs()[0].name
REC_OUTPUTS = [x.name for x in recognizer.get_outputs()]

# YuNet: tiny face detector returning box + 5 landmarks.
detector = cv2.FaceDetectorYN.create(
    DET_MODEL,
    "",
    (320, 320),
    0.70,
    0.30,
    5000,
    cv2.dnn.DNN_BACKEND_OPENCV,
    cv2.dnn.DNN_TARGET_CPU,
)

# Canonical ArcFace 112x112 five-point template.
ARC_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

INPUT_SIZE = 112
DEFAULT_THRESHOLD = float(os.getenv("VERIFY_THRESHOLD", "0.40"))


def decode_image(data: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return np.asarray(image)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")


def align_face(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # YuNet needs the actual image size for its fixed 2023 model.
    detector.setInputSize((w, h))
    _, faces = detector.detect(bgr)

    if faces is None or len(faces) == 0:
        raise HTTPException(
            422,
            "No face detected. Use a clear, front-facing image with one visible face.",
        )

    # Reject ambiguous images rather than silently comparing the wrong person.
    valid = [f for f in faces if float(f[14]) >= 0.70]
    if len(valid) != 1:
        raise HTTPException(
            422,
            f"Expected exactly one clear face, but detected {len(valid)}. "
            "Use an image containing one person.",
        )

    face = valid[0]
    landmarks = np.array(
        [
            [face[4], face[5]],
            [face[6], face[7]],
            [face[8], face[9]],
            [face[10], face[11]],
            [face[12], face[13]],
        ],
        dtype=np.float32,
    )

    # ArcFace-style similarity transform: detected 5 points -> canonical 112x112.
    M, _ = cv2.estimateAffinePartial2D(
        landmarks,
        ARC_TEMPLATE,
        method=cv2.LMEDS,
    )
    if M is None:
        raise HTTPException(422, "Could not align the detected face.")

    aligned = cv2.warpAffine(
        bgr,
        M,
        (INPUT_SIZE, INPUT_SIZE),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned


def preprocess(aligned_bgr: np.ndarray) -> np.ndarray:
    # Exact InsightFace ArcFace ONNX convention:
    # BGR image -> swapRB -> (pixel - 127.5) / 127.5 -> NCHW float32.
    blob = cv2.dnn.blobFromImage(
        aligned_bgr,
        scalefactor=1.0 / 127.5,
        size=(112, 112),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    )
    return np.ascontiguousarray(blob, dtype=np.float32)


def embed(data: bytes) -> np.ndarray:
    rgb = decode_image(data)
    aligned = align_face(rgb)
    x = preprocess(aligned)
    out = recognizer.run(REC_OUTPUTS, {REC_INPUT: x})[0]
    z = np.asarray(out[0], dtype=np.float32).reshape(-1)
    return z / (np.linalg.norm(z) + 1e-12)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "InsightFace MobileFaceNet / w600k_mbf.onnx",
        "runtime": "onnxruntime-cpu",
        "embedding_dim": 512,
        "input_size": [112, 112],
        "detector": "OpenCV YuNet",
        "alignment": "ArcFace 5-point similarity transform",
        "preprocessing": "(pixel - 127.5) / 127.5, RGB CHW",
        "threshold": DEFAULT_THRESHOLD,
    }


@app.post("/embedding")
async def embedding(file: UploadFile = File(...)):
    z = embed(await file.read())
    return {"embedding": z.tolist(), "dimension": int(z.shape[0])}


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
        "model": "w600k_mbf.onnx",
        "embedding_dim": 512,
        "detector": "YuNet",
        "alignment": "ArcFace 5-point",
    }
