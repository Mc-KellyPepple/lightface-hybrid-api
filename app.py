import io
import json
import os
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

app = FastAPI(title="LightFace Hybrid API", version="1.1")

MODEL_PATH = os.getenv("MODEL_PATH", "face_hybrid_int8.onnx")
META_PATH = os.getenv("META_PATH", "deployment_metadata.json")

try:
    with open(META_PATH, "r") as f:
        META = json.load(f)
except Exception:
    META = {}

DEFAULT_THRESHOLD = float(
    os.getenv(
        "VERIFY_THRESHOLD",
        META.get("verification", {}).get("recommended_deployment_threshold", 0.45),
    )
)

_so = ort.SessionOptions()
_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
_so.intra_op_num_threads = 1
_so.inter_op_num_threads = 1
session = ort.InferenceSession(MODEL_PATH, _so, providers=["CPUExecutionProvider"])

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ------------------------------------------------------------------
# Face detection + crop-with-margin.
#
# THIS IS THE FIX for "different people score ~95%". The previous
# version of this file (see the old README's own warning: "the API
# expects an already-cropped face; it does not perform face detection
# or alignment") just resized whatever raw photo came in straight to
# 112x112. The model was trained on tightly-cropped, roughly centered
# 112x112 face crops (standard ArcFace/MobileFaceNet-style
# preprocessing) — feeding it a normal photo or webcam capture, where
# the face is a small part of a bigger frame, off-center, at some
# random scale, puts it completely outside what it learned. In that
# regime the embedding stops encoding facial identity and starts
# encoding generic scene layout (centered face-blob, similar
# background/lighting), so two different people photographed the same
# way come out looking "similar" to the model.
#
# Fix: detect the face first, crop to it with a margin (so the crop
# looks like the training data — forehead to chin, not a tight box —
# instead of a raw uncropped photo), THEN resize to 112x112. Uses
# OpenCV's bundled Haar cascade — no extra model file to download or
# host, works fully offline, CPU-only, adds no new heavy dependency
# beyond opencv-python-headless (already common on these deployments).
#
# This is a real face detector, not a full 5-point-landmark aligner
# (eyes/nose/mouth-corner similarity-transform alignment, the way
# ArcFace training pipelines usually align faces) — that would need a
# landmark model this deployment doesn't have. A well-margined
# bounding-box crop is a large, well-documented improvement over no
# detection at all, and should resolve the reported symptom; if you
# still see inflated scores on some images after this, swapping in a
# landmark-based aligner (e.g. OpenCV's DNN face detector + a small
# 5-point landmark model) is the next step up in robustness.
# ------------------------------------------------------------------
_face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

FACE_MARGIN = 0.35  # fraction of the detected box's size, added on every side
DETECT_MAX_SIDE = 640  # downscale very large photos before detection, for speed


def _detect_face_box(rgb: np.ndarray):
    """Return (x1, y1, x2, y2) of the largest detected face, in the
    original image's coordinates, or None if no face was found."""
    h, w = rgb.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
    small = (
        cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else rgb
    )
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
    boxes = _face_detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(boxes) == 0:
        return None
    # If more than one face is in frame, assume the largest is the
    # intended subject (closest to the camera) rather than erroring out.
    bx, by, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
    return bx / scale, by / scale, (bx + bw) / scale, (by + bh) / scale


def _crop_with_margin(rgb: np.ndarray, box) -> np.ndarray:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * FACE_MARGIN, (y2 - y1) * FACE_MARGIN
    x1 = max(0, int(x1 - mx))
    y1 = max(0, int(y1 - my))
    x2 = min(w, int(x2 + mx))
    y2 = min(h, int(y2 + my))
    return rgb[y1:y2, x1:x2]


def prep(data: bytes) -> np.ndarray:
    try:
        im = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    rgb = np.array(im)
    box = _detect_face_box(rgb)
    if box is None:
        raise HTTPException(
            422,
            "No face detected in this image. Use a clear, front-facing "
            "photo with a single visible face and reasonable lighting.",
        )

    face = _crop_with_margin(rgb, box)
    face = cv2.resize(face, (112, 112), interpolation=cv2.INTER_LINEAR)

    x = face.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.transpose(x, (2, 0, 1))[None].astype(np.float32)


def embed(data: bytes) -> np.ndarray:
    z = session.run(["embedding"], {"image": prep(data)})[0][0].astype(np.float32)
    return z / (np.linalg.norm(z) + 1e-12)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "runtime": "onnxruntime-cpu",
        "embedding_dim": 256,
        "face_detection": "opencv-haar-cascade",
        "default_threshold": DEFAULT_THRESHOLD,
    }


@app.post("/embedding")
async def embedding(file: UploadFile = File(...)):
    z = embed(await file.read())
    return {"embedding": z.tolist(), "dimension": int(z.shape[0])}


@app.post("/verify")
async def verify(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: Optional[float] = None,
):
    z1 = embed(await file1.read())
    z2 = embed(await file2.read())
    score = float(np.dot(z1, z2))
    t = DEFAULT_THRESHOLD if threshold is None else float(threshold)
    return {"similarity": score, "threshold": t, "verified": bool(score >= t)}
