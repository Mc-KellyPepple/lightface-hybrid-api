import io
import os
import json
import numpy as np
import onnxruntime as ort

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="LightFace Hybrid API",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

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


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    "face_hybrid_int8.onnx"
)

META_PATH = os.getenv(
    "META_PATH",
    "deployment_metadata.json"
)


# ============================================================
# METADATA
# ============================================================

try:
    with open(META_PATH, "r") as f:
        META = json.load(f)
except Exception:
    META = {}


DEFAULT_THRESHOLD = float(
    os.getenv(
        "VERIFY_THRESHOLD",
        META.get(
            "verification",
            {}
        ).get(
            "recommended_deployment_threshold",
            0.45
        )
    )
)


# ============================================================
# ONNX RUNTIME
# ============================================================

so = ort.SessionOptions()

so.graph_optimization_level = (
    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
)

so.intra_op_num_threads = 1
so.inter_op_num_threads = 1

session = ort.InferenceSession(
    MODEL_PATH,
    so,
    providers=["CPUExecutionProvider"]
)


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
)


def prep(data):
    """
    Convert uploaded image into the
    model's expected NCHW float32 format.
    """

    try:
        im = (
            Image
            .open(io.BytesIO(data))
            .convert("RGB")
            .resize((112, 112))
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {e}"
        )

    x = (
        np.asarray(
            im,
            dtype=np.float32
        ) / 255.0
    )

    x = (x - MEAN) / STD

    return np.transpose(
        x,
        (2, 0, 1)
    )[None]


# ============================================================
# EMBEDDING
# ============================================================

def embed(data):

    try:

        z = session.run(
            ["embedding"],
            {
                "image": prep(data)
            }
        )[0][0].astype(np.float32)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {e}"
        )

    # L2 normalization
    z = z / (
        np.linalg.norm(z) + 1e-12
    )

    return z


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model": os.path.basename(MODEL_PATH),
        "runtime": "onnxruntime-cpu",
        "embedding_dim": 256
    }


# ============================================================
# EMBEDDING ENDPOINT
# ============================================================

@app.post("/embedding")
async def embedding(
    file: UploadFile = File(...)
):

    data = await file.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Empty image file."
        )

    z = embed(data)

    return {
        "embedding": z.tolist(),
        "dimension": int(z.shape[0])
    }


# ============================================================
# FACE VERIFICATION
# ============================================================

@app.post("/verify")
async def verify(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    threshold: float | None = None
):

    data1 = await file1.read()
    data2 = await file2.read()

    if not data1:
        raise HTTPException(
            status_code=400,
            detail="Face 1 image is empty."
        )

    if not data2:
        raise HTTPException(
            status_code=400,
            detail="Face 2 image is empty."
        )

    z1 = embed(data1)
    z2 = embed(data2)

    # Cosine similarity because embeddings
    # are L2 normalized.
    score = float(
        np.dot(z1, z2)
    )

    t = (
        DEFAULT_THRESHOLD
        if threshold is None
        else float(threshold)
    )

    return {
        "similarity": score,
        "threshold": t,
        "verified": bool(score >= t)
    }
