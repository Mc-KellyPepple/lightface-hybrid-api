import hashlib
import os
import urllib.request

MODELS = {
    "w600k_mbf.onnx": (
        "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_s/w600k_mbf.onnx?download=true",
        "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
    ),
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
}

for filename, (url, expected) in MODELS.items():
    if os.path.exists(filename) and os.path.getsize(filename) > 1024:
        with open(filename, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got == expected:
            print(f"OK: {filename}")
            continue
        os.remove(filename)

    print(f"Downloading {filename}")
    urllib.request.urlretrieve(url, filename)
    with open(filename, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    if got != expected:
        raise RuntimeError(f"SHA256 mismatch for {filename}: {got}")
    print(f"OK: {filename}")
