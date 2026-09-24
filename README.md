# LightFace MobileFaceNet API

Lightweight face verification API for Render Free.

## Model

Recognition: InsightFace `w600k_mbf.onnx` — MobileFaceNet (MBF), WebFace600K, 512-D embedding.

Model SHA256:
`9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f`

Detector: OpenCV YuNet `face_detection_yunet_2023mar.onnx`.

Detector SHA256:
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`

## Preprocessing

1. Decode image as RGB.
2. Detect exactly one face with YuNet.
3. Use YuNet's five landmarks.
4. Align to the canonical ArcFace 112x112 five-point template using a similarity transform.
5. Convert aligned BGR image through the InsightFace ArcFace convention:
   `(pixel - 127.5) / 127.5` with RGB channel order and NCHW float32 tensor.
6. Run ONNX Runtime CPU inference.
7. L2-normalize the 512-D embedding.
8. Compare embeddings using cosine similarity (dot product after normalization).

## API

GET `/health`

POST `/embedding` with multipart field `file`

POST `/verify` with multipart fields `file1` and `file2`

Optional threshold override:
`POST /verify?threshold=0.40`

## Render

The build automatically downloads the two model files and verifies their SHA256 hashes. No PyTorch or InsightFace Python package is installed at runtime.

## Important threshold note

`0.40` is a starting deployment threshold, not a universal biometric-security threshold. Calibrate it on your intended verification population and operating point before treating the result as an identity/security decision.

## License note

The InsightFace model zoo states that its pretrained models are available for non-commercial research purposes. Review the model license and obtain appropriate permission/licensing before commercial use.
