# LightFace Hybrid — Render Free Deployment

## Model
MobileFaceNet + MicroCNN + ECA + bounded gated fusion.

- Input: 112x112 RGB
- Embedding: 256-D L2-normalized vector
- Deployment: ONNX Runtime CPU + INT8 dynamic quantization
- PyTorch is not required on Render

## Production preprocessing
The API now prepares each uploaded image before verification:

1. Detect the largest visible face with OpenCV Haar cascade.
2. Crop the detected face with a 35% margin.
3. Resize the face to 112x112.
4. Apply the exact ImageNet normalization used by the supplied training pipeline.
5. Convert RGB HWC to NCHW for ONNX Runtime.
6. Generate the normalized 256-D embedding.
7. Compare the two embeddings with cosine similarity.

The previous deployment passed the entire uploaded photograph directly to the 112x112 model. That was inconsistent with the aligned/cropped face images used during training and could produce unreliable similarities on ordinary photographs.

## API

- `GET /health`
- `POST /embedding` with multipart field `file`
- `POST /verify` with multipart fields `file1` and `file2`

The API includes CORS support for the LightFace Vercel frontend.

## Important
The current calibrated threshold in `deployment_metadata.json` is preserved from the training experiment. The threshold can also be overridden through the Render `VERIFY_THRESHOLD` environment variable or the `/verify` query parameter.

For production/security calibration, use a threshold selected for the desired FAR rather than automatically assuming the accuracy-maximizing threshold is suitable for every application.
