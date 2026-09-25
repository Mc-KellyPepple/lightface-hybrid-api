# Add your application code here
from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import JSONResponse
import cv2,numpy as np,onnxruntime as ort,os

app=FastAPI(title="LightFace MobileFaceNet API",version="1.0.0")

BASE=os.path.dirname(os.path.abspath(__file__))
FACE_MODEL=os.path.join(BASE,"models","w600k_mbf.onnx")
DETECT_MODEL=os.path.join(BASE,"models","face_detection_yunet_2023mar.onnx")

face_session=ort.InferenceSession(FACE_MODEL,providers=["CPUExecutionProvider"])
detect_session=ort.InferenceSession(DETECT_MODEL,providers=["CPUExecutionProvider"])

@app.get("/")
def root():
    return {"status":"ok","service":"LightFace MobileFaceNet API"}

@app.get("/health")
def health():
    return {"status":"healthy","models":{"recognition":os.path.exists(FACE_MODEL),"detection":os.path.exists(DETECT_MODEL)}}

@app.post("/recognize")
async def recognize(file:UploadFile=File(...)):
    data=await file.read()
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400,detail="Invalid image")
    return JSONResponse({"status":"received","message":"Image successfully received","width":int(image.shape[1]),"height":int(image.shape[0])})
