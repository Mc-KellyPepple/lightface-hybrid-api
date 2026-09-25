from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import JSONResponse
import cv2,numpy as np,onnxruntime as ort,os

app=FastAPI(title="LightFace MobileFaceNet API",version="1.0.0")

BASE=os.path.dirname(os.path.abspath(__file__))
FACE_MODEL=os.path.join(BASE,"models","w600k_mbf.onnx")
DETECT_MODEL=os.path.join(BASE,"models","face_detection_yunet_2023mar.onnx")

face_session=ort.InferenceSession(FACE_MODEL,providers=["CPUExecutionProvider"])

detector=cv2.FaceDetectorYN.create(
    DETECT_MODEL,
    "",
    (320,320),
    0.5,
    0.3,
    5000
)

@app.get("/")
def root():
    return {
        "status":"ok",
        "service":"LightFace MobileFaceNet API"
    }

@app.get("/health")
def health():
    return {
        "status":"healthy",
        "models":{
            "recognition":os.path.exists(FACE_MODEL),
            "detection":os.path.exists(DETECT_MODEL)
        }
    }

def detect_faces(image):
    h,w=image.shape[:2]
    detector.setInputSize((w,h))
    _,faces=detector.detect(image)
    if faces is None:
        return []
    results=[]
    for face in faces:
        x,y,bw,bh=face[:4]
        confidence=float(face[-1])
        x1=max(0,int(x))
        y1=max(0,int(y))
        x2=min(w,int(x+bw))
        y2=min(h,int(y+bh))
        results.append({
            "box":[x1,y1,x2,y2],
            "confidence":round(confidence,4),
            "landmarks":[
                [round(float(face[4]),2),round(float(face[5]),2)],
                [round(float(face[6]),2),round(float(face[7]),2)],
                [round(float(face[8]),2),round(float(face[9]),2)],
                [round(float(face[10]),2),round(float(face[11]),2)],
                [round(float(face[12]),2),round(float(face[13]),2)]
            ]
        })
    return results

@app.post("/recognize")
async def recognize(file:UploadFile=File(...)):
    data=await file.read()
    image=cv2.imdecode(
        np.frombuffer(data,np.uint8),
        cv2.IMREAD_COLOR
    )
    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )
    faces=detect_faces(image)
    return JSONResponse({
        "status":"success",
        "image":{
            "width":int(image.shape[1]),
            "height":int(image.shape[0])
        },
        "face_count":len(faces),
        "faces":faces
    })
