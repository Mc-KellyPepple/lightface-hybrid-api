from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import JSONResponse
import cv2,numpy as np,os

cv2.setNumThreads(1)

app=FastAPI(title="LightFace MobileFaceNet API",version="1.0.0")

BASE=os.path.dirname(os.path.abspath(__file__))
FACE_MODEL=os.path.join(BASE,"models","w600k_mbf.onnx")
DETECT_MODEL=os.path.join(BASE,"models","face_detection_yunet_2023mar.onnx")

# Recognition model intentionally NOT loaded yet.
# This prevents w600k_mbf.onnx from consuming memory during startup.

detector=cv2.FaceDetectorYN.create(
    DETECT_MODEL,
    "",
    (320,320),
    0.5,
    0.3,
    50
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
            "detection":os.path.exists(DETECT_MODEL),
            "recognition_file":os.path.exists(FACE_MODEL),
            "recognition_loaded":False
        }
    }

def detect_faces(image):
    h,w=image.shape[:2]

    # Limit detector input size to control memory usage.
    max_side=1280
    scale=min(1.0,max_side/max(h,w))

    if scale<1.0:
        nw,nh=int(w*scale),int(h*scale)
        work=cv2.resize(
            image,
            (nw,nh),
            interpolation=cv2.INTER_AREA
        )
    else:
        work=image
        nw,nh=w,h

    detector.setInputSize((nw,nh))
    _,faces=detector.detect(work)

    if faces is None:
        return []

    sx=w/nw
    sy=h/nh
    results=[]

    for face in faces:
        x,y,bw,bh=map(float,face[:4])
        confidence=float(face[-1])

        results.append({
            "box":[
                max(0,int(x*sx)),
                max(0,int(y*sy)),
                min(w,int((x+bw)*sx)),
                min(h,int((y+bh)*sy))
            ],
            "confidence":round(confidence,4),
            "landmarks":[
                [
                    round(float(face[4])*sx,2),
                    round(float(face[5])*sy,2)
                ],
                [
                    round(float(face[6])*sx,2),
                    round(float(face[7])*sy,2)
                ],
                [
                    round(float(face[8])*sx,2),
                    round(float(face[9])*sy,2)
                ],
                [
                    round(float(face[10])*sx,2),
                    round(float(face[11])*sy,2)
                ],
                [
                    round(float(face[12])*sx,2),
                    round(float(face[13])*sy,2)
                ]
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
