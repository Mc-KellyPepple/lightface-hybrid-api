from fastapi import FastAPI,File,UploadFile,HTTPException
from fastapi.responses import JSONResponse
import cv2,numpy as np,os

cv2.setNumThreads(1)

app=FastAPI(title="LightFace MobileFaceNet API",version="1.1.0")

BASE=os.path.dirname(os.path.abspath(__file__))
FACE_MODEL=os.path.join(BASE,"models","w600k_mbf.onnx")
DETECT_MODEL=os.path.join(BASE,"models","face_detection_yunet_2023mar.onnx")

# MobileFaceNet intentionally NOT loaded yet.
# Alignment is tested independently before loading the recognition model.

detector=cv2.FaceDetectorYN.create(
    DETECT_MODEL,
    "",
    (320,320),
    0.5,
    0.3,
    50
)

# Standard 5-point ArcFace-style 112x112 reference landmarks.
REFERENCE_POINTS=np.array([
    [38.2946,51.6963],
    [73.5318,51.5014],
    [56.0252,71.7366],
    [41.5493,92.3655],
    [70.7299,92.2041]
],dtype=np.float32)


@app.get("/")
def root():
    return {
        "status":"ok",
        "service":"LightFace MobileFaceNet API",
        "version":"1.1.0"
    }


@app.get("/health")
def health():
    return {
        "status":"healthy",
        "models":{
            "detection":os.path.exists(DETECT_MODEL),
            "recognition_file":os.path.exists(FACE_MODEL),
            "recognition_loaded":False
        },
        "alignment":{
            "enabled":True,
            "output_size":[112,112]
        }
    }


def detect_faces(image):
    h,w=image.shape[:2]

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


def align_face(image,landmarks):
    src=np.asarray(landmarks,dtype=np.float32)

    if src.shape!=(5,2):
        raise ValueError("Exactly 5 facial landmarks are required.")

    transform,_=cv2.estimateAffinePartial2D(
        src,
        REFERENCE_POINTS,
        method=cv2.LMEDS
    )

    if transform is None:
        raise ValueError("Unable to calculate face alignment transform.")

    aligned=cv2.warpAffine(
        image,
        transform,
        (112,112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0)
    )

    return aligned


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
    aligned_faces=[]

    for i,face in enumerate(faces):
        try:
            aligned=align_face(
                image,
                face["landmarks"]
            )

            aligned_faces.append({
                "face_index":i,
                "size":{
                    "width":112,
                    "height":112
                },
                "alignment":"5-point_similarity",
                "status":"success"
            })

        except Exception as e:
            aligned_faces.append({
                "face_index":i,
                "size":{
                    "width":112,
                    "height":112
                },
                "alignment":"5-point_similarity",
                "status":"failed",
                "error":str(e)
            })

    return JSONResponse({
        "status":"success",
        "image":{
            "width":int(image.shape[1]),
            "height":int(image.shape[0])
        },
        "face_count":len(faces),
        "faces":faces,
        "aligned_faces":aligned_faces,
        "recognition":{
            "loaded":False,
            "embedding_generated":False
        }
    })
