from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import base64
import numpy as np
import cv2
import face_recognition

import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


def detect_aadharImage(liveImage: str, aadharImage: str):
    decoded_aadharImage_str = base64.b64decode(aadharImage)
    aadharImage_arr = np.frombuffer(decoded_aadharImage_str, dtype=np.uint8)
    aadharImageFile = cv2.imdecode(aadharImage_arr, flags=cv2.IMREAD_COLOR)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        height, width, c = aadharImageFile.shape

        # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
        results = face_detection.process(cv2.cvtColor(aadharImageFile, cv2.COLOR_BGR2RGB))

        if not results.detections:
            print('No faces detected.')
        else:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                bbox_points = {
                    "xmin": int(bbox.xmin * width),
                    "ymin": int(bbox.ymin * height),
                    "xmax": int(bbox.width * width + bbox.xmin * width),
                    "ymax": int(bbox.height * height + bbox.ymin * height)
                }

        cropped_image = aadharImageFile[bbox_points["ymin"] : bbox_points["ymax"], bbox_points["xmin"] : bbox_points["ymax"]].copy()

    result = detect_image(liveImage, cropped_image)

    return result


def detect_image(liveImage: str, aadharImage):
    decoded_liveImage_string = base64.b64decode(liveImage)
    liveImage_arr = np.frombuffer(decoded_liveImage_string, dtype=np.uint8)
    liveImageFile = cv2.imdecode(liveImage_arr, flags=cv2.IMREAD_COLOR)

    known_image = liveImageFile
    unknown_image = aadharImage

    known_encoding = face_recognition.face_encodings(known_image)[0]
    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    results = face_recognition.compare_faces([known_encoding], unknown_encoding)

    return results


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageData(BaseModel):
    LiveimgStream: str
    AadharimgStream: str


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/api/verifyImage")
async def say_hello(imgData: ImageData):
    results = detect_aadharImage(imgData.LiveimgStream, imgData.AadharimgStream)

    response = {
        "results": results
    }

    return response
