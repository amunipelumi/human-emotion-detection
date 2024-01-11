from fastapi import APIRouter, UploadFile, HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
from service.core.logic.onnx_inference import emotion_detector
from service.core.schemas.output import ApiOutput


detect_router = APIRouter()

@detect_router.post('/detect', response_model=ApiOutput)
def detect(img:UploadFile):

    if img.filename.split('.')[-1] in ('jpg', 'jpeg', 'png'):
        pass
    else:
        raise HTTPException(
            status_code=415, detail='Image not found'
        )
    
    image = Image.open(BytesIO(img.file.read()))
    image = np.float32(image)

    return emotion_detector(image)