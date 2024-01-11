from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as rt
from service.api.api import main_router

app = FastAPI(project_name='Emotion Detection')

app.include_router(main_router)

@app.get("/")
def read_root():
    return {"Hello": "World"}

model_dir = 'service/Eff_Net_Quantized.onnx'
provider = ['CPUExecutionProvider']
m_q = rt.InferenceSession(model_dir, providers=provider)
