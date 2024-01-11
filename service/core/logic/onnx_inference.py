import onnxruntime as rt
import cv2
import numpy as np
import time
import service.main as s

def emotion_detector(img_array):
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    test_image = cv2.resize(img_array, (256, 256))
    img_array = np.expand_dims(test_image, axis=0)
    
    time_start = time.time()

    onnx_pred = s.m_q.run(['dense_5'], {'input':img_array})

    time_end = time.time() - time_start

    emotion = ''
    if np.argmax(onnx_pred) == 0:
        emotion = 'Angry'
    elif np.argmax(onnx_pred) == 1:
        emotion = 'Happy'
    else:
        emotion = 'Sad'
    
    return {
        'emotion': emotion,
        # 'time4loading': str(time4loading),
        'total_time': str(time_end)
    } 