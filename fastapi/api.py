import requests
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
from fastapi import FastAPI
from typing import Union
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

app = FastAPI()

@app.get('/')
def prediction(url : Union[str, None]):
    url = url
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    model = ResNet50(weights='imagenet')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    res = decode_predictions(preds, top=3)[0]
    item = res[0][1]
    prob = res[0][2]
    print(item, prob)
    return {'item': f'{item}', 'prob': f'{prob}'}