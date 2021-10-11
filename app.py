from typing import Any

import onnxruntime
from PIL import Image
import numpy as np
from scipy.special import softmax
from flask import Flask, request
import os
import json


HOST_NAME="0.0.0.0"
PORT=3456

IMG_SIDE_LEN = 224

MAX_RET_NUM = 20 # 最多可以返回的猫猫个数

assert os.path.isdir("onnx"), "### onnx not found!"
with open("onnx/cat-ids.json", "r") as fp:
    catIDs = json.load(fp)
model = onnxruntime.InferenceSession("onnx/model.onnx")

def wrapOKRetVal(data: Any) -> str:
    return json.dumps({
        'ok': True,
        'message': 'OK',
        'data': data
    })

def wrapErrorRetVal(message: str) -> str:
    return json.dumps({
        'ok': False,
        'message': message,
        'data': None
    })

app = Flask(__name__)
# 限制post大小为10MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

@app.route("/recognizeCatPhoto", methods=["POST"])
def recognizeCatPhoto():
    try:
        photo = request.files['photo']
        img = Image.open(photo).convert("RGB").resize((IMG_SIDE_LEN, IMG_SIDE_LEN))
        imgData = np.array(img, dtype=np.float32).transpose((2, 0, 1)) / 255
        # inference
        probs = softmax(model.run(["prob"], {"photo": imgData[np.newaxis, :]})[0][0], axis=0).tolist()
        # 按概率排序
        catIDWithProb = sorted([dict(catID=catIDs[i], prob=probs[i]) for i in range(len(catIDs))], key=lambda item: item['prob'], reverse=True)
        return wrapOKRetVal(catIDWithProb[:MAX_RET_NUM])
    except BaseException as err:
        return wrapErrorRetVal(str(err))

if __name__ == "__main__":
    app.run(host=HOST_NAME, port=PORT, debug=False)