from typing import Any

import onnxruntime
from PIL import Image
import numpy as np
from scipy.special import softmax
from flask import Flask, request
import os
import json


IMG_SIDE_LEN = 128
RETURN_NUM = 3
HOST_NAME="localhost"
PORT=3456

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
        img = Image.open(photo).resize((IMG_SIDE_LEN, IMG_SIDE_LEN))
        imgData = np.array(img, dtype=np.float32).transpose((2, 0, 1)) / 255
        prob = softmax(model.run(["prob"], {"photo": imgData[np.newaxis, :]})[0][0], axis=0)
        catIDWithProb = sorted(zip(catIDs, prob.tolist()), key=lambda item: item[1], reverse=True)
        # 返回概率最高的前RETURN_NUM名及其对应概率
        return wrapOKRetVal(catIDWithProb[:RETURN_NUM])
    except BaseException as err:
        return wrapErrorRetVal(str(err))

if __name__ == "__main__":
    app.run(host=HOST_NAME, port=PORT, debug=False)