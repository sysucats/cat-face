from typing import Any
from werkzeug.datastructures import FileStorage

from yolo_interface import yolov5m
from PIL import Image
import numpy as np
import onnxruntime
from scipy.special import softmax
from flask import Flask, request
import os
import json
from base64 import b64encode
from hashlib import sha256


HOST_NAME="127.0.0.1"
PORT=3456

SECRET_KEY = "xxx"

IMG_SIZE = 100
FALLBACK_IMG_SIZE = 224

MAX_RET_NUM = 10 # 最多可以返回的猫猫个数

print("==> loading models...")
assert os.path.isdir("export"), "*** export directory not found! you should export the training checkpoint to ONNX model."

cropModel = yolov5m()

with open("export/cat.json", "r") as fp:
    catIDs = json.load(fp)
catModel = onnxruntime.InferenceSession("export/cat.onnx", providers=["CPUExecutionProvider"])

with open("export/fallback.json", "r") as fp:
    fallbackIDs = json.load(fp)
fallbackModel = onnxruntime.InferenceSession("export/fallback.onnx", providers=["CPUExecutionProvider"])

print("==> models are loaded.")

app = Flask(__name__)
# 限制post大小为10MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

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

def checkSignature(photo: FileStorage, signature: str) -> bool:
    photoBase64 = b64encode(photo.read()).decode()
    photo.seek(0) # 重置读取位置，避免影响后续操作
    signatureData = (photoBase64 + SECRET_KEY).encode()
    return signature == sha256(signatureData).hexdigest()

@app.route("/recognizeCatPhoto", methods=["POST"])
def recognizeCatPhoto():
    try:
        photo = request.files['photo']
        signature = request.form['signature']
        if not checkSignature(photo, signature=signature):
            return wrapErrorRetVal("fail signature check.")
        
        srcImg = Image.open(photo).convert("RGB")
        # 使用 YOLOv5 进行目标检测，结果为[{xmin, ymin, xmax, ymax, confidence, class, name}]格式
        results = cropModel(srcImg).pandas().xyxy[0].to_dict('records')
        # 过滤非cat目标
        catResults = list(filter(lambda target: target['name'] == 'cat', results))
        if len(catResults) == 1:
            # 裁剪出cat
            catResult = catResults[0]
            cropBox = catResult['xmin'], catResult['ymin'], catResult['xmax'], catResult['ymax']
            srcImg = srcImg.crop(cropBox)
            # 进行等比缩放+padding
            ratio = IMG_SIZE / max(srcImg.width, srcImg.height)
            unpadSize = int(round(srcImg.width * ratio)), int(round(srcImg.height * ratio))
            srcImg = srcImg.resize(unpadSize)
            dw = (IMG_SIZE - unpadSize[0]) / 2
            dh = (IMG_SIZE - unpadSize[1]) / 2
            left = int(round(dw - 0.1))
            top = int(round(dh - 0.1))
            padImg = Image.new(mode="RGB", size=(IMG_SIZE, IMG_SIZE), color=(114, 114, 114))
            padImg.paste(srcImg, box=(left, top))
            # 输入到cat模型
            imgData = np.array(padImg, dtype=np.float32).transpose((2, 0, 1)) / 255
            probs = softmax(catModel.run(["prob"], {"photo": imgData[np.newaxis, :]})[0][0], axis=0).tolist()
            # 按概率排序
            catIDWithProb = sorted([dict(catID=catIDs[i], prob=probs[i]) for i in range(len(catIDs))], key=lambda item: item['prob'], reverse=True)
        else:
            # 没有检测到cat或有多个cat
            # 进行等比缩放+padding
            ratio = FALLBACK_IMG_SIZE / max(srcImg.width, srcImg.height)
            unpadSize = int(round(srcImg.width * ratio)), int(round(srcImg.height * ratio))
            srcImg = srcImg.resize(unpadSize)
            dw = (FALLBACK_IMG_SIZE - unpadSize[0]) / 2
            dh = (FALLBACK_IMG_SIZE - unpadSize[1]) / 2
            left = int(round(dw - 0.1))
            top = int(round(dh - 0.1))
            padImg = Image.new(mode="RGB", size=(FALLBACK_IMG_SIZE, FALLBACK_IMG_SIZE), color=(114, 114, 114))
            padImg.paste(srcImg, box=(left, top))
            # 输入到fallback模型
            imgData = np.array(padImg, dtype=np.float32).transpose((2, 0, 1)) / 255
            probs = softmax(fallbackModel.run(["prob"], {"photo": imgData[np.newaxis, :]})[0][0], axis=0).tolist()
            # 按概率排序
            catIDWithProb = sorted([dict(catID=fallbackIDs[i], prob=probs[i]) for i in range(len(fallbackIDs))], key=lambda item: item['prob'], reverse=True)
        return wrapOKRetVal({
            'catBoxes': [{
                'xmin': item['xmin'],
                'ymin': item['ymin'],
                'xmax': item['xmax'],
                'ymax': item['ymax']
            } for item in catResults][:MAX_RET_NUM],
            'recognizeResults': catIDWithProb[:MAX_RET_NUM]
        })
    except BaseException as err:
        return wrapErrorRetVal(str(err))

if __name__ == "__main__":
    app.run(host=HOST_NAME, port=PORT, debug=False)