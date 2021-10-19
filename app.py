from typing import Any

import torch
from yolo_interface import yolov5m
from model import ResNet
from PIL import Image
import numpy as np
from flask import Flask, request
import json


HOST_NAME="localhost"
PORT=3456

IMG_SIZE = 100
FALLBACK_IMG_SIZE = 224

MAX_RET_NUM = 10 # 最多可以返回的猫猫个数

print("==> loading models...")

cropModel = yolov5m()

catModel = ResNet()
checkpoint = torch.load("cat.pth", map_location="cpu")
print(f"*** cat model at epoch {checkpoint['epoch'] + 1}, ACC: {checkpoint['ACC'] :.2f}%")
catModel.load_state_dict(checkpoint["model"])
catModel.eval()
catModel = torch.jit.trace(catModel, torch.randn((1, 3, IMG_SIZE, IMG_SIZE)))
catIDs = checkpoint["catIDs"]

fallbackModel = ResNet()
checkpoint = torch.load("fallback.pth", map_location="cpu")
print(f"*** fallback model at epoch {checkpoint['epoch'] + 1}, ACC: {checkpoint['ACC'] :.2f}%")
fallbackModel.load_state_dict(checkpoint["model"])
fallbackModel.eval()
fallbackModel = torch.jit.trace(fallbackModel, torch.randn((1, 3, FALLBACK_IMG_SIZE, FALLBACK_IMG_SIZE)))
fallbackIDs = checkpoint["catIDs"]

del checkpoint
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

@app.route("/recognizeCatPhoto", methods=["POST"])
@torch.no_grad()
def recognizeCatPhoto():
    try:
        photo = request.files['photo']
        srcImg = Image.open(photo).convert("RGB")
        # 使用 YOLOv5 进行目标检测，结果为[{xmin, ymin, xmax, ymax, confidence, class, name}]格式
        results = cropModel(srcImg).pandas().xyxy[0].to_dict('records')
        # 过滤非cat目标
        catResults = list(filter(lambda target: target['name'] == 'cat', results))
        numCats = len(catResults)
        if numCats == 1:
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
            probs = torch.softmax(catModel(torch.Tensor(imgData))[0], dim=0).tolist()
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
            probs = torch.softmax(fallbackModel(torch.Tensor(imgData))[0], dim=0).tolist()
            # 按概率排序
            catIDWithProb = sorted([dict(catID=fallbackIDs[i], prob=probs[i]) for i in range(len(fallbackIDs))], key=lambda item: item['prob'], reverse=True)
        return wrapOKRetVal({
            'numCats': numCats,
            'result': catIDWithProb[:MAX_RET_NUM]
        })
    except BaseException as err:
        return wrapErrorRetVal(str(err))

if __name__ == "__main__":
    app.run(host=HOST_NAME, port=PORT, debug=False)