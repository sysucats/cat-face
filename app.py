from typing import Any
from werkzeug.datastructures import FileStorage

from yolo_interface import yolov5m
from PIL import Image
import numpy as np
import onnxruntime
from flask import Flask, request
from dotenv import load_dotenv
import os
import json
import time
from base64 import b64encode
from hashlib import sha256

def main():
    load_dotenv()

    HOST_NAME = os.environ['HOST_NAME']
    PORT= os.environ['PORT']

    SECRET_KEY = os.environ['SECRET_KEY']
    TOLERANT_TIME_ERROR = 30 # 可以容忍的时间戳误差(s)

    IMG_SIZE = 128
    FALLBACK_IMG_SIZE = 224

    CAT_BOX_MAX_RET_NUM = 5 # 最多可以返回的猫猫框个数
    RECOGNIZE_MAX_RET_NUM = 20 # 最多可以返回的猫猫识别结果个数

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

    def checkSignature(photo: FileStorage, timestamp: int, signature: str) -> bool:
        if abs(timestamp - time.time()) > TOLERANT_TIME_ERROR:
            return False
        photoBase64 = b64encode(photo.read()).decode()
        photo.seek(0) # 重置读取位置，避免影响后续操作
        signatureData = (photoBase64 + str(timestamp) + SECRET_KEY).encode()
        return signature == sha256(signatureData).hexdigest()

    @app.route("/recognizeCatPhoto", methods=["POST"])
    def recognizeCatPhoto():
        try:
            photo = request.files['photo']
            timestamp = int(request.form['timestamp'])
            signature = request.form['signature']
            if not checkSignature(photo, timestamp=timestamp, signature=signature):
                return wrapErrorRetVal("fail signature check.")
            
            srcImg = Image.open(photo).convert("RGB")
            # 使用 YOLOv5 进行目标检测，结果为[{xmin, ymin, xmax, ymax, confidence, class, name}]格式
            results = cropModel(srcImg).pandas().xyxy[0].to_dict('records')
            # 过滤非cat目标
            catResults = list(filter(lambda target: target['name'] == 'cat', results))
            if len(catResults) >= 1:
                catIdx = int(request.form['catIdx']) if 'catIdx' in request.form and int(request.form['catIdx']) < len(catResults) else 0
                # 裁剪出(指定的)cat
                catResult = catResults[catIdx]
                cropBox = catResult['xmin'], catResult['ymin'], catResult['xmax'], catResult['ymax']
                # 裁剪后直接resize到正方形
                srcImg = srcImg.crop(cropBox).resize((IMG_SIZE, IMG_SIZE))
                # 输入到cat模型
                imgData = np.array(srcImg, dtype=np.float32).transpose((2, 0, 1)) / 255
                scores = catModel.run(["prob"], {"photo": imgData[np.newaxis, :]})[0][0].tolist()
                # 按概率排序
                catIDWithScore = sorted([dict(catID=catIDs[i], score=scores[i]) for i in range(len(catIDs))], key=lambda item: item['score'], reverse=True)
            else:
                # 没有检测到cat
                # 整张图片直接resize到正方形
                srcImg = srcImg.resize((FALLBACK_IMG_SIZE, FALLBACK_IMG_SIZE))
                imgData = np.array(srcImg, dtype=np.float32).transpose((2, 0, 1)) / 255
                scores = fallbackModel.run(["prob"], {"photo": imgData[np.newaxis, :]})[0][0].tolist()
                # 按概率排序
                catIDWithScore = sorted([dict(catID=fallbackIDs[i], score=scores[i]) for i in range(len(fallbackIDs))], key=lambda item: item['score'], reverse=True)
            return wrapOKRetVal({
                'catBoxes': [{
                    'xmin': item['xmin'],
                    'ymin': item['ymin'],
                    'xmax': item['xmax'],
                    'ymax': item['ymax']
                } for item in catResults][:CAT_BOX_MAX_RET_NUM],
                'recognizeResults': catIDWithScore[:RECOGNIZE_MAX_RET_NUM]
            })
        except BaseException as err:
            return wrapErrorRetVal(str(err))

    app.run(host=HOST_NAME, port=PORT, debug=False)

if __name__ == "__main__":
    main()
