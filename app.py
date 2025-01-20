from typing import Any
from werkzeug.datastructures import FileStorage

import torch
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
from ultralytics import YOLO

load_dotenv("./env", override=True)

HOST_NAME = os.environ["HOST_NAME"]
PORT = int(os.environ["PORT"])

SECRET_KEY = os.environ["SECRET_KEY"]
TOLERANT_TIME_ERROR = int(os.environ["TOLERANT_TIME_ERROR"])  # 可以容忍的时间戳误差(s)

IMG_SIZE = int(os.environ["IMG_SIZE"])
FALLBACK_IMG_SIZE = int(os.environ["FALLBACK_IMG_SIZE"])

CAT_BOX_MAX_RET_NUM = int(os.environ["CAT_BOX_MAX_RET_NUM"])  # 最多可以返回的猫猫框个数
RECOGNIZE_MAX_RET_NUM = int(os.environ["RECOGNIZE_MAX_RET_NUM"])  # 最多可以返回的猫猫识别结果个数

print("==> loading models...")
assert os.path.isdir(
    "export"
), "*** export directory not found! you should export the training checkpoint to ONNX model."

# crop_model = torch.hub.load("yolov5", "custom", "yolov5/yolov5m.onnx", source="local")
crop_model = YOLO("yolo11m.pt")

with open("export/cat.json", "r") as fp:
    cat_ids = json.load(fp)
cat_model = onnxruntime.InferenceSession(
    "export/cat.onnx", providers=["CPUExecutionProvider"]
)

with open("export/fallback.json", "r") as fp:
    fallback_ids = json.load(fp)
fallback_model = onnxruntime.InferenceSession(
    "export/fallback.onnx", providers=["CPUExecutionProvider"]
)

print("==> models are loaded.")

app = Flask(__name__)
# 限制post大小为10MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def wrap_ok_return_value(data: Any) -> str:
    return json.dumps({"ok": True, "message": "OK", "data": data})


def wrap_error_return_value(message: str) -> str:
    return json.dumps({"ok": False, "message": message, "data": None})


def check_signature(photo: FileStorage, timestamp: int, signature: str) -> bool:
    if abs(timestamp - time.time()) > TOLERANT_TIME_ERROR:
        return False
    photoBase64 = b64encode(photo.read()).decode()
    photo.seek(0)  # 重置读取位置，避免影响后续操作
    signatureData = (photoBase64 + str(timestamp) + SECRET_KEY).encode()
    return signature == sha256(signatureData).hexdigest()


@app.route("/recognizeCatPhoto", methods=["POST"])
@app.route("/recognizeCatPhoto/", methods=["POST"])
def recognize_cat_photo():
    try:
        photo = request.files["photo"]
        timestamp = int(request.form["timestamp"])
        signature = request.form["signature"]
        if not check_signature(photo, timestamp=timestamp, signature=signature):
            return wrap_error_return_value("fail signature check.")

        src_img = Image.open(photo).convert("RGB")
        # 使用 YOLOv5 进行目标检测，结果为[{xmin, ymin, xmax, ymax, confidence, class, name}]格式
        results = crop_model(src_img)
        # 过滤非cat目标
        cat_results = []
        for result in results:
            for box in result.boxes:
                # print(result.names[box.cls.tolist()[0]], box.xyxy.tolist())
                if result.names[box.cls.tolist()[0]] == "cat":
                    cat_results.append(box.xyxy.tolist())

        if len(cat_results) >= 1:
            cat_idx = (
                int(request.form["catIdx"])
                if "catIdx" in request.form
                and int(request.form["catIdx"]) < len(cat_results)
                else 0
            )

            # 裁剪出(指定的)cat
            cat_result = cat_results[cat_idx]
            crop_box = (
                cat_result[0][0],
                cat_result[0][1],
                cat_result[0][2],
                cat_result[0][3],
            )
            # 裁剪后直接resize到正方形
            src_img = src_img.crop(crop_box).resize((IMG_SIZE, IMG_SIZE))

            # 输入到cat模型
            img = np.array(src_img, dtype=np.float32).transpose((2, 0, 1)) / 255
            scores = cat_model.run(
                [node.name for node in cat_model.get_outputs()],
                {cat_model.get_inputs()[0].name: img[np.newaxis, :]},
            )[0][0].tolist()

            # 按概率排序
            cat_id_with_score = sorted(
                [dict(catID=cat_ids[i], score=scores[i]) for i in range(len(cat_ids))],
                key=lambda item: item["score"],
                reverse=True,
            )
        else:
            # 没有检测到cat
            # 整张图片直接resize到正方形
            src_img = src_img.resize((FALLBACK_IMG_SIZE, FALLBACK_IMG_SIZE))

            img = np.array(src_img, dtype=np.float32).transpose((2, 0, 1)) / 255
            scores = fallback_model.run(
                [node.name for node in fallback_model.get_outputs()],
                {fallback_model.get_inputs()[0].name: img[np.newaxis, :]},
            )[0][0].tolist()

            # 按概率排序
            cat_id_with_score = sorted(
                [
                    dict(catID=fallback_ids[i], score=scores[i])
                    for i in range(len(fallback_ids))
                ],
                key=lambda item: item["score"],
                reverse=True,
            )

        return wrap_ok_return_value(
            {
                "catBoxes": [
                    {
                        "xmin": item[0][0],
                        "ymin": item[0][1],
                        "xmax": item[0][2],
                        "ymax": item[0][3],
                    }
                    for item in cat_results
                ][:CAT_BOX_MAX_RET_NUM],
                "recognizeResults": cat_id_with_score[:RECOGNIZE_MAX_RET_NUM],
            }
        )
    except BaseException as err:
        return wrap_error_return_value(str(err))


if __name__ == "__main__":
    app.run(host=HOST_NAME, port=PORT, debug=False)
