"""
Python 对照测试脚本：用 cat-face conda 环境运行，输出基准识别结果
用法: python test/compare.py <图片路径> [catIdx]
输出: JSON 格式的 catBoxes 和 recognizeResults，供 Node 端比对

运行环境: D:/ProgramData/Anaconda3/envs/cat-face/python.exe
"""
import sys
import os
import json
import numpy as np
from PIL import Image
from ultralytics import YOLO
import onnxruntime

# 模型路径（相对项目根目录）
# compare.py 在 node-server/test/ 下，需向上3层到项目根目录 cat-face/
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

IMG_SIZE = 256
FALLBACK_IMG_SIZE = 512
CAT_BOX_MAX_RET_NUM = 5
RECOGNIZE_MAX_RET_NUM = 20

print("==> loading models...", file=sys.stderr)
crop_model = YOLO("yolo11m.pt")
with open("export/cat.json", "r") as fp:
    cat_ids = json.load(fp)
cat_model = onnxruntime.InferenceSession("export/cat.onnx", providers=["CPUExecutionProvider"])
with open("export/fallback.json", "r") as fp:
    fallback_ids = json.load(fp)
fallback_model = onnxruntime.InferenceSession("export/fallback.onnx", providers=["CPUExecutionProvider"])
print("==> models are loaded.", file=sys.stderr)


def recognize(photo_path, cat_idx_param=None):
    src_img = Image.open(photo_path).convert("RGB")

    # YOLO 检测
    results = crop_model(src_img)
    cat_results = []
    for result in results:
        for box in result.boxes:
            if result.names[box.cls.tolist()[0]] == "cat":
                cat_results.append(box.xyxy.tolist())

    if len(cat_results) >= 1:
        cat_idx = 0
        if cat_idx_param is not None and cat_idx_param < len(cat_results):
            cat_idx = cat_idx_param
        cat_result = cat_results[cat_idx]
        crop_box = (cat_result[0][0], cat_result[0][1], cat_result[0][2], cat_result[0][3])
        src_img_cropped = src_img.crop(crop_box).resize((IMG_SIZE, IMG_SIZE))
        img = np.array(src_img_cropped, dtype=np.float32).transpose((2, 0, 1)) / 255
        scores = cat_model.run(
            [node.name for node in cat_model.get_outputs()],
            {cat_model.get_inputs()[0].name: img[np.newaxis, :]},
        )[0][0].tolist()
        ids = cat_ids
    else:
        src_img_resized = src_img.resize((FALLBACK_IMG_SIZE, FALLBACK_IMG_SIZE))
        img = np.array(src_img_resized, dtype=np.float32).transpose((2, 0, 1)) / 255
        scores = fallback_model.run(
            [node.name for node in fallback_model.get_outputs()],
            {fallback_model.get_inputs()[0].name: img[np.newaxis, :]},
        )[0][0].tolist()
        ids = fallback_ids

    cat_id_with_score = sorted(
        [dict(catID=ids[i], score=scores[i]) for i in range(len(ids))],
        key=lambda item: item["score"], reverse=True,
    )

    return {
        "catBoxes": [
            {"xmin": item[0][0], "ymin": item[0][1], "xmax": item[0][2], "ymax": item[0][3]}
            for item in cat_results
        ][:CAT_BOX_MAX_RET_NUM],
        "recognizeResults": cat_id_with_score[:RECOGNIZE_MAX_RET_NUM],
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python test/compare.py <图片路径> [catIdx]", file=sys.stderr)
        sys.exit(1)
    photo_path = sys.argv[1]
    cat_idx = int(sys.argv[2]) if len(sys.argv) >= 3 else None
    result = recognize(photo_path, cat_idx)
    print(json.dumps(result, ensure_ascii=False, indent=2))
