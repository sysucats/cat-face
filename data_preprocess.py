import os
import shutil
import torch
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO

SRC = "data/photos"
DEST = "data/crop_photos"
# SRC = "test/photos"
# DEST = "test/crop_photos"

if __name__ == "__main__":
    print("loading YOLO11 model...")
    model = YOLO("yolo11m.pt")
    # model = torch.hub.load("yolov5", "custom", "yolov5/yolov5m.pt", source="local")

    num_photos = 0
    num_skipped_photos = 0

    if os.path.exists(DEST):
        shutil.rmtree(DEST)
    os.mkdir(DEST)

    print("processing photos...")
    for dir_name in tqdm(os.listdir(SRC), leave=False, desc="processing"):
        src_path = os.path.join(SRC, dir_name)
        if not os.path.isdir(src_path):
            continue

        dest_path = os.path.join(DEST, dir_name)
        os.mkdir(dest_path)

        for file_name in tqdm(os.listdir(src_path), leave=False, desc=dir_name):
            num_photos += 1

            src_file_path = os.path.join(src_path, file_name)
            dest_file_path = os.path.join(dest_path, file_name)
            # 使用 YOLOv5 进行目标检测，结果为[{xmin, ymin, xmax, ymax, confidence, class, name}]格式
            try:
                # results = model(src_file_path).pandas().xyxy[0].to_dict("records")
                results = model(src_file_path)
                # print(results[0].boxes.xyxy.tolist(), results[0].names[results[0].boxes.cls.tolist()[0]])
            except Exception as err:
                # 发现有的图片有问题，会导致 PIL 抛出 OSError: image file is truncated
                # 全部都跳过
                num_skipped_photos += 1
                continue
            # 过滤非cat目标
            cat_results = []
            for result in results:
                for box in result.boxes:
                    print(result.names[box.cls.tolist()[0]], box.xyxy.tolist())
                    if result.names[box.cls.tolist()[0]] == "cat":
                        cat_results.append(box.xyxy.tolist())
            # 跳过图片内检测不到cat或有多个cat的图片
            if len(cat_results) == 0:
                num_skipped_photos += 1
                continue
            # 裁剪出cat
            cat_result = cat_results[0]
            crop_box = (
                cat_result[0][0],
                cat_result[0][1],
                cat_result[0][2],
                cat_result[0][3],
            )
            Image.open(src_file_path).convert("RGB").crop(crop_box).save(
                dest_file_path, format="JPEG"
            )

    print(
        f"done. {num_photos - num_skipped_photos} photos processed, {num_skipped_photos} photos skipped."
    )
