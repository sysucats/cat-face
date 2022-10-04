#!/bin/bash

# 拉取 YOLOv5 代码
git submodule update --init

# 使用代理地址提前下载 YOLOv5 需要的模型文件，以避免运行时因为访问不到 Github 卡住
wget https://ghproxy.com/https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m.pt -O ./yolov5/yolov5m.pt

# 进入 YOLOv5 项目，导出 ONNX 模型
cd yolov5
python export.py --weights yolov5m.pt --include onnx