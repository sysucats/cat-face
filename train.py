from ultralytics import YOLO
import argparse
import os
import shutil
import torch

# ==================== GPU配置核心部分 ====================
# 1. 检查GPU是否可用
if torch.cuda.is_available():
    device = 0  # 使用第0块GPU（多GPU可设为[0,1]或具体编号）
    print(f"✅ 使用GPU训练，设备编号: {device}")
    print(f"📌 GPU名称: {torch.cuda.get_device_name(device)}")
else:
    device = 'cpu'
    print("⚠️ 未检测到GPU，将使用CPU训练（速度较慢）")


def main():
    parser = argparse.ArgumentParser(description="Cat Recognize Model Trainer")
    parser.add_argument(
        "--data",
        default="data/dataset-cat",
        type=str,
        help="photo data directory (default: data/dataset-cat)",
    )
    parser.add_argument(
        "--size", default=256, type=int, help="image size (default: 256)"
    )
    parser.add_argument(
        "--epoch", default=150, type=int, help="number of epoches to run (default: 150)"
    )
    parser.add_argument(
        "--name", default="cat", type=str, help="model name (default: cat)"
    )
    args = parser.parse_args()

    model = YOLO("yolo11m-cls.pt")
    export_dir = "./export"

    results = model.train(data=f"{args.data}", epochs=args.epoch, imgsz=args.size, device=device)

    # Export the model
    path_to_model = model.export(format="onnx")
    new_model_path = os.path.join(export_dir, f"{args.name}.onnx")

    # 移动并重命名模型文件
    shutil.move(path_to_model, new_model_path)
    print(f"{args.name} done.")


if __name__ == "__main__":
    main()
