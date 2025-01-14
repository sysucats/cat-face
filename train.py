from ultralytics import YOLO
import argparse
import os
import shutil


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

    results = model.train(data=f"{args.data}", epochs=args.epoch, imgsz=args.size)

    # Export the model
    path_to_model = model.export(format="onnx")
    new_model_path = os.path.join(export_dir, f"{args.name}.onnx")

    # 移动并重命名模型文件
    shutil.move(path_to_model, new_model_path)
    print(f"{args.name} done.")


if __name__ == "__main__":
    main()
