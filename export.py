import torch
from torch.onnx import TrainingMode
from model import ResNet
import os
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Export cat recognization training checkpoint to onnx model and json file.")
    parser.add_argument("--checkpoint", default="cat.pth", type=str, help="path of traning checkpoint file")
    parser.add_argument("--size", default=128, type=int, help="image size")
    args = parser.parse_args()

    print("==> loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    catIDs = checkpoint["catIDs"]
    print(f"==> checkpoint loaded. info: number of cats = {len(catIDs)}, model is trained for {checkpoint['epoch'] + 1} epoches, test ACC = {checkpoint['ACC'] :.2f}%")

    print("==> building model...")
    model = ResNet(numTargets=len(catIDs))
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if not os.path.isdir("export"):
        os.mkdir("export")
    
    # 获取不带拓展名的文件名
    fileName = os.path.splitext(os.path.basename(args.checkpoint))[0]

    print(f"==> saving catIDs to export/{fileName}.json...")
    with open(f"export/{fileName}.json", "w") as fp:
        json.dump(catIDs, fp)

    print(f"==> exporting model to export/{fileName}.onnx...")
    exampleTensor = torch.randn(1, 3, args.size, args.size)
    torch.onnx.export(model, exampleTensor, f"export/{fileName}.onnx", training=TrainingMode.EVAL, do_constant_folding=True, input_names=['photo'], output_names=['prob'])

    print("==> all done.")

if __name__ == "__main__":
    main()
