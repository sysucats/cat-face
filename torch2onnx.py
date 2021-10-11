import torch
from torch.onnx import TrainingMode
from model import ResNet
import os
import json

IMG_SIDE_LEN = 128

print("==> loading model...")
assert os.path.isdir("checkpoint"), "### checkpoint not found!"
checkpoint = torch.load("./checkpoint/ckpt.pth", map_location="cpu")
catIDs = checkpoint["catIDs"]
model = ResNet(numTargets=len(catIDs))
model.load_state_dict(checkpoint["model"])
model.eval()
print(f"==> model loaded. info: number of cats = {len(catIDs)}, model is trained for {checkpoint['epoch'] + 1} epoches, test ACC = {checkpoint['ACC'] :.2f}%")
# 手动回收内存
del checkpoint

if not os.path.isdir("onnx"):
    os.mkdir("onnx")

print("==> saving catIDs to JSON file...")
with open("onnx/cat-ids.json", "w") as fp:
    json.dump(catIDs, fp)

print("==> exporting model to onnx file...")
exampleTensor = torch.randn(1, 3, IMG_SIDE_LEN, IMG_SIDE_LEN)
torch.onnx.export(model, exampleTensor, "onnx/model.onnx", training=TrainingMode.EVAL, do_constant_folding=True, input_names=['photo'], output_names=['prob'])

print("==> export to onnx should be done.")