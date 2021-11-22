import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from data import CatPhotoDatasetHelper
from model import ResNet
import argparse
import os
from tqdm import tqdm

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Cat Face Recognization")
    parser.add_argument("--data", default="crop-photos", type=str, help="photo data directory")
    parser.add_argument("--size", default=128, type=int, help="image size")
    parser.add_argument("--filter", default=10, type=int, help="cats whose photo number are less than this would be filtered")
    parser.add_argument("--balance", default=30, type=int, help="dataset class pseudo balance number")
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate")
    parser.add_argument("--batch", default=8, type=int, help="batch size")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    args = parser.parse_args()

    # 确定计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 准备数据
    print("==> preparing data...")

    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # 亮度、对比度、饱和度均在0.9~1.1间随机变化
        transforms.RandomHorizontalFlip(p=0.5), # 以0.5的概率发生水平翻转
        transforms.RandomVerticalFlip(p=0.5), # 以0.5的概率发生垂直翻转
        transforms.RandomCrop(args.size, padding=args.size // 8, fill=0), # 四周填充1/8大小后再随机裁剪为args.sizexargs.size
        transforms.RandomRotation(degrees=15, fill=0), # 随机旋转-15~15度
    ])

    dataHelper = CatPhotoDatasetHelper(root=args.data, size=args.size, filterNum=args.filter, balanceNum=args.balance, transform=transform)

    trainLoader = torch.utils.data.DataLoader(dataset=dataHelper.trainDataset, batch_size=args.batch, shuffle=True, num_workers=2)
    testLoader = torch.utils.data.DataLoader(dataset=dataHelper.testDataset, batch_size=args.batch, shuffle=False, num_workers=2)
    
    # 构建模型
    print("==> building model...")
    model = ResNet(numTargets=len(dataHelper.catIDs))
    model = model.to(device=device)

    bestACC = 0.
    startEpoch = 0

    # 从检查点恢复训练
    if args.resume:
        print("==> resuming from checkpoint...")
        assert os.path.isdir("checkpoint"), "### checkpoint not found!"
        checkpoint = torch.load("./checkpoint/ckpt.pth", map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        bestACC = checkpoint["ACC"]
        startEpoch = checkpoint["epoch"] + 1
        assert checkpoint["catIDs"] == dataHelper.catIDs, "### failed checking catIDs!"

    # 损失函数及优化器
    lossFunc = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    for epoch in range(startEpoch, startEpoch + 50):
        print(f"---------- epoch {epoch :>3d} ----------")

        # 在训练集上训练模型
        model.train()
        trainLoss = 0.
        trainNum = 0
        trainCorrect = 0
        for x, y in tqdm(trainLoader, leave=False, desc="training"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = lossFunc(out, y)
            loss.backward()
            optimizer.step()
            # 统计损失值及正确率
            trainLoss += loss.item()
            _, pred = out.max(dim=1)
            trainNum += y.size(0)
            trainCorrect += (pred == y).sum().item()
        print(f"train mean loss: {trainLoss / trainNum :.4g}, accuracy: {100 * trainCorrect / trainNum :.2f}%")
        
        # 统计在测试集上的表现
        model.eval()
        testLoss = 0.
        testNum = 0
        testCorrect = 0
        with torch.no_grad():
            for x, y in tqdm(testLoader, leave=False, desc="testing"):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = lossFunc(out, y)
                # 统计损失值及正确率
                testLoss += loss.item()
                _, pred = out.max(dim=1)
                testNum += y.size(0)
                testCorrect += (pred == y).sum().item()
        print(f"test mean loss: {testLoss / testNum :.4g}, accuracy: {100 * testCorrect / testNum :.2f}%")
        
        # 检查测试集准确率是否得到提高，提高则保存检查点
        ACC = 100 * testCorrect / testNum
        if ACC > bestACC:
            print("==> saving checkpoint...")
            state = {
                "model": model.state_dict(),
                "ACC": ACC,
                "epoch": epoch,
                "catIDs": dataHelper.catIDs
            }
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state, "checkpoint/ckpt.pth")
            bestACC = ACC

if __name__ == "__main__":
    main() 
