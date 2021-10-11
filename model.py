import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, inPlanes: int, outPlanes: int, stride: int, groupPlanes: int,
        convShortcut: bool):

        super(Block, self).__init__()
        hiddenPlanes = outPlanes
        # 1x1卷积核
        self.conv1 = nn.Conv2d(inPlanes, hiddenPlanes, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hiddenPlanes)
        # 3x3卷积核
        groups = hiddenPlanes // groupPlanes
        self.conv2 = nn.Conv2d(hiddenPlanes, hiddenPlanes, kernel_size=3, stride=stride,
            padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(hiddenPlanes)
        # 1x1卷积核
        self.conv3 = nn.Conv2d(hiddenPlanes, outPlanes, kernel_size=1, stride=1,
            padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(outPlanes)

        if convShortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inPlanes, outPlanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outPlanes)
            )
        else:
            self.shortcut = nn.Sequential()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)) + self.shortcut(x))
        return out

class ResNet(nn.Module):
    def __init__(self, numTargets: int):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.lastPlane = 64

        self.depths = [1, 1, 4, 7]
        self.planes = [24, 56, 152, 368]
        self.strides = [1, 1, 2, 2]
        self.groupPlanes = 8

        self.layer1 = self.makeLayer(0)
        self.layer2 = self.makeLayer(1)
        self.layer3 = self.makeLayer(2)
        self.layer4 = self.makeLayer(3)
        self.linear = nn.Linear(self.planes[-1], numTargets)
    
    def makeLayer(self, index: int) -> nn.Module:
        depth = self.depths[index]
        plane = self.planes[index]
        stride = self.strides[index]
        groupPlanes = self.groupPlanes

        layers = []
        for i in range(depth):
            s = stride if i == 0 else 1
            layers.append(
                Block(inPlanes=self.lastPlane, outPlanes=plane, stride=s, groupPlanes=groupPlanes, convShortcut=(i == 0))
            )
            self.lastPlane = plane
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
