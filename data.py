from typing import List, Any, Tuple

import os
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

class CatPhotoDatasetHelper:
    def __init__(self, root: str, size: int, filterNum: int, balanceNum: int, transform: Any = None):
        catPhotos = dict()

        for dirName in os.listdir(root):
            dirPath = os.path.join(root, dirName)

            if os.path.isdir(dirPath):
                catPhotos[dirName] = []
            
                for fileName in os.listdir(dirPath):
                    filePath = os.path.join(dirPath, fileName)
                    catPhotos[dirName].append(filePath)
                
                catPhotos[dirName].sort()
        
        catIDs = [catId for catId in catPhotos]
        print(f"*** total number of cats: {len(catIDs)}")
        
        # 过滤照片少于filterNum张的猫猫
        catWithEnoughPhotoIDs = list(filter(lambda catID: len(catPhotos[catID]) >= filterNum, catIDs))
        catWithEnoughPhotoIDs.sort()
        print(f"*** number of cats with enough photos: {len(catWithEnoughPhotoIDs)}, other cats would be ignored")
        self.catIDs = catWithEnoughPhotoIDs
        
        catTrainData = []
        catTestData = []
        for catID in tqdm(self.catIDs, leave=False, desc="loading"):
            photos = catPhotos[catID]

            numPhotos = len(photos)
            # 80% 作为训练集
            numTrain = int(numPhotos * 0.8)

            trainData = []
            testData = []

            subProcessBar = tqdm(total=numPhotos, leave=False, desc=catID)

            for i in range(0, numTrain):
                # 等比放缩然后使用padding
                srcImg = Image.open(photos[i]).convert("RGB")
                ratio = size / max(srcImg.width, srcImg.height)
                unpadSize = int(round(srcImg.width * ratio)), int(round(srcImg.height * ratio))
                srcImg = srcImg.resize(unpadSize)
                dw = (size - unpadSize[0]) / 2
                dh = (size - unpadSize[1]) / 2
                left = int(round(dw - 0.1))
                top = int(round(dh - 0.1))
                padImg = Image.new(mode="RGB", size=(size, size), color=(114, 114, 114))
                padImg.paste(srcImg, box=(left, top))
                imgData = np.array(padImg, dtype=np.float32).transpose((2, 0, 1)) / 255
                trainData.append(Tensor(imgData))
                subProcessBar.update(1)
            
            for i in range(numTrain, numPhotos):
                # 等比放缩然后使用padding
                srcImg = Image.open(photos[i]).convert("RGB")
                ratio = size / max(srcImg.width, srcImg.height)
                unpadSize = int(round(srcImg.width * ratio)), int(round(srcImg.height * ratio))
                srcImg = srcImg.resize(unpadSize)
                dw = (size - unpadSize[0]) / 2
                dh = (size - unpadSize[1]) / 2
                left = int(round(dw - 0.1))
                top = int(round(dh - 0.1))
                padImg = Image.new(mode="RGB", size=(size, size), color=(114, 114, 114))
                padImg.paste(srcImg, box=(left, top))
                imgData = np.array(padImg, dtype=np.float32).transpose((2, 0, 1)) / 255
                testData.append(Tensor(imgData))
                subProcessBar.update(1)

            subProcessBar.close()

            catTrainData.append(trainData)
            catTestData.append(testData)
        
        self.trainDataset = CatPhotoDataset(targetData=catTrainData, balanceNum=balanceNum, transform=transform)
        # self.testDataset = CatPhotoDataset(targetData=catTestData)
        self.testDataset = CatPhotoDataset(targetData=catTestData, balanceNum=int(balanceNum * 0.2))

class CatPhotoDataset(Dataset):
    def __init__(self, targetData: List[List[Tensor]], balanceNum: int = 0, transform: Any = None):
        self.balanceNum = balanceNum
        self.transform = transform

        # 当balanceNum不为0时，使用采样的方式对数据集分类样本数进行平衡，每个分类的样本数均伪装成balanceNum
        if self.balanceNum > 0:
            self.targetData = targetData
            # 使用choiceRange记录本次训练中尚未使用过的样本，尽可能避免重复
            self.targetChoiceRange = [list(range(len(data))) for data in self.targetData]
        else:
            self.data = [datum for data in targetData for datum in data]
            self.labels = [index for (index, data) in enumerate(targetData) for _ in data]
    
    def __len__(self) -> int:
        if self.balanceNum > 0:
            return len(self.targetData) * self.balanceNum
        else:
            return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        if self.balanceNum > 0:
            # 计算index所属分类
            target = index // self.balanceNum

            # 在所属分类中抽取一张
            choiceRange = self.targetChoiceRange[target]
            if len(choiceRange) == 0:
                choiceRange = list(range(len(self.targetData[target])))
            dataIdx = np.random.choice(choiceRange)
            choiceRange.remove(dataIdx)

            img = self.targetData[target][dataIdx]

            if self.transform is not None:
                img = self.transform(img)

            return img, target
        else:
            img, target = self.data[index], self.labels[index]

            if self.transform is not None:
                img = self.transform(img)
            
            return img, target
    
    def reset(self):
        if self.balanceNum > 0:
            self.targetChoiceRange = [list(range(len(data))) for data in self.targetImgData]