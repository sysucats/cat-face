from typing import List, Any, Tuple

import os
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm

class CatPhotoDatasetHelper:
    def __init__(self, root: str, size: int, filterNum: int, transform: Any = None):
        catPhotos = {}
        for dirName in os.listdir(root):
            dirPath = os.path.join(root, dirName)
            if os.path.isdir(dirPath):
                catPhotos[dirName] = []
                for fileName in os.listdir(dirPath):
                    filePath = os.path.join(dirPath, fileName)
                    catPhotos[dirName].append(filePath)
                catPhotos[dirName].sort() # 排序，使多次运行时对训练集和测试集的划分保持一致
        
        catIDs = list(catPhotos.keys())
        print(f"*** total number of cats: {len(catIDs)}")
        
        # 过滤照片少于filterNum张的猫猫
        catWithEnoughPhotoIDs = list(filter(lambda catID: len(catPhotos[catID]) >= filterNum, catIDs))
        catWithEnoughPhotoIDs.sort() # 排序，使多次运行时ID对应的索引保持一致
        print(f"*** number of cats with enough photos: {len(catWithEnoughPhotoIDs)}, other cats would be ignored")
        self.catIDs = catWithEnoughPhotoIDs

        # 按照片数量反比计算猫猫数据权重，减少不平衡数据集对模型倾向的影响
        reciprocalSum = np.sum(list(map(lambda catID: 1 / len(catPhotos[catID]), self.catIDs)))
        self.catWeight = {catID: (1 / len(catPhotos[catID])) / reciprocalSum for catID in self.catIDs}
        
        trainData = []
        testData = []
        for catIdx, catID in tqdm(list(enumerate(self.catIDs)), leave=False, desc="loading"):
            photos = catPhotos[catID]
            weight = self.catWeight[catID]

            numPhotos = len(photos)
            # 80% 作为训练集
            numTrain = int(numPhotos * 0.8)

            for i in tqdm(range(0, numPhotos), leave=False, desc=catID):
                # 直接resize到正方形
                srcImg = Image.open(photos[i]).convert("RGB").resize((size, size))
                imgData = np.array(srcImg, dtype=np.float32).transpose((2, 0, 1)) / 255
                (trainData if i < numTrain else testData).append((Tensor(imgData), catIdx, weight))
        
        self.trainDataset = CatPhotoDataset(data=trainData, transform=transform)
        self.testDataset = CatPhotoDataset(data=testData)

class CatPhotoDataset(Dataset):
    def __init__(self, data: List[Tuple[Tensor, int, float]], transform: Any = None):
        self.transform = transform
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        img, catIdx, weight = self.data[index]
        return self.transform(img) if self.transform is not None else img, catIdx, weight