from typing import List, Tuple

import os
import math
import random
import pyarrow as pa
import numpy as np
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class CatPhotoDataset(Dataset):
    def __init__(self, data: List[Tuple[str, int, float]], size: int, num_classes: int, balance_num: int, augmentation: bool = False):
        self.data = pa.array(data) # 使用pyarrow避免pytorch dataloader worker多份拷贝
        self.size = size
        self.num_classes = num_classes
        self.balance_num = balance_num

        if augmentation:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
                transforms.RandomVerticalFlip(p=0.5), # 随机垂直翻转
                transforms.RandomCrop(size, padding=size // 8, fill=0), # 四周填充1/8大小后再随机裁剪为(size,size)
                transforms.RandomRotation(degrees=15, fill=0) # 随机旋转-15~15度
            ])
        else:
            self.transform = None
    
    def __len__(self) -> int:
        return self.num_classes * self.balance_num

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        class_idx = index // self.balance_num

        class_data = self.data[class_idx].as_py()
        
        # TODO:数据库里有打不开的错误图片。。暂时这样打个补丁吧
        while True:
            try:
                img_path = random.choice(class_data)
                img = Image.open(img_path).convert('RGB').resize((self.size, self.size))
                break
            except:
                continue
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1)) / 255
        img = Tensor(img)

        return self.transform(img) if self.transform != None else img, class_idx

class CatPhotoDataModule(LightningDataModule):
    def __init__(self, root: str, size: int, filter_num: int, balance_num: int, batch_size: int):
        super(CatPhotoDataModule, self).__init__()

        self.save_hyperparameters()

        cat_photos = {}
        for dir_name in os.listdir(root):
            dir_path = os.path.join(root, dir_name)
            if os.path.isdir(dir_path):
                cat_photos[dir_name] = []
                for file_name in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file_name)
                    cat_photos[dir_name].append(file_path)
                cat_photos[dir_name].sort() # 排序，使多次运行时对训练集和测试集的划分保持一致
        
        cat_ids = list(cat_photos.keys())
        print(f'total number of cats: {len(cat_ids)}')
        
        # 过滤照片少于filter_num张的猫猫
        cat_with_enough_photo_ids = [cat_id for cat_id in cat_ids if len(cat_photos[cat_id]) >= filter_num]
        cat_with_enough_photo_ids.sort() # 排序，使多次运行时ID对应的索引保持一致
        print(f'number of cats with enough photos: {len(cat_with_enough_photo_ids)}, other cats would be ignored')
        self.cat_ids = cat_with_enough_photo_ids
        
        train_data = [None for cat_id in self.cat_ids]
        val_data = [None for cat_id in self.cat_ids]

        for cat_idx, cat_id in enumerate(self.cat_ids):
            photos = cat_photos[cat_id]

            num_photos = len(photos)
            # 80% 作为训练集
            num_train = math.floor(num_photos * 0.8)

            train_data[cat_idx] = photos[:num_train]
            val_data[cat_idx] = photos[num_train:]
        
        self.train_dataset = CatPhotoDataset(train_data, size, len(self.cat_ids), balance_num, augmentation=True)
        self.val_dataset = CatPhotoDataset(val_data, size, len(self.cat_ids), math.ceil(balance_num * 0.2))
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=4)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.hparams['batch_size'], shuffle=False, num_workers=4)