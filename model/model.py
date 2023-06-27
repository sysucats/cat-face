import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from pytorch_lightning import LightningModule
import torchmetrics
from typing import Tuple

class CatFaceModule(LightningModule):
    def __init__(self, num_classes: int, lr: float):
        super(CatFaceModule, self).__init__()

        self.save_hyperparameters()

        self.net = models.densenet121(num_classes=num_classes)
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.LongTensor], batch_idx: int) -> torch.Tensor:
        loss, acc = self.do_step(batch)

        self.log('train/loss', loss, on_step=True, on_epoch=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx: int):
        loss, acc = self.do_step(batch)

        self.log('val/loss', loss, on_step=False, on_epoch=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True)
    
    def do_step(self, batch: Tuple[torch.Tensor, torch.LongTensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # shape: x (B, C, H, W), y (B), w (B)
        x, y = batch

        # shape: out (B, num_classes)
        out = self.net(x)

        loss = self.loss_func(out, y)

        with torch.no_grad():
            # 每个类别分别计算准确率，以平衡地综合考虑每只猫的准确率
            accuracy_per_class = torchmetrics.functional.accuracy(out, y, task="multiclass", num_classes=self.hparams['num_classes'], average=None)
            # 去掉batch中没有出现的类别，这些位置为nan
            nan_mask = accuracy_per_class.isnan()
            accuracy_per_class = accuracy_per_class.masked_fill(nan_mask, 0)
            # 剩下的位置取均值
            acc = accuracy_per_class.sum() / (~nan_mask).sum()
        
        return loss, acc

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.Adam(self.parameters(), lr=self.hparams['lr'])