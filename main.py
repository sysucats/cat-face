import argparse
import torch
from model.model import CatFaceModule
from model.data import CatPhotoDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import json

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Cat Recognize Model Trainer")
    parser.add_argument("--data", default="data/crop_photos", type=str, help="photo data directory (default: data/crop_photos)")
    parser.add_argument("--size", default=128, type=int, help="image size (default: 128)")
    parser.add_argument("--filter", default=10, type=int, help="cats whose number of photos is less than this value will be filtered (default: 10)")
    parser.add_argument("--balance", default=30, type=int, help="data sampling number of each cat in an epoch for balancing (default: 50)")
    parser.add_argument("--lr", default=3e-4, type=float, help="learning rate (default: 3e-4)")
    parser.add_argument("--batch", default=32, type=int, help="batch size (default: 32)")
    parser.add_argument("--epoch", default=150, type=int, help="number of epoches to run (default: 150)")
    parser.add_argument("--name", default='cat', type=str, help="model name (default: cat)")
    args = parser.parse_args()

    # 加载数据情况，并分割数据
    data_module = CatPhotoDataModule(args.data, args.size, args.filter, args.balance, args.batch)

    # 创建模型
    model = CatFaceModule(len(data_module.cat_ids), args.lr)

    # 判断 GPU 是否可用
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    print(f'training device: {device}')

    # 训练模型
    logger = TensorBoardLogger('./', version=args.name, default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', filename=args.name, monitor='val/acc', mode='max')
    trainer = Trainer(
        accelerator=device,
        devices=1 if device == 'gpu' else None, # GPU 避免多卡，有 bug
        logger=logger,
        callbacks=[checkpoint_callback],
        max_epochs=args.epoch
    )
    trainer.fit(model, datamodule=data_module)

    print('exporting model...')

    if not os.path.exists('export/'):
        os.mkdir('export/')

    # 从检查点载入最佳模型
    model = CatFaceModule.load_from_checkpoint(f'checkpoints/{args.name}.ckpt')
    # 导出模型到 ONNX 文件
    model.to_onnx(f'export/{args.name}.onnx', torch.randn(1, 3, args.size, args.size), export_params=True)
    
    # 保存模型使用的 cat id 映射顺序
    with open(f'export/{args.name}.json', 'w') as fp:
        json.dump(data_module.cat_ids, fp)
    
    print('done.')

if __name__ == '__main__':
    main()