# 中大猫谱猫脸识别

从猫谱小程序云存储拉取猫图，然后训练一个神经网络模型来识别猫猫。

## 前期准备

1. 每只猫猫积攒了超过10张高质量照片（照片清晰，且照片中只有一只猫猫）。
2. 虽然可以使用CPU进行神经网络的训练，但是速度较慢，因此最好准备一张足够强大的GPU，显存应不小于10G（如果显存不够，你可以根据你的显存大小适当调整图像分辨率和batch size，但是可能会导致模型训练效果下降）。此外，CPU内存也需要有足够的空间（取决于你的小程序已经积累的图片数量）。
3. 网络服务器一台，用于部署识别服务（需要自备web后端技能）。
4. 已注册备案域名一个，获取SSL，用于小程序调用服务（非必需，但中国大陆的服务器未备案域名不能使用默认的443端口提供https服务）。

## Quick Start

1. NPM安装fetch-data文件夹中JS脚本的依赖。
2. 使用Node.js运行fetch-data中的fetch.js，根据小程序数据库记录拉取小程序云存储中的图片。（记得填写`tcb.init`所需的信息。）
3. PIP安装requirements.txt中的Python依赖。（需要Python>=3.6。）
4. \[可选\]执行prepare_yolo.sh脚本准备YOLOv5目标检测模型所需的内容。（如在脚本运行上遇到问题，你也可以手动使用git拉取子模块的内容。剩余的下载步骤可以在下面运行时自动完成。）
5. 运行data_preprocess.py程序，使用YOLOv5从fetch-data拉取的数据中识别出猫猫并截取到crop-photos文件夹。
6. 多次运行train.py，训练一个识别猫猫图片的模型，并把训练输出checkpoint/ckpt.pth移动到当前目录下命名为cat.pth。同样，训练一个全图识别的模型（这个模型将在YOLO无法检测到猫猫时使用），把训练输出checkpoint/ckpt.pth移动到当前目录下命名为fallback.pth。（程序参数可以通过`python trian.py --help`获取帮助。）
7. 两次运行export.py，将cat.pth和fallback.pth分别导出成ONNX模型。对应的文件会放在export目录下。（程序参数可以通过`python export.py --help`获取帮助。）
8. 修改app.py中必要的常量。现在你可以通过运行app.py启动后端服务了。
