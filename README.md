# 中大猫谱猫脸识别

从猫谱小程序云存储拉取猫图，然后训练一个神经网络模型来识别猫猫。

## Quick Start

1. NPM安装fetch-data文件夹中JS脚本的依赖。
2. 使用Node.js运行fetch-data中的fetch.js，根据小程序数据库记录拉取小程序云存储中的图片。（记得填写`tcb.init`所需的信息。）
3. PIP安装requirements.txt中的Python依赖。（需要Python>=3.6。使用CONDA管理环境是更好地选择。）
4. 执行prepare_yolo.sh脚本准备YOLOv5目标检测模型所需的内容。
5. 运行data_preprocess.py程序，使用YOLOv5从fetch-data拉取的数据中识别出猫猫并截取到crop-photos文件夹。
6. 多次运行train.py，训练一个识别截取猫猫图片的模型（并把训练输出checkpoint/ckpt.pth移动到当前目录下命名为cat.pth）和一个全图识别的模型（并把训练输出checkpoint/ckpt.pth移动到当前目录下命名为fallback.pth）。（程序参数可以通过`python trian.py --help`获取帮助。你可能需要一张足够强大的GPU。）
7. 两次运行export.py，将cat.pth和fallback.pth分别导出成ONNX模型。对应的文件会放在export目录下。（程序参数可以通过`python export.py --help`获取帮助。）
8. 修改app.py中必要的常量。现在你可以通过运行app.py启动后端服务了。