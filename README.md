# 中大猫谱猫脸识别

从猫谱小程序云存储拉取猫图，然后训练一个神经网络模型来识别猫猫。

## 前期准备

1. 每只猫猫积攒了超过10张高质量照片（照片清晰，且照片中只有一只猫猫）。
2. 虽然可以使用CPU进行神经网络的训练，但是速度较慢，因此最好准备一张足够强大的GPU，显存应不小于10G（如果显存不够，你可以根据你的显存大小适当调整图像分辨率和batch size，但是可能会导致模型训练效果下降）。此外，CPU内存也需要有足够的空间（取决于你的小程序已经积累的图片数量）。
3. 网络服务器一台，用于部署识别服务（需要自备web后端技能）。
4. 已注册备案域名一个，获取SSL，用于小程序调用服务（非必需，但中国大陆的服务器未备案域名不能使用默认的443端口提供https服务）。

## Quick Start

1. 进入fetch-data目录，执行`npm install`安装依赖。
2. 在fetch-data目录创建.env文件，填写小程序云环境的`SECRET_ID`、`SECRET_KEY`和环境名称`ENV`。示例：
    ```bash
    SECRET_ID=abcd
    SECRET_KEY=xyz
    ENV=opq
    ```
3. 执行`npm start`，脚本将根据小程序数据库记录拉取小程序云存储中的图片。
4. 返回仓库根目录，执行`pip install -r requirements.txt`安装依赖。（需要Python>=3.6。你也可以使用`conda`。）
5. 执行`bash prepare_yolo.sh`拉取YOLOv5目标检测模型所需的代码和数据。
6. 执行`python3 data_preprocess.py`，脚本将使用YOLOv5从fetch-data拉取的图片中识别出猫猫并截取到crop-photos目录。
7. 运行train.py，训练一个识别猫猫图片的模型，并把训练输出checkpoint/ckpt.pth移动到当前目录下命名为cat.pth。同样，训练一个全图识别的模型（这个模型将在YOLO无法检测到猫猫时使用），把训练输出checkpoint/ckpt.pth移动到当前目录下命名为fallback.pth。（程序参数可以通过`python trian.py --help`获取帮助。）
8. 运行export.py，将cat.pth和fallback.pth分别导出成ONNX模型。对应的文件会放在export目录下。（程序参数可以通过`python export.py --help`获取帮助。）
9. 在仓库根目录中创建.env文件，填写服务运行参数主机名`HOST_NAME`、端口`PORT`和接口密钥`SECRET_KEY`。示例：
    ```bash
    HOST_NAME=127.0.0.1
    PORT=3456
    SECRET_KEY='keyboard cat'
    ```
10. 现在，执行`python3 app.py`，HTTP接口服务将被启动。🎉
