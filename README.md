# 中大猫谱猫脸识别

从猫谱小程序云存储拉取猫图，然后训练一个神经网络模型来识别猫猫。

## 前期准备

1. 每只猫猫积攒了超过10张高质量照片（照片清晰，且照片中只有一只猫猫）。
2. 虽然可以使用CPU进行神经网络的训练，但是速度较慢，因此最好准备一张具有足够显存的GPU。（如果显存不够，你可以根据你的显存大小适当调整图像分辨率和batch size，但是可能会导致模型训练效果下降。）
3. 网络服务器一台，用于部署识别服务。（需要自备web后端技能。）服务器不需要GPU。
4. 已注册备案域名一个，获取SSL，用于小程序调用服务。

## Quick Start

### 1. 训练模型

1. 在训练服务器（或你的个人PC）上拉取本仓库代码。
2. 进入data目录，执行`npm install`安装依赖。（需要 Node.js 环境，不确定老版本 Node.js 兼容性，建议使用最新版本。）
3. 在data目录创建.env文件，填写小程序云环境的`SECRET_ID`、`SECRET_KEY`和环境名称`ENV`。示例：
    ```bash
    SECRET_ID=<your_secret_id>
    SECRET_KEY=<your_secret_key>
    ENV=<your_env>
    ```
4. 执行`npm start`，脚本将根据小程序数据库记录拉取小程序云存储中的图片。
5. 返回仓库根目录，执行`pip install -r requirements.txt`安装依赖。（需要Python>=3.8。不建议使用特别新版本的 Python，可能有兼容性问题。）
6. 执行`bash prepare_yolov5.sh`拉取YOLOv5目标检测模型所需的代码，然后下载并预处理模型数据。
7. 执行`python3 data_preprocess.py`，脚本将使用YOLOv5从data/photos的图片中识别出猫猫并截取到data/crop_photos目录。
8. 执行`python3 main.py`，使用默认参数训练一个识别猫猫图片的模型。（你可以通过`python3 main.py --help`查看帮助来自定义一些训练参数。）程序运行结束时，你应当看到目录的export文件夹下存在cat.onnx和cat.json两个文件。（训练数据使用TensorBoard记录在lightning_logs文件夹下。若要查看准确率等信息，请自行运行TensorBoard。）
9. 执行`python3 main.py --data data/photos --size 224 --name fallback`，使用修改后的参数训练一个在YOLOv5无法找到猫猫时使用的全图识别模型。程序运行结束时，你应当看到目录的export文件夹下存在fallback.onnx和fallback.json两个文件。

### 2. 部署服务

1. 在部署服务器上拉取本仓库代码。
2. 执行`pip install -r requirements.txt`安装依赖。（需要Python>=3.8。不建议使用特别新版本的 Python，可能有兼容性问题。）
3. 执行`bash prepare_yolov5.sh`拉取YOLOv5目标检测模型所需的代码，然后下载并预处理模型数据。
4. 将export文件夹从训练服务器中拷贝到部署服务器中。export文件夹中应包括四个文件：cat.onnx, cat.json, fallback.onnx, fallback.json。
5. 在仓库根目录中创建.env文件，填写服务运行参数。示例：
    ```bash
    HOST_NAME=127.0.0.1 # 主机名
    PORT=3456 # HTTP服务端口

    SECRET_KEY='keyboard cat' # 接口密钥（用于哈希签名认证）
    TOLERANT_TIME_ERROR=60 # 调用接口时附带的时间戳参数与服务器时间之间的最大允许误差（单位：s）

    IMG_SIZE=128 # 猫猫识别模型的输入图像大小
    FALLBACK_IMG_SIZE=224 # 无法检测到猫猫时使用的全图识别模型的输入图像大小

    CAT_BOX_MAX_RET_NUM=5 # 接口返回的图片中检测到的猫猫的最大个数
    RECOGNIZE_MAX_RET_NUM=20 # 接口返回的猫猫识别结果候选列表的最大个数
    ```
6. 现在，执行`python3 app.py`，HTTP接口服务将被启动。🎉（你可以自行使用gunicorn等框架完成更规范的部署。）
