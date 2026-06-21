# cat-face node-server

猫脸识别后端，Node.js 22 实现。从 Python Flask 版迁移而来，可部署到阿里云 EMAS Serverless。

两阶段识别：YOLO11 检测猫脸 → ONNX 分类模型识别具体哪只猫。

## 环境准备

需要 Node.js >= 22。

```bash
cd node-server
npm install
npm install --os=linux --cpu=x64 sharp   # 装 Linux 平台 sharp，为部署 EMAS 准备
```

### 准备模型文件

`models/` 目录需要 5 个文件（被 gitignore，需自行生成）：

| 文件 | 来源 |
|------|------|
| `models/yolo11m.onnx` | 从根目录 `yolo11m.pt` 导出（见下方命令） |
| `models/cat.onnx` | 从训练流程 `export/cat.onnx` 复制 |
| `models/cat.json` | 从训练流程 `export/cat.json` 复制 |
| `models/fallback.onnx` | 从训练流程 `export/fallback.onnx` 复制 |
| `models/fallback.json` | 从训练流程 `export/fallback.json` 复制 |

导出 yolo11m.onnx（需 Python + ultralytics）：

```bash
cd ..   # 回到项目根目录
python -c "from ultralytics import YOLO; YOLO('yolo11m.pt').export(format='onnx', imgsz=640, opset=12)"
cp yolo11m.onnx node-server/models/
```

### 配置 SECRET_KEY

创建 `node-server/env` 文件（被 gitignore，需自行创建）：

```
SECRET_KEY=你的密钥
```

值需与 EMAS 数据库 `app_secret` 集合中的 `SECRET_KEY` 一致。

## 本地测试

```bash
# 启动本地 HTTP 服务
npm start

# 另开终端测试
node test/test.js http <图片路径>      # 测 HTTP 服务
node test/test.js emas <图片路径>      # 测 EMAS 云函数入口（本地直接调用 index.js）
node test/test.js deploy              # 冒烟测试（返回 "v1.0"）
node test/test.js batch <图片目录> 5  # 批量测试
```

### Docker 模拟 EMAS Linux 环境

上传 EMAS 前建议先用 Docker 验证 Linux 二进制能否正常加载（sharp / onnxruntime-node）：

```bash
node test/test-docker.js          # deploy_test + 模型加载测试
node test/test-docker.js --full   # 完整识别测试（挂载测试图片）
```

需先启动 Docker Desktop。

## 打包部署

```bash
node pack-emas.js --trim
```

生成 `../catFace.zip`（约 176MB），含 Linux 平台二进制，裁剪掉 win32/darwin。

上传 `catFace.zip` 到 EMAS Serverless 平台即可。

## 文件结构

```
node-server/
├── index.js          EMAS 云函数入口
├── server.js         本地 HTTP 服务入口（express）
├── lib/              核心逻辑（config/models/signature/image/yoloPost/recognize）
├── pack-emas.js      打包脚本
├── Dockerfile        Docker Linux 测试
├── test/             测试脚本
├── models/           模型文件（gitignore）
├── env               配置（gitignore）
└── emas-test-requests.txt   EMAS 网页测试请求模板
```

## EMAS 云函数入参

`ctx.args` 字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `photoBase64` 或 `photoURL` | string | 图片（二选一，base64 适合小图，URL 适合大图） |
| `timestamp` | number | 秒级时间戳 |
| `signature` | string | `SHA256(base64(图片) + str(timestamp) + SECRET_KEY)` |
| `catIdx` | number? | 可选，选择第几个猫框，默认 0 |
| `deploy_test` | true | 冒烟测试，返回 `"v1.0"` |

返回：`{ ok: true, message: 'OK', data: { catBoxes, recognizeResults } }`
