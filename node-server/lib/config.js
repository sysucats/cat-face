const path = require('path');

/**
 * 服务配置（纯常量，不再读取 env 文件）
 *
 * ⚠️ SECRET_KEY 不在此处配置。
 *   - 生产环境（EMAS）：运行时从数据库 app_secret 集合读取，见 index.js 的 getSecretKey()
 *   - 本地测试：通过 env 文件 + 环境变量提供，见 test/test-config.js
 */
const config = {
  // HTTP 服务监听（仅本地 server.js 用，EMAS 云函数不使用）
  HOST_NAME: '0.0.0.0',
  PORT: 5000,

  // 签名时间戳容忍误差（秒）
  TOLERANT_TIME_ERROR: 60,

  // 猫脸识别模型输入尺寸
  IMG_SIZE: 256,
  FALLBACK_IMG_SIZE: 512,

  // 返回结果数量上限
  CAT_BOX_MAX_RET_NUM: 5,
  RECOGNIZE_MAX_RET_NUM: 20,

  // 模型文件路径
  MODEL_PATHS: {
    yolo: path.join(__dirname, '../models/yolo11m.onnx'),
    cat: path.join(__dirname, '../models/cat.onnx'),
    catJson: path.join(__dirname, '../models/cat.json'),
    fallback: path.join(__dirname, '../models/fallback.onnx'),
    fallbackJson: path.join(__dirname, '../models/fallback.json'),
  },
};

module.exports = { config };
