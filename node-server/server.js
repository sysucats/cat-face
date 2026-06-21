const path = require('path');
const express = require('express');
const multer = require('multer');
require('dotenv').config({ path: path.join(__dirname, 'env') });
const { config } = require('./lib/config');
const { loadModels } = require('./lib/models');
const { recognize } = require('./lib/recognize');

// 本地 HTTP 服务的 SECRET_KEY 从 env 文件读取
// （EMAS 云函数的 SECRET_KEY 从数据库读取，不走此路径）
const SECRET_KEY = process.env.SECRET_KEY || '';

// multer: 内存存储，单文件最大 10MB（与 Python MAX_CONTENT_LENGTH 一致）
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 },
});

const app = express();

// 请求处理函数（同时注册带/不带尾斜杠两个路由）
async function handler(req, res) {
  try {
    if (!req.file) throw new Error('photo field missing');
    const result = await recognize(req.file.buffer, req.body, SECRET_KEY);
    res.json({ ok: true, message: 'OK', data: result });
  } catch (err) {
    res.json({ ok: false, message: String(err.message || err), data: null });
  }
}

app.post('/recognizeCatPhoto', upload.single('photo'), handler);
app.post('/recognizeCatPhoto/', upload.single('photo'), handler);

// 启动：先加载模型，再监听端口
(async () => {
  await loadModels();
  if (!SECRET_KEY) {
    console.log('⚠️  SECRET_KEY 未配置（env 文件中为空），签名校验将全部失败');
  }
  app.listen(config.PORT, config.HOST_NAME, () => {
    console.log(`==> server running at http://${config.HOST_NAME}:${config.PORT}`);
  });
})();
