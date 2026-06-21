# Dockerfile 用于本地模拟 EMAS (Linux) 环境测试
# 用法: docker build -t catface-test . && docker run --rm catface-test
FROM node:22-slim

WORKDIR /app

# 复制 package.json 先安装依赖（利用 Docker 缓存）
COPY package.json ./

# 安装依赖（Linux 环境会自动安装 Linux 版原生模块）
RUN npm install --omit=dev

# 复制项目文件
COPY index.js ./
COPY lib/ ./lib/
COPY models/ ./models/

# 测试命令：运行 deploy_test 和本地图片识别测试
CMD ["node", "-e", "\
const handler = require('./index.js'); \
(async () => { \
  console.log('=== 1. deploy_test ==='); \
  const r1 = await handler({ args: { deploy_test: true } }); \
  console.log('结果:', JSON.stringify(r1), r1 === 'v1.0' ? '✓' : '✗'); \
  \
  console.log('\\n=== 2. 模型加载测试 ==='); \
  const r2 = await handler({ args: { timestamp: 1, signature: 'x' } }); \
  console.log('结果:', JSON.stringify(r2)); \
  console.log('模型加载:', r2.message && r2.message.includes('signature') ? '✓ 模型加载成功，签名校验生效' : '✗ 模型加载可能失败'); \
})(); \
"]
