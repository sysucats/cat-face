/**
 * 本地测试 EMAS 云函数入口 index.js
 * 模拟 EMAS ctx 结构，直接调用 index.js
 *
 * 用法:
 *   node test/test-emas.js <图片路径>              # 用图片文件模拟 photoBase64
 *   node test/test-emas.js <图片路径> --url        # 启动本地文件服务模拟 photoURL
 *
 * 说明：本地测试用 test-config.js 的 SECRET_KEY 模拟 EMAS 数据库 app_secret 集合的返回值，
 *       完整测试 index.js 的 getSecretKey → recognize 流程。
 */
const fs = require('fs');
const http = require('http');
const crypto = require('crypto');

const handler = require('../index.js');
const { SECRET_KEY } = require('./test-config.js');

// 用真实 SECRET_KEY 计算合法签名，便于测试通过签名校验
function makeSignature(photoBuffer, timestamp, secretKey) {
  const b64 = photoBuffer.toString('base64');
  const sigData = b64 + String(timestamp) + secretKey;
  return crypto.createHash('sha256').update(sigData, 'utf8').digest('hex');
}

/**
 * 构造模拟 EMAS ctx
 * - mpserverless.db: 模拟 app_secret 集合查询，返回本地 SECRET_KEY
 * - httpclient: 模拟 EMAS 内置 HTTP 客户端（用 Node 内置 http）
 */
function buildMockCtx(overrides = {}) {
  return {
    ...overrides,
    mpserverless: {
      db: {
        collection(name) {
          return {
            findOne: async () => ({ result: { SECRET_KEY } }),
          };
        },
      },
    },
  };
}

async function main() {
  const photoPath = process.argv[2];
  const useUrl = process.argv[3] === '--url';

  if (!photoPath) {
    console.error('用法: node test/test-emas.js <图片路径> [--url]');
    process.exit(1);
  }

  const photoBuffer = fs.readFileSync(photoPath);
  console.log('图片大小:', photoBuffer.length, 'bytes');
  console.log('SECRET_KEY:', SECRET_KEY ? `(已配置, 长度${SECRET_KEY.length})` : '(空! 请先在 env 中设置)');

  // 用真实密钥生成合法签名
  const timestamp = Math.floor(Date.now() / 1000);
  const signature = makeSignature(photoBuffer, timestamp, SECRET_KEY);

  if (useUrl) {
    // 启动本地 HTTP 服务托管图片，模拟 photoURL 模式
    const server = http.createServer((req, res) => {
      res.writeHead(200, { 'Content-Type': 'image/jpeg' });
      res.end(photoBuffer);
    });
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    const port = server.address().port;
    const photoURL = `http://127.0.0.1:${port}/photo.jpg`;
    console.log('本地图片服务:', photoURL);

    // 模拟 ctx.httpclient（用 Node 内置 http）
    const ctx = buildMockCtx({
      args: { photoURL, timestamp, signature },
      httpclient: {
        request(url, options) {
          return new Promise((resolve, reject) => {
            http.get(url, (res) => {
              const chunks = [];
              res.on('data', (c) => chunks.push(c));
              res.on('end', () => resolve({
                status: res.statusCode,
                headers: res.headers,
                data: Buffer.concat(chunks),
              }));
              res.on('error', reject);
            });
          });
        },
      },
    });

    const result = await handler(ctx);
    console.log('\n=== EMAS 云函数返回 ===');
    console.log(JSON.stringify(result, null, 2));
    server.close();
  } else {
    // photoBase64 模式
    const photoBase64 = photoBuffer.toString('base64');
    const ctx = buildMockCtx({ args: { photoBase64, timestamp, signature } });

    const result = await handler(ctx);
    console.log('\n=== EMAS 云函数返回 ===');
    console.log(JSON.stringify(result, null, 2));
  }

  // 单独测试 deploy_test（不走 DB，直接返回）
  const deployResult = await handler({ args: { deploy_test: true } });
  console.log('\n=== deploy_test 返回 ===');
  console.log(deployResult);
}

main().catch((err) => {
  console.error('错误:', err);
  process.exit(1);
});
