/**
 * 生成可直接粘贴到 EMAS 网页测试框的 JSON（photoBase64 模式）
 *
 * 读取本地图片 → 计算 photoBase64 + 当前时间戳 + 合法签名 → 写入 JSON 文件
 * 签名算法：SHA256( photoBase64 + str(timestamp) + SECRET_KEY )，与 lib/signature.js 对齐
 *
 * 用法:
 *   node test/gen-emas-request.js <图片路径> [-o 输出文件]
 *
 * 说明：
 * - EMAS 测试框会把整个 JSON 作为 ctx.args 传给云函数，所以这里直接写 args 内容，
 *   不再外包一层 "args"（与 emas-test-requests.txt 模板保持一致）。
 * - JSON 写入文件（默认 node-server/test/emas-request.json），不打印到 stdout。
 * - 诊断信息走 stderr。
 * - ⚠️ 生成的 JSON 受 TOLERANT_TIME_ERROR（默认 60 秒）限制，需尽快粘贴测试。
 */
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { config } = require('../lib/config.js');
const { SECRET_KEY } = require('./test-config.js');

const DEFAULT_OUTPUT = path.join(__dirname, 'emas-request.json');

/**
 * 计算签名，与 lib/signature.js 的 checkSignature 逐字节对齐
 * @param {string} photoBase64 - 图片原始字节的 base64
 * @param {number} timestamp - 秒级时间戳
 * @param {string} secretKey - 与 EMAS 线上一致的密钥
 * @returns {string} 64 位 hex 签名
 */
function makeSignature(photoBase64, timestamp, secretKey) {
  const sigData = photoBase64 + String(timestamp) + secretKey;
  return crypto.createHash('sha256').update(sigData, 'utf8').digest('hex');
}

/**
 * 解析命令行参数
 * @returns {{ photoPath: string, outputPath: string }}
 */
function parseArgs() {
  const argv = process.argv.slice(2);
  let photoPath = null;
  let outputPath = DEFAULT_OUTPUT;

  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '-o' || a === '--output') {
      outputPath = argv[++i];
      if (!outputPath) {
        console.error('错误: -o 需要跟一个文件路径');
        process.exit(1);
      }
    } else if (a === '-h' || a === '--help') {
      console.error('用法: node test/gen-emas-request.js <图片路径> [-o 输出文件]');
      console.error('示例: node test/gen-emas-request.js ../data/sample.jpg');
      console.error('      node test/gen-emas-request.js ../data/sample.jpg -o req.json');
      console.error('默认输出: ' + DEFAULT_OUTPUT);
      process.exit(0);
    } else if (!photoPath) {
      photoPath = a;
    } else {
      console.error('错误: 未知参数 ' + a);
      process.exit(1);
    }
  }

  if (!photoPath) {
    console.error('用法: node test/gen-emas-request.js <图片路径> [-o 输出文件]');
    console.error('示例: node test/gen-emas-request.js ../data/sample.jpg');
    process.exit(1);
  }

  if (!fs.existsSync(photoPath)) {
    console.error('错误: 图片不存在: ' + photoPath);
    process.exit(1);
  }

  return { photoPath, outputPath };
}

function main() {
  const { photoPath, outputPath } = parseArgs();

  // 1. 读取图片字节
  const photoBuffer = fs.readFileSync(photoPath);
  const photoBase64 = photoBuffer.toString('base64');

  // 2. 当前时间戳（秒）
  const timestamp = Math.floor(new Date().getTime() / 1000);

  // 3. 用 SECRET_KEY 计算合法签名
  const signature = makeSignature(photoBase64, timestamp, SECRET_KEY);

  // 4. 组装 EMAS 测试框 JSON（直接是 args 内容，不再外包 args）
  const payload = { photoBase64, timestamp, signature };

  // 5. 写入文件
  const jsonStr = JSON.stringify(payload, null, 2);
  fs.writeFileSync(outputPath, jsonStr, 'utf8');
  const fileSize = fs.statSync(outputPath).size;

  // —— 诊断信息 → stderr ——
  console.error('图片路径:      ' + photoPath);
  console.error('图片大小:      ' + photoBuffer.length + ' bytes');
  console.error('photoBase64:   ' + photoBase64.length + ' 字符');
  if (!SECRET_KEY || SECRET_KEY === 'please_change_me') {
    console.error('SECRET_KEY:    ⚠️ 为空或占位符 (please_change_me)，请确认与 EMAS 数据库 app_secret 集合中的值一致');
  } else {
    console.error('SECRET_KEY:    已配置 (长度 ' + SECRET_KEY.length + ')');
  }
  console.error('timestamp:     ' + timestamp + ' (' + new Date(timestamp * 1000).toISOString() + ')');
  console.error('签名:          ' + signature);
  console.error('JSON 文件:     ' + outputPath + ' (' + fileSize + ' bytes)');
  console.error('⚠️ 此 JSON 有效期仅 ' + config.TOLERANT_TIME_ERROR + ' 秒，请尽快粘贴到 EMAS 测试框');
}

main();
