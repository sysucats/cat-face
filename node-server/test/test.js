/**
 * cat-face node-server 综合测试脚本
 *
 * 用法:
 *   node test/test.js http    <图片路径> [图片路径...]    测试本地 HTTP 服务 (需先启动 server.js)
 *   node test/test.js emas    <图片路径> [图片路径...]    测试 EMAS 云函数入口 (直接调用 index.js)
 *   node test/test.js both    <图片路径> [图片路径...]    同时测试两者并对比结果
 *   node test/test.js batch   <图片目录> [数量]           批量测试目录下的图片
 *   node test/test.js deploy                             仅测试 EMAS deploy_test
 *
 * 示例:
 *   node test/test.js http ../data/photos/XJtU7JT75u2201yT/ff0bd99a-c103-4358-98c4-83ebd4d1d24c.jpg
 *   node test/test.js emas  ../data/photos/XJtU7JT75u2201yT/ff0bd99a-c103-4358-98c4-83ebd4d1d24c.jpg
 *   node test/test.js both  ../data/photos/XJtU7JT75u2201yT/ff0bd99a-c103-4358-98c4-83ebd4d1d24c.jpg
 *   node test/test.js batch ../data/photos/XJtU7JT75u2201yT 3
 *   node test/test.js deploy
 *
 * 环境变量:
 *   SERVER_URL  本地 HTTP 服务地址，默认 http://127.0.0.1:5000
 *   EMAS_URL    远程 EMAS 云函数地址（可选，留空则直接本地调用 index.js）
 */

const fs = require('fs');
const path = require('path');
const http = require('http');
const crypto = require('crypto');
const { config } = require('../lib/config.js');
const { SECRET_KEY } = require('./test-config.js');

const SERVER_URL = process.env.SERVER_URL || 'http://127.0.0.1:5000';
const EMAS_URL = process.env.EMAS_URL || ''; // 留空 = 本地直接调用 index.js

// ============ 工具函数 ============

/** 用真实 SECRET_KEY 生成合法签名 */
function makeSignature(photoBuffer, timestamp, secretKey) {
  const b64 = photoBuffer.toString('base64');
  const sigData = b64 + String(timestamp) + secretKey;
  return crypto.createHash('sha256').update(sigData, 'utf8').digest('hex');
}

/** 简单 HTTP 请求（POST multipart/form-data）*/
function httpPostMultipart(url, fields) {
  return new Promise((resolve, reject) => {
    const boundary = '----FormBoundary' + Math.random().toString(16).slice(2);
    const parts = [];

    for (const [key, value] of Object.entries(fields)) {
      if (key === 'photo' && Buffer.isBuffer(value)) {
        parts.push(
          `--${boundary}\r\n` +
          `Content-Disposition: form-data; name="photo"; filename="photo.jpg"\r\n` +
          `Content-Type: image/jpeg\r\n\r\n`
        );
        parts.push(value);
        parts.push('\r\n');
      } else {
        parts.push(
          `--${boundary}\r\n` +
          `Content-Disposition: form-data; name="${key}"\r\n\r\n` +
          `${value}\r\n`
        );
      }
    }
    parts.push(`--${boundary}--\r\n`);

    const body = Buffer.concat(parts.map((p) => (Buffer.isBuffer(p) ? p : Buffer.from(p, 'utf8'))));

    const parsed = new URL(url);
    const req = http.request(
      {
        hostname: parsed.hostname,
        port: parsed.port,
        path: parsed.pathname,
        method: 'POST',
        headers: {
          'Content-Type': `multipart/form-data; boundary=${boundary}`,
          'Content-Length': body.length,
        },
      },
      (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          try {
            resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
          } catch (e) {
            resolve({ ok: false, message: '非 JSON 响应: ' + Buffer.concat(chunks).toString().slice(0, 200), data: null });
          }
        });
      }
    );
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

/** HTTP POST JSON（用于远程 EMAS 调用）*/
function httpPostJson(url, json) {
  return new Promise((resolve, reject) => {
    const body = Buffer.from(JSON.stringify(json), 'utf8');
    const parsed = new URL(url);
    const req = http.request(
      {
        hostname: parsed.hostname,
        port: parsed.port,
        path: parsed.pathname,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': body.length,
        },
      },
      (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          try {
            resolve(JSON.parse(Buffer.concat(chunks).toString('utf8')));
          } catch (e) {
            resolve(Buffer.concat(chunks).toString());
          }
        });
      }
    );
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

/** 提取识别结果摘要 */
function summarize(result) {
  if (!result || !result.ok || !result.data) return '失败: ' + (result?.message || '无数据');
  const d = result.data;
  const top = d.recognizeResults?.[0];
  return `猫框${d.catBoxes?.length || 0}个, top1=${top?.catID || '无'} score=${top?.score?.toFixed(4) || '0'}`;
}

/** 对比两个结果是否一致 */
function compareResults(r1, r2) {
  if (!r1?.ok || !r2?.ok) return '一方失败，无法对比';
  const t1 = r1.data?.recognizeResults?.[0]?.catID;
  const t2 = r2.data?.recognizeResults?.[0]?.catID;
  if (t1 === t2) return `✓ top1 一致: ${t1}`;
  return `✗ top1 不一致: ${t1} vs ${t2}`;
}

// ============ 测试函数 ============

/** 测试本地 HTTP 服务 */
async function testHttp(photoPath) {
  const photoBuffer = fs.readFileSync(photoPath);
  const timestamp = Math.floor(Date.now() / 1000);
  const signature = makeSignature(photoBuffer, timestamp, SECRET_KEY);

  const t0 = Date.now();
  const result = await httpPostMultipart(`${SERVER_URL}/recognizeCatPhoto`, {
    photo: photoBuffer,
    timestamp: String(timestamp),
    signature,
  });
  const elapsed = Date.now() - t0;
  return { result, elapsed };
}

/** 测试 EMAS 云函数（本地直接调用或远程 HTTP）*/
async function testEmas(photoPath) {
  const photoBuffer = fs.readFileSync(photoPath);
  const timestamp = Math.floor(Date.now() / 1000);
  const signature = makeSignature(photoBuffer, timestamp, SECRET_KEY);
  const photoBase64 = photoBuffer.toString('base64');

  const t0 = Date.now();
  let result;
  if (EMAS_URL) {
    // 远程调用 EMAS
    result = await httpPostJson(EMAS_URL, {
      args: { photoBase64, timestamp, signature },
    });
  } else {
    // 本地直接调用 recognize（绕过 index.js 的 DB 读取，直接传 SECRET_KEY）
    const { recognize } = require('../lib/recognize.js');
    const { loadModels } = require('../lib/models.js');
    await loadModels();
    try {
      const data = await recognize(photoBuffer, { timestamp, signature }, SECRET_KEY);
      result = { ok: true, message: 'OK', data };
    } catch (err) {
      result = { ok: false, message: String(err.message || err), data: null };
    }
  }
  const elapsed = Date.now() - t0;
  return { result, elapsed };
}

/** 测试 EMAS deploy_test */
async function testDeploy() {
  console.log('=== EMAS deploy_test 测试 ===\n');
  const handler = require('../index.js');
  const result = await handler({ args: { deploy_test: true } });
  console.log('返回值:', JSON.stringify(result));
  console.log('期望值: "v1.0"');
  console.log('结果:', result === 'v1.0' ? '✓ 通过' : '✗ 失败');
}

/** 测试签名校验（错误签名应被拒绝）*/
async function testSignatureRejection() {
  console.log('=== 签名校验测试 ===\n');

  // 本地 HTTP
  if (SERVER_URL) {
    try {
      const r = await httpPostMultipart(`${SERVER_URL}/recognizeCatPhoto`, {
        photo: Buffer.from('fake'),
        timestamp: '1',
        signature: 'wrong',
      });
      console.log('HTTP 错误签名:', r.ok === false && r.message.includes('signature') ? '✓ 正确拒绝' : '✗ 未拒绝');
    } catch (e) {
      console.log('HTTP 服务未启动，跳过');
    }
  }

  // EMAS（本地直接调用 recognize，绕过 index.js 的 DB 读取）
  const { recognize } = require('../lib/recognize.js');
  const { loadModels } = require('../lib/models.js');
  try {
    await loadModels();
    await recognize(Buffer.from('fake'), { timestamp: 1, signature: 'wrong' }, SECRET_KEY);
    console.log('EMAS 错误签名: ✗ 未拒绝');
  } catch (err) {
    const msg = String(err.message || err);
    console.log('EMAS 错误签名:', msg.includes('signature') ? '✓ 正确拒绝' : '✗ 未拒绝 (' + msg + ')');
  }
}

// ============ 主流程 ============

async function main() {
  const mode = process.argv[2];
  const args = process.argv.slice(3);

  console.log('SECRET_KEY:', SECRET_KEY ? `已配置(长度${SECRET_KEY.length})` : '⚠️ 未配置! 请在 env 文件中设置或在环境变量中提供');
  console.log('SERVER_URL:', SERVER_URL);
  console.log('EMAS_URL:', EMAS_URL || '(本地直接调用 index.js)');
  console.log('');

  if (!mode) {
    console.log('用法: node test/test.js <http|emas|both|batch|deploy|signature> [参数...]');
    console.log('示例:');
    console.log('  node test/test.js http  <图片路径>');
    console.log('  node test/test.js emas  <图片路径>');
    console.log('  node test/test.js both  <图片路径>');
    console.log('  node test/test.js batch <图片目录> [数量]');
    console.log('  node test/test.js deploy');
    console.log('  node test/test.js signature');
    process.exit(1);
  }

  switch (mode) {
    case 'deploy': {
      await testDeploy();
      break;
    }

    case 'signature': {
      await testSignatureRejection();
      break;
    }

    case 'http':
    case 'emas':
    case 'both': {
      if (args.length === 0) {
        console.error('请提供图片路径');
        process.exit(1);
      }
      for (const photoPath of args) {
        console.log(`=== ${mode.toUpperCase()} 测试: ${photoPath} ===`);

        if (mode === 'http' || mode === 'both') {
          try {
            const { result, elapsed } = await testHttp(photoPath);
            console.log(`[HTTP] ${elapsed}ms | ${summarize(result)}`);
            if (!result.ok) console.log(`       错误: ${result.message}`);
          } catch (e) {
            console.log(`[HTTP] ✗ 请求失败: ${e.message}`);
            console.log('       请确认 server.js 已启动 (npm start)');
          }
        }

        if (mode === 'emas' || mode === 'both') {
          try {
            const { result, elapsed } = await testEmas(photoPath);
            console.log(`[EMAS] ${elapsed}ms | ${summarize(result)}`);
            if (!result.ok) console.log(`       错误: ${result.message}`);
          } catch (e) {
            console.log(`[EMAS] ✗ 调用失败: ${e.message}`);
          }
        }

        if (mode === 'both') {
          // 对比
          try {
            const r1 = await testHttp(photoPath);
            const r2 = await testEmas(photoPath);
            console.log(`[对比] ${compareResults(r1.result, r2.result)}`);
          } catch (e) {
            console.log(`[对比] 跳过: ${e.message}`);
          }
        }
        console.log('');
      }
      break;
    }

    case 'batch': {
      const dir = args[0];
      const count = parseInt(args[1] || '5', 10);
      if (!dir) {
        console.error('请提供图片目录');
        process.exit(1);
      }

      // 收集目录下的图片
      let photos = [];
      function walk(d) {
        const entries = fs.readdirSync(d, { withFileTypes: true });
        for (const e of entries) {
          const full = path.join(d, e.name);
          if (e.isDirectory()) walk(full);
          else if (/\.(jpg|jpeg|png)$/i.test(e.name)) photos.push(full);
        }
      }
      walk(dir);
      photos = photos.slice(0, count);

      console.log(`批量测试 ${photos.length} 张图片\n`);

      const results = { http: { ok: 0, fail: 0 }, emas: { ok: 0, fail: 0 } };
      const agreeCount = { value: 0, total: 0 };

      for (const photoPath of photos) {
        const shortName = photoPath.replace(dir, '.');
        process.stdout.write(`${shortName}: `);

        let httpResult = null, emasResult = null;

        try {
          const { result } = await testHttp(photoPath);
          httpResult = result;
          if (result.ok) results.http.ok++;
          else results.http.fail++;
          process.stdout.write(`HTTP ${result.ok ? '✓' : '✗'} `);
        } catch (e) {
          results.http.fail++;
          process.stdout.write('HTTP 跳过 ');
        }

        try {
          const { result } = await testEmas(photoPath);
          emasResult = result;
          if (result.ok) results.emas.ok++;
          else results.emas.fail++;
          process.stdout.write(`EMAS ${result.ok ? '✓' : '✗'}`);
        } catch (e) {
          results.emas.fail++;
          process.stdout.write('EMAS 跳过');
        }

        if (httpResult && emasResult && httpResult.ok && emasResult.ok) {
          agreeCount.total++;
          const t1 = httpResult.data?.recognizeResults?.[0]?.catID;
          const t2 = emasResult.data?.recognizeResults?.[0]?.catID;
          if (t1 === t2) {
            agreeCount.value++;
            process.stdout.write(` | top1一致(${t1.slice(0, 8)}...)`);
          } else {
            process.stdout.write(` | top1不一致!`);
          }
        }
        process.stdout.write('\n');
      }

      console.log('\n=== 汇总 ===');
      console.log(`HTTP: ${results.http.ok} 成功, ${results.http.fail} 失败`);
      console.log(`EMAS: ${results.emas.ok} 成功, ${results.emas.fail} 失败`);
      if (agreeCount.total > 0) {
        console.log(`top1 一致率: ${agreeCount.value}/${agreeCount.total} (${((agreeCount.value / agreeCount.total) * 100).toFixed(1)}%)`);
      }
      break;
    }

    default:
      console.error(`未知模式: ${mode}`);
      process.exit(1);
  }
}

main().catch((err) => {
  console.error('测试出错:', err);
  process.exit(1);
});
