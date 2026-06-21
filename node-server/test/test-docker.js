#!/usr/bin/env node
/**
 * 本地 Docker Linux 环境测试脚本
 *
 * 功能：在 Windows 上用 Docker 构建一个 Linux 容器，模拟 EMAS 运行环境
 *       验证 sharp / onnxruntime-node 的 Linux 二进制能否正常加载
 *
 * 用法：
 *   node test/test-docker.js              构建 + 运行基本测试（deploy_test + 模型加载）
 *   node test/test-docker.js --build      仅构建镜像
 *   node test/test-docker.js --run        仅运行镜像（需先 build）
 *   node test/test-docker.js --full       构建 + 运行 + 完整识别测试（需挂载测试图片）
 *
 * 前提：Docker Desktop 已启动
 */

const { execSync, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const NODE_SERVER = path.resolve(__dirname, '..');
const IMAGE_NAME = 'catface-test';

function run(cmd, opts = {}) {
  console.log(`> ${cmd}`);
  execSync(cmd, { stdio: 'inherit', cwd: NODE_SERVER, ...opts });
}

function checkDocker() {
  try {
    execSync('docker info', { stdio: 'pipe' });
    return true;
  } catch (e) {
    return false;
  }
}

async function main() {
  const mode = process.argv[2] || 'default';

  console.log('========================================');
  console.log('  cat-face Docker Linux 测试');
  console.log('========================================\n');

  if (!checkDocker()) {
    console.error('✗ Docker 未运行，请先启动 Docker Desktop');
    process.exit(1);
  }
  console.log('✓ Docker 可用\n');

  const shouldBuild = mode === '--build' || mode === 'default' || mode === '--full';
  const shouldRun = mode === '--run' || mode === 'default' || mode === '--full';
  const isFull = mode === '--full';

  // 1. 构建
  if (shouldBuild) {
    console.log('--- 构建 Docker 镜像 ---');
    console.log('这会创建一个 Linux 容器并安装 Linux 版依赖（验证 sharp/onnxruntime 兼容性）');
    console.log('');

    // 创建 .dockerignore 避免复制 node_modules（让容器内重新安装 Linux 版）
    const dockerignorePath = path.join(NODE_SERVER, '.dockerignore');
    const hadDockerignore = fs.existsSync(dockerignorePath);
    if (!hadDockerignore) {
      fs.writeFileSync(dockerignorePath, 'node_modules\ntest\nenv\n*.zip\n');
    }

    try {
      run(`docker build -t ${IMAGE_NAME} -f Dockerfile .`);
      console.log('\n✓ 镜像构建成功\n');
    } finally {
      if (!hadDockerignore && fs.existsSync(dockerignorePath)) {
        fs.unlinkSync(dockerignorePath);
      }
    }
  }

  // 2. 基本运行测试
  if (shouldRun && !isFull) {
    console.log('--- 运行基本测试 ---\n');
    run(`docker run --rm ${IMAGE_NAME}`);
    console.log('\n✓ 基本测试完成\n');
    console.log('说明：');
    console.log('  - deploy_test 返回 "v1.0" → 入口正确');
    console.log('  - 签名校验生效 → 模型已成功加载（sharp + onnxruntime Linux 二进制正常）');
    console.log('  - 如果报错，请检查 Dockerfile 和 package.json');
  }

  // 3. 完整识别测试（挂载测试图片）
  if (isFull) {
    console.log('--- 运行完整识别测试 ---\n');

    // 找一张测试图片
    const testPhotoDir = path.resolve(NODE_SERVER, '../data/photos');
    let testPhoto = null;
    if (fs.existsSync(testPhotoDir)) {
      const cats = fs.readdirSync(testPhotoDir);
      for (const cat of cats) {
        const catDir = path.join(testPhotoDir, cat);
        if (fs.statSync(catDir).isDirectory()) {
          const photos = fs.readdirSync(catDir).filter((f) => /\.(jpg|jpeg)$/i.test(f));
          if (photos.length > 0) {
            testPhoto = path.join(catDir, photos[0]);
            break;
          }
        }
      }
    }

    if (!testPhoto) {
      console.log('⚠️  未找到测试图片，跳过完整识别测试');
      return;
    }

    console.log('测试图片:', testPhoto.replace(NODE_SERVER, '.'));

    // 在容器内运行完整识别（挂载图片目录 + 读取 env 中的 SECRET_KEY）
    const { SECRET_KEY } = require('./test-config.js');
    const photoRel = path.relative(path.resolve(NODE_SERVER, '..'), testPhoto).replace(/\\/g, '/');

    const testScript = `
      const fs = require('fs');
      const crypto = require('crypto');
      const { recognize } = require('./lib/recognize.js');
      const { loadModels } = require('./lib/models.js');

      (async () => {
        // 读取挂载的图片
        const photoBuffer = fs.readFileSync('/test-photo.jpg');
        const timestamp = Math.floor(Date.now() / 1000);
        const secretKey = process.env.SECRET_KEY || '${SECRET_KEY}';
        const b64 = photoBuffer.toString('base64');
        const signature = crypto.createHash('sha256').update(b64 + String(timestamp) + secretKey, 'utf8').digest('hex');

        console.log('图片大小:', photoBuffer.length, 'bytes');
        await loadModels();
        const data = await recognize(photoBuffer, { timestamp, signature }, secretKey);

        if (data.catBoxes) {
          console.log('✓ 识别成功！');
          console.log('  猫框数量:', data.catBoxes.length);
          const top1 = data.recognizeResults[0];
          console.log('  top1:', top1.catID, 'score:', top1.score.toFixed(4));
        } else {
          console.log('✗ 识别失败:', JSON.stringify(data));
        }
      })().catch(e => { console.error('错误:', e); process.exit(1); });
    `;

    // 将测试脚本通过 -v 挂载进去
    const tmpScript = path.join(NODE_SERVER, '_docker-test.js');
    fs.writeFileSync(tmpScript, testScript);

    try {
      run(`docker run --rm ` +
        `-v "${testPhoto}:/test-photo.jpg:ro" ` +
        `-v "${tmpScript}:/app/_docker-test.js:ro" ` +
        `${IMAGE_NAME} node _docker-test.js`);
      console.log('\n✓ 完整识别测试完成\n');
    } finally {
      if (fs.existsSync(tmpScript)) fs.unlinkSync(tmpScript);
    }
  }
}

main().catch((err) => {
  console.error('测试失败:', err.message);
  process.exit(1);
});
