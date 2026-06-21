#!/usr/bin/env node
/**
 * cat-face EMAS 部署打包脚本
 *
 * 功能：把 node-server/ 中 EMAS 云函数必要的文件打包成 catFace.zip
 *
 * 用法：
 *   node pack-emas.js              默认模式：含 node_modules（适合本地测试）
 *   node pack-emas.js --no-deps    不含 node_modules（让 EMAS 从 package.json 安装）
 *   node pack-emas.js --trim       含 node_modules 但裁剪非 Linux 平台文件
 *   node pack-emas.js --output xxx.zip  自定义输出文件名
 *
 * 输出：catFace.zip（在 cat-face 根目录下）
 *
 * ⚠️ 原生模块跨平台问题：
 *   onnxruntime-node 和 sharp 含平台特定的 .node 二进制文件。
 *   需先安装 Linux 平台包：npm install --os=linux --cpu=x64 sharp
 *   - 方案 A（推荐）：用 --no-deps，让 EMAS 从 package.json 安装 Linux 版依赖
 *   - 方案 B：先装好 Linux 平台包，用默认模式打包（含双平台二进制）
 *   - 方案 C：用 --trim 裁剪掉非 Linux 文件（需先装好 Linux 平台包）
 */

const fs = require('fs');
const path = require('path');
const { execSync, execFileSync } = require('child_process');
const os = require('os');

// ============ 配置 ============

// 本脚本位于 node-server/ 下，ROOT 指向 cat-face 根目录（用于输出 zip）
const ROOT = path.resolve(__dirname, '..');
const NODE_SERVER = __dirname;
const OUTPUT = path.join(ROOT, 'catFace.zip');

// 命令行参数
const args = process.argv.slice(2);
const noDeps = args.includes('--no-deps');
const trim = args.includes('--trim');
const outputIdx = args.indexOf('--output');
const outputFile = outputIdx >= 0 ? path.resolve(ROOT, args[outputIdx + 1]) : OUTPUT;

// 需要打包的文件/目录（相对 node-server/）
const INCLUDE_FILES = [
  'index.js',        // EMAS 入口
  'package.json',    // 依赖清单（EMAS 安装依赖时需要）
  'lib/',            // 核心逻辑模块
  'models/',         // ONNX 模型 + 类别 JSON
];

// 需要排除的文件/目录
const EXCLUDE_FILES = [
  'server.js',       // express HTTP 入口（EMAS 不需要）
  'env',             // 环境配置（EMAS 控制台配置）
  'test/',           // 测试脚本
  'package-lock.json', // 锁定文件（可选，EMAS 可重新生成）
  '.workbuddy/',
];

// trim 模式下要排除的非 Linux 平台目录
const TRIM_PATTERNS = [
  // onnxruntime-node: 保留 linux，删除 darwin/win32
  'node_modules/onnxruntime-node/bin/napi-v3/darwin',
  'node_modules/onnxruntime-node/bin/napi-v3/win32',
  // @img/sharp-*: 只保留 linux，删除其他平台
  // (sharp 的平台包名格式: @img/sharp-{platform}-{arch})
];

// ============ 工具函数 ============

function formatSize(bytes) {
  if (bytes < 1024) return bytes + 'B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + 'KB';
  return (bytes / (1024 * 1024)).toFixed(1) + 'MB';
}

function getDirSize(dirPath) {
  let total = 0;
  const entries = fs.readdirSync(dirPath, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dirPath, entry.name);
    if (entry.isDirectory()) total += getDirSize(full);
    else if (entry.isFile()) total += fs.statSync(full).size;
  }
  return total;
}

function copyRecursive(src, dest) {
  const stat = fs.statSync(src);
  if (stat.isDirectory()) {
    fs.mkdirSync(dest, { recursive: true });
    const entries = fs.readdirSync(src, { withFileTypes: true });
    for (const entry of entries) {
      copyRecursive(path.join(src, entry.name), path.join(dest, entry.name));
    }
  } else {
    fs.copyFileSync(src, dest);
  }
}

/** 获取 zip 命令路径 */
function findZip() {
  try {
    const which = process.platform === 'win32' ? 'where' : 'which';
    const result = execSync(`${which} zip`, { encoding: 'utf8' }).trim();
    if (result) return result.split('\n')[0].trim();
  } catch (e) {
    // zip not found
  }
  return null;
}

/** 收集 node_modules 中非 Linux 的 @img/sharp-* 包名 */
function getNonLinuxSharpPackages(nodeModulesDir) {
  const atImgDir = path.join(nodeModulesDir, '@img');
  if (!fs.existsSync(atImgDir)) return [];
  const pkgs = fs.readdirSync(atImgDir);
  // 保留含 "linux" 的包（sharp-linux-x64, sharp-libvips-linux-x64, sharp-linuxmusl-x64 等）
  // 排除其他平台（sharp-win32-x64, sharp-darwin-* 等）
  return pkgs.filter((p) => p.startsWith('sharp-') && !p.includes('linux'));
}

// ============ 主流程 ============

function main() {
  console.log('========================================');
  console.log('  cat-face EMAS 打包工具');
  console.log('========================================\n');

  const mode = noDeps ? 'no-deps（不含 node_modules）' : trim ? 'trim（裁剪非 Linux）' : 'full（完整）';
  console.log(`打包模式: ${mode}`);
  console.log(`输出文件: ${outputFile}\n`);

  // 1. 创建临时暂存目录
  const stagingDir = path.join(os.tmpdir(), `catface-pack-${Date.now()}`);
  console.log('暂存目录:', stagingDir);

  try {
    fs.mkdirSync(stagingDir, { recursive: true });

    // 2. 复制必要文件
    console.log('\n--- 复制文件 ---');
    let totalSize = 0;

    for (const item of INCLUDE_FILES) {
      const src = path.join(NODE_SERVER, item);
      if (!fs.existsSync(src)) {
        console.log(`  跳过（不存在）: ${item}`);
        continue;
      }
      const dest = path.join(stagingDir, item);
      copyRecursive(src, dest);

      const size = fs.statSync(src).isDirectory() ? getDirSize(src) : fs.statSync(src).size;
      totalSize += size;
      console.log(`  ✓ ${item} (${formatSize(size)})`);
    }

    // 3. 复制 node_modules（根据模式）
    if (!noDeps) {
      const nodeModulesSrc = path.join(NODE_SERVER, 'node_modules');
      if (fs.existsSync(nodeModulesSrc)) {
        console.log('\n--- 复制 node_modules ---');
        const nodeModulesDest = path.join(stagingDir, 'node_modules');
        copyRecursive(nodeModulesSrc, nodeModulesDest);

        let nmSize = getDirSize(nodeModulesDest);

        // trim 模式：删除非 Linux 平台文件
        if (trim) {
          console.log('\n--- 裁剪非 Linux 平台文件 ---');
          let savedSize = 0;

          // onnxruntime-node 平台目录
          for (const pattern of TRIM_PATTERNS) {
            const trimPath = path.join(stagingDir, pattern);
            if (fs.existsSync(trimPath)) {
              const sz = getDirSize(trimPath);
              fs.rmSync(trimPath, { recursive: true, force: true });
              savedSize += sz;
              console.log(`  删除: ${pattern} (${formatSize(sz)})`);
            }
          }

          // @img/sharp-* 非 Linux 包
          const nonLinuxSharp = getNonLinuxSharpPackages(nodeModulesDest);
          for (const pkg of nonLinuxSharp) {
            const pkgPath = path.join(nodeModulesDest, '@img', pkg);
            if (fs.existsSync(pkgPath)) {
              const sz = getDirSize(pkgPath);
              fs.rmSync(pkgPath, { recursive: true, force: true });
              savedSize += sz;
              console.log(`  删除: @img/${pkg} (${formatSize(sz)})`);
            }
          }

          // onnxruntime-node 的 DirectML.dll（仅 Windows 用）
          const dmlPath = path.join(nodeModulesDest, 'onnxruntime-node', 'bin', 'napi-v3', 'win32');
          // 已在上面处理

          nmSize -= savedSize;
          console.log(`  裁剪节省: ${formatSize(savedSize)}`);
        }

        totalSize += nmSize;
        console.log(`  ✓ node_modules (${formatSize(nmSize)})`);
      } else {
        console.log('\n  ⚠️  node_modules 不存在，跳过');
      }
    } else {
      console.log('\n  跳过 node_modules（--no-deps 模式）');
    }

    console.log(`\n暂存目录总大小: ${formatSize(totalSize)}`);

    // 4. 删除旧 zip
    if (fs.existsSync(outputFile)) {
      fs.unlinkSync(outputFile);
      console.log(`\n删除旧文件: ${path.basename(outputFile)}`);
    }

    // 5. 打包
    console.log('\n--- 创建 zip ---');
    const zipCmd = findZip();
    if (zipCmd) {
      // 使用 zip 命令
      console.log(`使用 zip: ${zipCmd}`);
      execFileSync(zipCmd, ['-r', '-q', outputFile, '.'], {
        cwd: stagingDir,
        stdio: 'inherit',
      });
    } else {
      // 回退到 PowerShell Compress-Archive（Windows）
      console.log('zip 命令未找到，使用 PowerShell Compress-Archive');
      const psCmd = `Compress-Archive -Path "${stagingDir}\\*" -DestinationPath "${outputFile}" -Force`;
      execSync(`powershell -Command "${psCmd}"`, { stdio: 'inherit' });
    }

    // 6. 验证
    if (fs.existsSync(outputFile)) {
      const zipSize = fs.statSync(outputFile).size;
      console.log(`\n========================================`);
      console.log(`  ✓ 打包成功！`);
      console.log(`========================================`);
      console.log(`  文件: ${outputFile}`);
      console.log(`  大小: ${formatSize(zipSize)} (${zipSize} bytes)`);
      console.log(`  原始: ${formatSize(totalSize)}`);
      console.log(`  压缩率: ${((1 - zipSize / totalSize) * 100).toFixed(1)}%`);

      // 列出 zip 内容摘要
      console.log(`\n--- zip 内容摘要 ---`);
      try {
        const list = execSync(`"${zipCmd}" -l "${outputFile}"`, { encoding: 'utf8' });
        const lines = list.split('\n').filter((l) => l.trim());
        // 显示前 20 行和最后几行
        if (lines.length > 25) {
          lines.slice(0, 20).forEach((l) => console.log('  ' + l));
          console.log('  ...');
          lines.slice(-5).forEach((l) => console.log('  ' + l));
        } else {
          lines.forEach((l) => console.log('  ' + l));
        }
      } catch (e) {
        console.log('  (无法列出内容)');
      }
    } else {
      throw new Error('zip 文件未生成');
    }

    // 7. 部署提示
    console.log(`\n--- EMAS 部署提示 ---`);
    if (noDeps) {
      console.log('  • 上传 catFace.zip 到 EMAS');
      console.log('  • EMAS 会根据 package.json 自动安装依赖');
      console.log('  • 在 EMAS 控制台配置环境变量：SECRET_KEY 等');
    } else if (trim) {
      console.log('  • 已裁剪非 Linux 平台文件，仅保留 Linux 二进制');
      console.log('  • 首次使用前需安装 Linux 平台包: npm install --os=linux --cpu=x64 sharp');
      console.log('  • 建议先用 Docker 测试: node test/test-docker.js');
      console.log('  • 在 EMAS 控制台配置环境变量：SECRET_KEY 等');
    } else {
      console.log('  • 包含全平台二进制（win32 + linux），体积较大');
      console.log('  • 建议用 --trim 模式减小体积: node pack-emas.js --trim');
      console.log('  • 在 EMAS 控制台配置环境变量：SECRET_KEY 等');
    }

  } finally {
    // 清理暂存目录
    if (fs.existsSync(stagingDir)) {
      fs.rmSync(stagingDir, { recursive: true, force: true });
      console.log('\n暂存目录已清理');
    }
  }
}

main();
