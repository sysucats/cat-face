/**
 * Node 端推理链路测试脚本（绕过 HTTP 和签名校验）
 * 直接调用底层模块验证：模型加载 → letterbox → YOLO 推理 → 后处理 → 分类推理
 * 用法: node test/test-node.js <图片路径>
 */
const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');
const { config } = require('../lib/config');
const { loadModels, getYoloSession, getCatSession, getFallbackSession, getCatIds, getFallbackIds } = require('../lib/models');
const { letterboxToRGBRaw, cropResizeToRGBRaw, resizeToRGBRaw, rawRGBToNCHW } = require('../lib/image');
const { postprocess } = require('../lib/yoloPost');

async function runInference(session, inputData, h, w) {
  const inputName = session.inputNames[0];
  const tensor = new ort.Tensor('float32', inputData, [1, 3, h, w]);
  const results = await session.run({ [inputName]: tensor });
  const outputName = session.outputNames[0];
  return results[outputName].data;
}

async function main() {
  const photoPath = process.argv[2];
  if (!photoPath) {
    console.error('用法: node test/test-node.js <图片路径>');
    process.exit(1);
  }

  const photoBuffer = fs.readFileSync(photoPath);
  console.log('图片大小:', photoBuffer.length, 'bytes');

  await loadModels();

  // YOLO 检测
  console.log('\n=== YOLO 检测 ===');
  const t0 = Date.now();
  const yoloMeta = await letterboxToRGBRaw(photoBuffer, 640);
  console.log('letterbox: 原图', yoloMeta.ow + 'x' + yoloMeta.oh, '→ 640x640, r=' + yoloMeta.r.toFixed(4) + ', left=' + yoloMeta.left + ', top=' + yoloMeta.top);
  const yoloInput = rawRGBToNCHW(yoloMeta.raw, yoloMeta.paddedW, yoloMeta.paddedH);
  const yoloOutput = await runInference(getYoloSession(), yoloInput, yoloMeta.paddedH, yoloMeta.paddedW);
  console.log('YOLO 输出长度:', yoloOutput.length, '(期望 705600 = 84*8400)');
  const catBoxes = postprocess(yoloOutput, yoloMeta);
  console.log('检测到猫框数量:', catBoxes.length);
  catBoxes.forEach((b, i) => {
    console.log('  框' + i + ': xmin=' + b.xmin.toFixed(1) + ' ymin=' + b.ymin.toFixed(1) + ' xmax=' + b.xmax.toFixed(1) + ' ymax=' + b.ymax.toFixed(1) + ' score=' + b.score.toFixed(4));
  });
  console.log('YOLO 耗时:', Date.now() - t0, 'ms');

  let scores;
  let ids;

  if (catBoxes.length >= 1) {
    // cat 模型
    console.log('\n=== Cat 模型识别 ===');
    const t1 = Date.now();
    const box = catBoxes[0];
    const raw = await cropResizeToRGBRaw(photoBuffer, box, config.IMG_SIZE);
    const input = rawRGBToNCHW(raw, config.IMG_SIZE, config.IMG_SIZE);
    scores = await runInference(getCatSession(), input, config.IMG_SIZE, config.IMG_SIZE);
    ids = getCatIds();
    console.log('分类输出长度:', scores.length, '(类别数)');
    console.log('Cat 模型耗时:', Date.now() - t1, 'ms');
  } else {
    // fallback 模型
    console.log('\n=== Fallback 模型识别（未检测到猫）===');
    const t1 = Date.now();
    const raw = await resizeToRGBRaw(photoBuffer, config.FALLBACK_IMG_SIZE);
    const input = rawRGBToNCHW(raw, config.FALLBACK_IMG_SIZE, config.FALLBACK_IMG_SIZE);
    scores = await runInference(getFallbackSession(), input, config.FALLBACK_IMG_SIZE, config.FALLBACK_IMG_SIZE);
    ids = getFallbackIds();
    console.log('分类输出长度:', scores.length, '(类别数)');
    console.log('Fallback 模型耗时:', Date.now() - t1, 'ms');
  }

  // 排序输出 top 5
  const scoreArray = Array.from(scores);
  const ranked = scoreArray.map((s, i) => ({ catID: ids[i], score: s })).sort((a, b) => b.score - a.score);
  console.log('\n=== Top 5 识别结果 ===');
  ranked.slice(0, 5).forEach((r, i) => {
    console.log('  ' + (i + 1) + '. catID=' + r.catID + ' score=' + r.score.toFixed(4));
  });

  console.log('\n总耗时:', Date.now() - t0, 'ms');
}

main().catch((err) => {
  console.error('错误:', err);
  process.exit(1);
});
