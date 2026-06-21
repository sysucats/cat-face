const ort = require('onnxruntime-node');
const { config } = require('./config');
const { checkSignature } = require('./signature');
const { letterboxToRGBRaw, cropResizeToRGBRaw, resizeToRGBRaw, rawRGBToNCHW } = require('./image');
const { postprocess } = require('./yoloPost');
const {
  getYoloSession, getCatSession, getFallbackSession, getCatIds, getFallbackIds,
} = require('./models');

/**
 * 通用 ONNX 推理：构造 NCHW tensor，运行会话，返回输出 data
 */
async function runInference(session, inputData, h, w) {
  const inputName = session.inputNames[0];
  const tensor = new ort.Tensor('float32', inputData, [1, 3, h, w]);
  const results = await session.run({ [inputName]: tensor });
  const outputName = session.outputNames[0];
  return results[outputName].data; // Float32Array
}

/**
 * 识别主流程，复刻 Python app.py 的 /recognizeCatPhoto 逻辑
 * @param {Buffer} photoBuffer - 图片原始字节
 * @param {Object} formFields - { timestamp, signature, catIdx? }
 * @param {string} secretKey - 签名密钥（由调用方提供：EMAS 从 DB 读取，本地测试从 env 读取）
 * @returns {Object} { catBoxes, recognizeResults }
 */
async function recognize(photoBuffer, formFields, secretKey) {
  // 1. 签名校验
  const timestamp = parseInt(formFields.timestamp, 10);
  const signature = formFields.signature;
  if (!checkSignature(photoBuffer, timestamp, signature, secretKey, config.TOLERANT_TIME_ERROR)) {
    throw new Error('signature verification failed');
  }

  // 2. YOLO 检测猫的位置
  const yoloMeta = await letterboxToRGBRaw(photoBuffer, 640);
  const yoloInput = rawRGBToNCHW(yoloMeta.raw, yoloMeta.paddedW, yoloMeta.paddedH);
  const yoloOutput = await runInference(getYoloSession(), yoloInput, yoloMeta.paddedH, yoloMeta.paddedW);
  const catBoxes = postprocess(yoloOutput, yoloMeta);

  let scores;
  let ids;

  if (catBoxes.length >= 1) {
    // 3a. 检测到猫：crop 指定猫 → cat 模型识别
    let catIdx = 0;
    if (formFields.catIdx !== undefined && formFields.catIdx !== '') {
      const idx = parseInt(formFields.catIdx, 10);
      if (idx < catBoxes.length) catIdx = idx;
    }
    const box = catBoxes[catIdx];
    const raw = await cropResizeToRGBRaw(photoBuffer, box, config.IMG_SIZE);
    const input = rawRGBToNCHW(raw, config.IMG_SIZE, config.IMG_SIZE);
    scores = await runInference(getCatSession(), input, config.IMG_SIZE, config.IMG_SIZE);
    ids = getCatIds();
  } else {
    // 3b. 未检测到猫：整图 → fallback 模型识别
    const raw = await resizeToRGBRaw(photoBuffer, config.FALLBACK_IMG_SIZE);
    const input = rawRGBToNCHW(raw, config.FALLBACK_IMG_SIZE, config.FALLBACK_IMG_SIZE);
    scores = await runInference(getFallbackSession(), input, config.FALLBACK_IMG_SIZE, config.FALLBACK_IMG_SIZE);
    ids = getFallbackIds();
  }

  // 4. 按 score 降序排序
  const scoreArray = Array.from(scores);
  const catIdWithScore = scoreArray.map((s, i) => ({ catID: ids[i], score: s }));
  catIdWithScore.sort((a, b) => b.score - a.score);

  // 5. 返回结果（catBoxes 只含坐标，不含 score）
  return {
    catBoxes: catBoxes.slice(0, config.CAT_BOX_MAX_RET_NUM).map((b) => ({
      xmin: b.xmin,
      ymin: b.ymin,
      xmax: b.xmax,
      ymax: b.ymax,
    })),
    recognizeResults: catIdWithScore.slice(0, config.RECOGNIZE_MAX_RET_NUM),
  };
}

module.exports = { recognize };
