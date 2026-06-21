/**
 * YOLO11 ONNX 后处理
 * 输出格式: float32[1, 84, 8400]
 *   - 84 = 4(xywh) + 80(COCO 类别分数)
 *   - 8400 = 候选框数
 *   - cat 在 COCO 中是索引 15
 * 数据布局（行优先）: data[(4+c)*8400 + w]
 *   - c=0..3: cx, cy, w, h
 *   - c=4..83: 80 个类别分数
 *
 * Ultralytics predict 默认参数: conf=0.25, iou=0.7, per-class NMS
 * 本项目只取 cat，等价于对 cat 框做 NMS
 */

const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.7;
const CAT_CLASS_ID = 15;
const NUM_ANCHORS = 8400;

function clamp(v, min, max) {
  return v < min ? min : v > max ? max : v;
}

function iou(a, b) {
  const interX1 = Math.max(a.xmin, b.xmin);
  const interY1 = Math.max(a.ymin, b.ymin);
  const interX2 = Math.min(a.xmax, b.xmax);
  const interY2 = Math.min(a.ymax, b.ymax);
  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  const interArea = interW * interH;
  const areaA = (a.xmax - a.xmin) * (a.ymax - a.ymin);
  const areaB = (b.xmax - b.xmin) * (b.ymax - b.ymin);
  const union = areaA + areaB - interArea;
  return union > 0 ? interArea / union : 0;
}

/**
 * YOLO 后处理
 * @param {Float32Array|number[]} outputData - ONNX 输出，长度 1*84*8400
 * @param {Object} meta - letterboxToRGBRaw 返回的 { r, left, top, ow, oh }
 * @returns {Array<{xmin,ymin,xmax,ymax,score}>} 已 NMS、已反 letterbox、已 clamp 的 cat 框
 */
function postprocess(outputData, meta) {
  const { r, left, top, ow, oh } = meta;
  const candidates = [];

  // 收集 cat 候选框（conf > 0.25）
  for (let w = 0; w < NUM_ANCHORS; w++) {
    const cx = outputData[0 * NUM_ANCHORS + w];
    const cy = outputData[1 * NUM_ANCHORS + w];
    const bw = outputData[2 * NUM_ANCHORS + w];
    const bh = outputData[3 * NUM_ANCHORS + w];
    const catScore = outputData[(4 + CAT_CLASS_ID) * NUM_ANCHORS + w];

    if (catScore > CONF_THRESHOLD) {
      // xywh → xyxy (padded 坐标系)
      const x1 = cx - bw / 2;
      const y1 = cy - bh / 2;
      const x2 = cx + bw / 2;
      const y2 = cy + bh / 2;
      // 反 letterbox 到原图坐标，并 clamp
      candidates.push({
        xmin: clamp((x1 - left) / r, 0, ow),
        ymin: clamp((y1 - top) / r, 0, oh),
        xmax: clamp((x2 - left) / r, 0, ow),
        ymax: clamp((y2 - top) / r, 0, oh),
        score: catScore,
      });
    }
  }

  // 按 score 降序排序
  candidates.sort((a, b) => b.score - a.score);

  // NMS（IoU > 0.7 抑制）
  const kept = [];
  while (candidates.length > 0) {
    const best = candidates.shift();
    kept.push(best);
    for (let i = candidates.length - 1; i >= 0; i--) {
      if (iou(best, candidates[i]) > IOU_THRESHOLD) {
        candidates.splice(i, 1);
      }
    }
  }

  return kept;
}

module.exports = { postprocess, CONF_THRESHOLD, IOU_THRESHOLD, CAT_CLASS_ID };
