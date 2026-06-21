const sharp = require('sharp');

/**
 * letterbox 预处理（复刻 Ultralytics 的 letterbox 算法）
 * 按比例缩放使最长边=size，短边居中填充灰色(114,114,114)，得到 size×size
 * 返回 { raw, r, left, top, ow, oh } 供后处理反 letterbox 使用
 */
async function letterboxToRGBRaw(buffer, size = 640) {
  const meta = await sharp(buffer).metadata();
  const ow = meta.width;
  const oh = meta.height;
  const r = Math.min(size / ow, size / oh);
  const newW = Math.round(ow * r);
  const newH = Math.round(oh * r);
  const dw = (size - newW) / 2;
  const dh = (size - newH) / 2;
  // Ultralytics 的 floor/ceil 分配方式
  const left = Math.max(0, Math.round(dw - 0.1));
  const right = Math.max(0, Math.round(dw + 0.1));
  const top = Math.max(0, Math.round(dh - 0.1));
  const bottom = Math.max(0, Math.round(dh + 0.1));

  const raw = await sharp(buffer)
    .resize(newW, newH, { fit: 'fill', kernel: 'lanczos3' })
    .removeAlpha()
    .extend({
      top, bottom, left, right,
      background: { r: 114, g: 114, b: 114 },
    })
    .raw()
    .toBuffer();

  return { raw, r, left, top, ow, oh, paddedW: size, paddedH: size };
}

/**
 * 分类模型输入：crop + resize 到正方形
 * 对应 Python: src_img.crop(box).resize((size, size))
 */
async function cropResizeToRGBRaw(buffer, box, size) {
  // 先获取原图尺寸用于 clamp（PIL crop 超界会填充，sharp extract 需严格在范围内）
  const meta = await sharp(buffer).metadata();
  const ow = meta.width;
  const oh = meta.height;
  const xmin = Math.max(0, Math.floor(box.xmin));
  const ymin = Math.max(0, Math.floor(box.ymin));
  const xmax = Math.min(ow, Math.ceil(box.xmax));
  const ymax = Math.min(oh, Math.ceil(box.ymax));
  const w = Math.max(1, xmax - xmin);
  const h = Math.max(1, ymax - ymin);

  const raw = await sharp(buffer)
    .extract({ left: xmin, top: ymin, width: w, height: h })
    .removeAlpha()
    .resize(size, size, { fit: 'fill' })
    .raw()
    .toBuffer();
  return raw;
}

/**
 * fallback 输入：整图 resize 到正方形
 * 对应 Python: src_img.resize((size, size))
 */
async function resizeToRGBRaw(buffer, size) {
  const raw = await sharp(buffer)
    .removeAlpha()
    .resize(size, size, { fit: 'fill' })
    .raw()
    .toBuffer();
  return raw;
}

/**
 * raw RGB Buffer → float32 NCHW 归一化
 * 对应 Python: np.array(dtype=float32).transpose((2,0,1)) / 255
 * raw: Buffer，每像素3字节 RGB，长度 = w*h*3
 * 返回 Float32Array，长度 3*w*h，顺序 CHW
 */
function rawRGBToNCHW(raw, w, h) {
  const out = new Float32Array(3 * w * h);
  const plane = w * h;
  for (let i = 0; i < plane; i++) {
    out[i] = raw[i * 3] / 255;           // R
    out[plane + i] = raw[i * 3 + 1] / 255; // G
    out[2 * plane + i] = raw[i * 3 + 2] / 255; // B
  }
  return out;
}

module.exports = { letterboxToRGBRaw, cropResizeToRGBRaw, resizeToRGBRaw, rawRGBToNCHW };
