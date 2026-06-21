/**
 * EMAS Serverless 云函数入口
 *
 * 复用 lib/recognize.js 的识别主流程，仅做入口适配：
 * - module.exports = async (ctx) => {} 是 EMAS 云函数标准入口
 * - ctx.args.deploy_test === true 用于部署冒烟测试
 * - 模型懒加载（云函数实例复用时只加载一次）
 * - 图片支持两种传入方式：
 *   1) photoBase64: base64 字符串（不含 data:image/... 前缀，适合 < 500KB 的小图）
 *   2) photoURL:    HTTP/COS URL，云函数用 ctx.httpclient 下载（适合大图，客户端先上传 COS）
 *
 * 入参（ctx.args）:
 *   - photoBase64 | photoURL  二选一
 *   - timestamp    number   时间戳（秒）
 *   - signature    string   SHA256(base64(photo) + str(timestamp) + SECRET_KEY)
 *   - catIdx       number?  可选，选择第几个猫框，默认 0
 *
 * SECRET_KEY 来源：
 *   运行时从 EMAS 数据库 app_secret 集合读取（不再从 env/config 读取）
 *   读取方式：ctx.mpserverless.db.collection('app_secret').findOne()
 *   实例复用期间缓存，避免每次请求都查库
 *
 * 返回:
 *   { ok: true,  message: 'OK', data: { catBoxes, recognizeResults } }
 *   { ok: false, message: '<错误信息>', data: null }
 */

const { recognize } = require('./lib/recognize.js');
const { loadModels } = require('./lib/models.js');

// 模型懒加载标志（云函数实例复用期间只加载一次）
let modelsReady = false;
let loadingPromise = null;

/**
 * 确保模型已加载（并发请求只触发一次加载）
 * 云函数冷启动后首次请求会触发，后续复用实例直接跳过
 */
async function ensureModelsLoaded() {
  if (modelsReady) return;
  if (!loadingPromise) {
    loadingPromise = loadModels().then(() => {
      modelsReady = true;
      loadingPromise = null;
    }).catch((err) => {
      loadingPromise = null; // 失败后允许重试
      throw err;
    });
  }
  await loadingPromise;
}

// SECRET_KEY 缓存（云函数实例复用期间只查一次库）
let cachedSecretKey = null;
let secretKeyPromise = null;

/**
 * 从 EMAS 数据库 app_secret 集合读取 SECRET_KEY
 * 实例复用期间缓存，避免每次请求都查库
 * @param {Object} ctx - EMAS 云函数上下文
 * @returns {Promise<string>} SECRET_KEY
 */
async function getSecretKey(ctx) {
  if (cachedSecretKey) return cachedSecretKey;
  if (!secretKeyPromise) {
    secretKeyPromise = ctx.mpserverless.db.collection('app_secret').findOne()
      .then(({ result }) => {
        if (!result || !result.SECRET_KEY) {
          throw new Error('app_secret 集合中未找到 SECRET_KEY 字段');
        }
        cachedSecretKey = result.SECRET_KEY;
        secretKeyPromise = null;
        return cachedSecretKey;
      })
      .catch((err) => {
        secretKeyPromise = null; // 失败后允许重试
        throw err;
      });
  }
  return secretKeyPromise;
}

/**
 * 从 photoURL 下载图片为 Buffer
 * ctx.httpclient 是 EMAS 内置的 HTTP 客户端（类似 urllib）
 */
async function downloadPhoto(ctx, url) {
  const rsp = await ctx.httpclient.request(url, {
    method: 'GET',
    responseType: 'arraybuffer',
    timeout: 10000,
  });
  if (rsp.status !== 200) {
    throw new Error(`download photo failed: HTTP ${rsp.status}`);
  }
  return Buffer.from(rsp.data);
}

module.exports = async (ctx) => {
  // 部署冒烟测试
  if (ctx.args?.deploy_test === true) {
    return 'v1.0';
  }

  try {
    // 懒加载模型
    await ensureModelsLoaded();

    const args = ctx.args || {};
    const { photoBase64, photoURL, timestamp, signature, catIdx } = args;

    // 获取图片 Buffer（两种方式二选一）
    let photoBuffer;
    if (photoBase64) {
      // 去除可能的 data:image/...;base64, 前缀
      const b64 = String(photoBase64).replace(/^data:[^;]+;base64,/, '');
      photoBuffer = Buffer.from(b64, 'base64');
    } else if (photoURL) {
      photoBuffer = await downloadPhoto(ctx, photoURL);
    } else {
      return { ok: false, message: 'missing photoBase64 or photoURL', data: null };
    }

    // 校验必填字段
    if (timestamp === undefined || timestamp === null || !signature) {
      return { ok: false, message: 'missing timestamp or signature', data: null };
    }

    // 组装表单字段，调用识别主流程
    const formFields = { timestamp, signature };
    if (catIdx !== undefined && catIdx !== null) formFields.catIdx = catIdx;

    // 从 EMAS 数据库读取 SECRET_KEY（带缓存）
    const secretKey = await getSecretKey(ctx);

    const result = await recognize(photoBuffer, formFields, secretKey);

    return { ok: true, message: 'OK', data: result };
  } catch (err) {
    return { ok: false, message: String(err.message || err), data: null };
  }
};
