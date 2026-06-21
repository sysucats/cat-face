const ort = require('onnxruntime-node');
const fs = require('fs');
const { config } = require('./config');

// 模块单例
let yoloSession = null;
let catSession = null;
let fallbackSession = null;
let catIds = null;
let fallbackIds = null;

/**
 * 启动时加载三个 ONNX 模型 + 两个类别 ID 列表（全局单例）
 * 强制 CPU 推理，与 Python app.py providers=["CPUExecutionProvider"] 一致
 */
async function loadModels() {
  console.log('==> loading models...');
  const [yolo, cat, fallback] = await Promise.all([
    ort.InferenceSession.create(config.MODEL_PATHS.yolo, { executionProviders: ['cpu'] }),
    ort.InferenceSession.create(config.MODEL_PATHS.cat, { executionProviders: ['cpu'] }),
    ort.InferenceSession.create(config.MODEL_PATHS.fallback, { executionProviders: ['cpu'] }),
  ]);
  yoloSession = yolo;
  catSession = cat;
  fallbackSession = fallback;
  catIds = JSON.parse(fs.readFileSync(config.MODEL_PATHS.catJson, 'utf8'));
  fallbackIds = JSON.parse(fs.readFileSync(config.MODEL_PATHS.fallbackJson, 'utf8'));
  console.log('==> models are loaded.');
}

function getYoloSession() { return yoloSession; }
function getCatSession() { return catSession; }
function getFallbackSession() { return fallbackSession; }
function getCatIds() { return catIds; }
function getFallbackIds() { return fallbackIds; }

module.exports = {
  loadModels,
  getYoloSession,
  getCatSession,
  getFallbackSession,
  getCatIds,
  getFallbackIds,
};
