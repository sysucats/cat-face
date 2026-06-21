/**
 * 本地测试用：从 env 文件加载 SECRET_KEY
 *
 * ⚠️ 仅用于本地测试脚本（test/*.js）。
 *    生产环境（EMAS 云函数）的 SECRET_KEY 从数据库 app_secret 集合读取，见 index.js。
 *
 * 用法：
 *   const { SECRET_KEY } = require('./test-config');
 *
 * 也可通过环境变量 SECRET_KEY 直接覆盖（优先级高于 env 文件）：
 *   SECRET_KEY=xxx node test/test.js emas photo.jpg
 */
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '../env') });

const SECRET_KEY = process.env.SECRET_KEY || '';

module.exports = { SECRET_KEY };
