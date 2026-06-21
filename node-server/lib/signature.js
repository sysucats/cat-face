const crypto = require('crypto');

/**
 * 签名校验，与 Python app.py 逐字节对齐
 * 算法：SHA256( base64(photo原始字节) + str(timestamp) + SECRET_KEY )
 * 并校验时间戳防重放：abs(timestamp - now) <= tolerant
 */
function checkSignature(photoBuffer, timestamp, signature, secretKey, tolerant) {
  const now = Math.floor(Date.now() / 1000);
  if (Math.abs(timestamp - now) > tolerant) return false;
  const b64 = photoBuffer.toString('base64');
  const sigData = b64 + String(timestamp) + secretKey;
  const hash = crypto.createHash('sha256').update(sigData, 'utf8').digest('hex');
  return hash === signature;
}

module.exports = { checkSignature };
