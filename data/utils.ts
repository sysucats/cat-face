import fs from 'fs';
import request from "request";

import COS from 'cos-nodejs-sdk-v5';
import { config } from "./config";

export function stdlog(content: string, color: string = 'default') {
    // 定义颜色
    const colorsDict: { [key: string]: string } = {
        black: '\x1B[30m',
        red: '\x1B[31m',
        green: '\x1B[32m',
        yellow: '\x1B[33m',
        blue: '\x1B[34m',
        magenta: '\x1B[35m',
        cyan: '\x1B[36m',
        white: '\x1B[37m',
        default: '',
    };
    process.stdout.write(`${colorsDict[color]}${content}\x1B[0m`);
}

export function clearLine() {
    process.stdout.clearLine(1);
}

function _getCosKey() {
    return new Promise((resolve, reject) => {
        request.post({
            url: `https://${config.LAF_APPID}.laf.run/getTempCOS`,
        }, function (err, httpResponse, body) {
            if (err) {
                reject(err)
            } else {
                resolve(JSON.parse(body))
            }
        });
    });

}

export async function getCos() {
    let cosKey: any = await _getCosKey();
    var cos = new COS({
        getAuthorization: async function (options: Object, callback: Function) {
            // 初始化时不会调用，只有调用 cos 方法（例如 cos.putObject）时才会进入
            var cosTemp = cosKey;
            if (!cosTemp || !cosTemp.Credentials) {
                console.error("无效cosTemp信息: ", cosTemp)
                callback({
                    TmpSecretId: "empty",
                    TmpSecretKey: "empty",
                    SecurityToken: "empty",
                    ExpiredTime: "1111111111",
                });
                return;
            }

            callback({
                TmpSecretId: cosTemp.Credentials.TmpSecretId,        // 临时密钥的 tmpSecretId
                TmpSecretKey: cosTemp.Credentials.TmpSecretKey,      // 临时密钥的 tmpSecretKey
                SecurityToken: cosTemp.Credentials.Token,            // 临时密钥的 sessionToken
                ExpiredTime: cosTemp.ExpiredTime,                    // 临时密钥失效时间戳，是申请临时密钥时，时间戳加 durationSeconds
            });
        }
    });

    return cos;
}


export function downloadCosPath(cos: COS, path: string, localPath: string) {
    const pathObj = getRegionBucketPath(path);
    return new Promise((resolve, reject) => {
        cos.getObjectUrl(
            {
                Bucket: pathObj.bucket, /* 填入您自己的存储桶，必须字段 */
                Region: pathObj.region, /* 存储桶所在地域，例如 ap-beijing，必须字段 */
                Key: pathObj.filePath, /* 存储在桶里的对象键（例如1.jpg，a/b/test.txt），支持中文，必须字段 */
                Sign: true,
            },
            function (err, data) {
                if (err) {
                    stdlog(" downloadCosPath error");
                    reject(err);
                    return;
                }

                var writeStream = fs.createWriteStream(localPath);
                try {
                    request(data.Url).on('response', (response) => {
                        if (!/^image\//.test(response.headers['content-type'] as string)) {
                            writeStream.close();
                            fs.unlinkSync(localPath);
                            reject(new Error('Download failed: Unauthorized'));
                        }
                    })
                        .pipe(writeStream)
                        .on('finish', resolve)
                        .on('error', (error) => {
                            writeStream.close();
                            fs.unlinkSync(localPath);
                            stdlog(" downloadCosPath error");
                            reject(error);
                        });
                } catch (err) {
                    stdlog(" request err");
                    reject(err);
                }
            }
        );
    })
}


function _splitOnce(str: string, sep: string) {
    const idx = str.indexOf(sep);
    return [str.slice(0, idx), str.slice(idx + 1)];
}

// 提取COS的region和bucket字段
export function getRegionBucketPath(url: string) {
    // 返回：{region: 'ap-guangzhou', bucket: 'bucket-name', filePath: "xxx/xxx.xxx"}
    const regex = /http[s]*:\/\//i;
    const newUrl = url.replace(regex, '');
    const items = _splitOnce(newUrl, '/');
    const firstItems = items[0].split('.');

    if (firstItems[0] !== 'cos') {
        // 例如：https://bucket-name.cos.ap-guangzhou.myqcloud.com/sample.png
        return { region: firstItems[2], bucket: firstItems[0], filePath: items[1] }
    }

    // 例如：https://cos.ap-guangzhou.myqcloud.com/bucket-name/sample.png
    const path = _splitOnce(items[1], '/');
    return { region: firstItems[1], bucket: path[0], filePath: path[1] }
}
