import fs from 'fs';
import { stdlog, downloadCosPath, getCos, getRegionBucketPath, clearLine } from './utils';

import MPServerless from '@alicloud/mpserverless-node-sdk';
import { space_id, private_key, space_endpoint } from './config'

const mpServerless = new MPServerless({
    timeout: 60 * 1000,
    spaceId: space_id, // 服务空间标识
    endpoint: space_endpoint, // 服务空间地址，从小程序 serverless 控制台处获得
    serverSecret: private_key,
});

let cos :any;
let photoCount = 0, curNum = 0, numDownload = 0;
let localPhotos: {
    [file: string]: {
        catId: string,
        cloudStatusCheck: boolean
    } | undefined
} = {};

async function downloadPhotos(photos: Array<any>) {
    try {
        for (let photo of photos) {
            curNum++;
            stdlog(`\r[${curNum}/${photoCount}] `);

            let catId = photo.cat_id;
            let cloudPath = photo.photo_compressed;// photo.photo_compressed || photo.photo_id;
            stdlog("cat: ");
            stdlog(catId, "magenta");
            stdlog(", url: ");

            if (!cloudPath) {
                stdlog(", skipped for unavailable cloud path.", "red");
                clearLine();
                stdlog("\n");
                continue;
            }
            stdlog(getRegionBucketPath(cloudPath).filePath, "blue");

            let fileName = cloudPath.split("/").pop();

            let localDir = "./photos/" + catId;
            if (!fs.existsSync(localDir)) {
                try {
                    fs.mkdirSync(localDir);
                } catch (err) {
                    stdlog(`, failed create dir, message: ${err}.`, "red");
                    clearLine();
                    stdlog("\n");
                    continue;
                }
            }
            let localPath = localDir + "/" + fileName;
            if (!fs.existsSync(localPath) || localPhotos[fileName] === undefined) {
                try {
                    if (curNum % 5000 === 0) {
                        // 初始化cos
                        cos = await getCos();
                    }
                    await downloadCosPath(cos, cloudPath, localPath)
                    numDownload++;
                    stdlog(`, downloaded to ${localPath}`, "green");
                    clearLine();
                } catch (err) {
                    stdlog(`, failed download, message: ${err}.`, "red");
                    clearLine();
                    stdlog("\n");
                }
            } else {
                localPhotos[fileName]!.cloudStatusCheck = true;
                stdlog(`, already at ${localPath}`, "green");
                clearLine();
            }
        }
    } catch (err) {
        stdlog(`[Error] downloadPhotos failed: ${err}.`, "red");
        stdlog("\n");
    }
}


async function main() {
    try {
        stdlog("Scaning local files...", "yellow");
        if (!fs.existsSync("./photos")) {
            fs.mkdirSync("./photos");
        } else {
            for (let dir of fs.readdirSync("./photos", { withFileTypes: true })) {
                if (!dir.isDirectory()) continue;
                for (let file of fs.readdirSync("./photos/" + dir.name)) {
                    localPhotos[file] = {
                        catId: dir.name,
                        cloudStatusCheck: false,
                    }
                }
            }
        }
        stdlog(`${Object.keys(localPhotos).length} found.\n`, "yellow");

        photoCount = (await mpServerless.db.collection("photo").count({})).result;
        if (!photoCount) {
            stdlog("Failed count photos.\n", "red");
            process.exit(1);
        }
        stdlog(`${photoCount} found in cloud database.\n`, "magenta");

        const MAX_LIMIT = 1000;
        let photoNumQuery = Math.ceil(photoCount / MAX_LIMIT);
        cos = await getCos();


        for (let i = 0; i < photoNumQuery; i++) {
            try {
                let photos = (await mpServerless.db.collection("photo").find({}, {
                    skip: i * MAX_LIMIT,
                    limit: MAX_LIMIT,
                })).result;

                // 将 photos 均分为 10 份
                const chunkSize = Math.ceil(photos.length / 10);
                const chunks: any[][] = [];
                for (let j = 0; j < photos.length; j += chunkSize) {
                    chunks.push(photos.slice(j, j + chunkSize));
                }

                // 并行下载，确保每个chunk的下载失败不会影响其他chunk
                await Promise.all(chunks.map(chunk => 
                    downloadPhotos(chunk).catch(err => {
                        stdlog(`[Error] Chunk download failed: ${err}.`, "red");
                        stdlog("\n");
                    })
                ));
            } catch (err) {
                stdlog(`[Error] Batch ${i} failed: ${err}.`, "red");
                stdlog("\n");
            }
        }

        stdlog("Cleaning...\n", "yellow");
        let numDelete = 0;
        let numCatDelete = 0;
        for (let file in localPhotos) {
            let photo = localPhotos[file]!;
            if (photo.cloudStatusCheck) continue;

            let dir = "./photos/" + photo.catId;
            let path = dir + "/" + file;
            try {
                fs.rmSync(path, {recursive: true});
                numDelete++;

                if (fs.readdirSync(dir).length === 0) {
                    fs.rmSync(dir, {recursive: true});
                    numCatDelete++;
                }
            } catch (err) {
                stdlog(`[Error] Failed to delete ${path}: ${err}.`, "red");
                stdlog("\n");
            }
        }

        stdlog(`Done. ${numDownload} photos downloaded, ${numDelete} photos deleted. (${numCatDelete} cats deleted for no photo exists anymore.)\n`, "magenta");

        // 使用writeFile函数创建done文件
        fs.writeFile("./done", '', (err) => {
            if (err) stdlog(`[Error] Failed to create done file: ${err}.`, "red");
        });
    } catch (err) {
        stdlog(`[Fatal Error] Main process failed: ${err}.`, "red");
        stdlog("\n");
    }
}

main();
