import fs from 'fs';
// import dotenv from 'dotenv';
import { Cloud } from "laf-client-sdk";
import { stdlog, downloadCosPath, getCos, getRegionBucketPath, clearLine } from './utils';
import { config } from "./config";

// dotenv.config({ path: './env' });

// 初始化laf链接
const cloud = new Cloud({
    baseUrl: `https://${config.LAF_APPID}.laf.run`,   // <APP_ID> 在首页应用列表获取
    getAccessToken: () => "",
    dbProxyUrl: "/proxy/catface",
});
const db = cloud.database();


async function main() {
    stdlog("Scaning local files...", "yellow");
    let localPhotos: {
        [file: string]: {
            catId: string,
            cloudStatusCheck: boolean
        } | undefined
    } = {};
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

    let photoCount = (await db.collection("photo").count()).total;
    if (!photoCount) {
        stdlog("Failed count photos.\n", "red");
        process.exit(1);
    }
    stdlog(`${photoCount} found in cloud database.\n`, "magenta");

    const MAX_LIMIT = 100;
    let photoNumQuery = Math.ceil(photoCount / MAX_LIMIT);
    let curNum = 0;
    let numDownload = 0;

    let cos :any = await getCos();

    for (let i = 0; i < photoNumQuery; i++) {
        let photos = (await db.collection("photo").skip(i * MAX_LIMIT).limit(MAX_LIMIT).get()).data;

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
                fs.mkdirSync(localDir);
            }
            let localPath = localDir + "/" + fileName;
            if (!fs.existsSync(localPath) || localPhotos[fileName] === undefined) {
                try {
                    if (curNum % 100 === 0) {
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
    }

    stdlog("Cleaning...\n", "yellow");
    let numDelete = 0;
    let numCatDelete = 0;
    for (let file in localPhotos) {
        let photo = localPhotos[file]!;
        if (photo.cloudStatusCheck) continue;

        let dir = "./photos/" + photo.catId;
        let path = dir + "/" + file;
        fs.rmSync(path, {recursive: true});
        numDelete++;

        if (fs.readdirSync(dir).length === 0) {
            fs.rmSync(dir, {recursive: true});
            numCatDelete++;
        }
    }

    stdlog(`Done. ${numDownload} photos downloaded, ${numDelete} photos deleted. (${numCatDelete} cats deleted for no photo exists anymore.)\n`, "magenta");

    // 使用writeFile函数创建done文件
    fs.writeFile("./done", '', (err) => {
        if (err) throw err;
    });
}

main();

// process.on('uncaughtException', (err) => {
//     console.error('Uncaught Exception:', err);
//     // 可以在这里添加一些错误处理逻辑
// });

// process.on('unhandledRejection', (reason, promise) => {
//     console.error('Unhandled Rejection:', reason);
//     // 可以在这里添加一些错误处理逻辑
// });

// let retryTimes = 50;

// while (retryTimes) {
//     try {
//         main();
//         stdlog("done");
//         break;
//     } catch (err) {
//         retryTimes --;
//         stdlog(`err: ${err}\nretry, last ${retryTimes}...`, "yellow");
//     }
// }
