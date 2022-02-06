import fs from 'fs';
import tcb from '@cloudbase/node-sdk';
import dotenv from 'dotenv';

dotenv.config();

const SECRET_ID = process.env.SECRET_ID;
const SECRET_KEY = process.env.SECRET_KEY;
const ENV = process.env.ENV;

const app = tcb.init({
    secretId: SECRET_ID,
    secretKey: SECRET_KEY,
    env: ENV
});
const db = app.database();

async function main() {
    process.stdout.write("\033[33mScaning local files... ");
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
    process.stdout.write(`${Object.keys(localPhotos).length} found.` + "\033[0m\n");

    let photoCount = (await db.collection("photo").count()).total;
    if (!photoCount) {
        process.stdout.write("\033[31mFailed count photos.\033[0m\n");
        process.exit(1);
    }
    process.stdout.write("\033[35m" + `${photoCount} found in cloud database.` + "\033[0m\n");

    const MAX_LIMIT = 100;
    let photoNumQuery = Math.ceil(photoCount / MAX_LIMIT);
    let curNum = 0;
    let numDownload = 0;

    for (let i = 0; i < photoNumQuery; i++) {
        let photos = (await db.collection("photo").skip(i * MAX_LIMIT).limit(MAX_LIMIT).get()).data;

        for (let photo of photos) {
            curNum++;
            process.stdout.write(`[${curNum}/${photoCount}] `);

            let catId = photo.cat_id;
            let cloudPath = photo.photo_compressed;// photo.photo_compressed || photo.photo_id;
            process.stdout.write("cat: \033[35m" + catId + "\033[0m ");
            process.stdout.write("photo cloud path: \033[34m" + cloudPath + "\033[0m ");

            if (!cloudPath) {
                process.stdout.write("\033[31mskipped for unavailable cloud path.\033[0m\n");
                continue;
            }

            let fileName = cloudPath.split("/").pop();

            let localDir = "./photos/" + catId;
            if (!fs.existsSync(localDir)) {
                fs.mkdirSync(localDir);
            }
            let localPath = localDir + "/" + fileName;
            if (!fs.existsSync(localPath)) {
                try {
                    await app.downloadFile({
                        fileID: cloudPath,
                        tempFilePath: localPath
                    });
                    numDownload++;
                    process.stdout.write("\033[32mdownloaded to " + localPath + ".\033[0m\n");
                } catch (err) {
                    const e = <tcb.IErrorInfo>err;
                    process.stdout.write("\033[31mfailed download, code: " + e.code + ", message: " + e.message + ".\033[0m\n");
                }
            } else {
                localPhotos[fileName]!.cloudStatusCheck = true;
                process.stdout.write("\033[32malready downloaded at " + localPath + ".\033[0m\n");
            }
        }
    }

    process.stdout.write("\033[33mCleaning...\033[0m\n");
    let numDelete = 0;
    let numCatDelete = 0;
    for (let file in localPhotos) {
        let photo = localPhotos[file]!;
        if (photo.cloudStatusCheck) continue;

        let dir = "./photos/" + photo.catId;
        let path = dir + "/" + file;
        fs.rmSync(path);
        numDelete++;

        if (fs.readdirSync(dir).length === 0) {
            fs.rmSync(dir);
            numCatDelete++;
        }
    }

    process.stdout.write("\033[35m" + `Done. ${numDownload} photos downloaded, ${numDelete} photos deleted. (${numCatDelete} cats deleted for no photo exists anymore.)` + "\033[0m\n");
}

main();