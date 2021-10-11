const tcb = require("@cloudbase/node-sdk");
const fs = require("fs");

const app = tcb.init({
    secretId: "xxx",
    secretKey: "xxx",
    env: "xxx"
});
const db = app.database();

async function main() {
    if (!fs.existsSync("./photos")) {
        fs.mkdirSync("./photos");
    }

    let photoCount = (await db.collection("photo").count()).total;
    process.stdout.write(`Count of photo collection is ${photoCount}. \n`);

    const MAX_LIMIT = 100;
    let photoNumQuery = Math.ceil(photoCount / MAX_LIMIT);
    let curNum = 0;

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
                    process.stdout.write("\033[32mdownloaded to " + localPath + ".\033[0m\n");
                } catch (err) {
                    process.stdout.write("\033[31mfailed download, code: " + err.code + ", message: " + err.message + ".\033[0m\n");
                }
            } else {
                process.stdout.write("\033[32malready downloaded at " + localPath + ".\033[0m\n");
            }
        }
    }
}

main();