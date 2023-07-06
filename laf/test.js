const onnx = require('onnxruntime-node');
const fs = require('fs');
const jpeg = require('jpeg-js');
const sharp = require('sharp');

async function reshapeImage(fileName) {
  try {
    // 打开图像
    const { data, info } = await sharp(fileName)
      // 将图像reshape为640x640的大小
      .resize(640, 640)
      // 将图像转换为Buffer
      .raw()
      .toBuffer({ resolveWithObject: true });

    console.log(data.length);

    // 将Buffer转换为Float32Array

    const input = new Float32Array(data.length);
    for (let i = 0; i < 640 * 640; i++) {
      input[i] = data[i * 3] / 255.0;
      input[i + 640 * 640] = data[i * 3 + 1] / 255.0;
      input[i + 640 * 640 * 2] = data[i * 3 + 2] / 255.0;
    }

    return input;
  } catch (err) {
    console.error(err);
  }
}

async function readImage(path) {
  const imgData = fs.readFileSync(path);
  const img = jpeg.decode(imgData, true);

  const input = new Float32Array(img.width * img.height * 3);
  for (let i = 0; i < img.width * img.height; i++) {
    input[i] = img.data[i * 4] / 255.0;
    input[i + img.width * img.height] = img.data[i * 4 + 1] / 255.0;
    input[i + img.width * img.height * 2] = img.data[i * 4 + 2] / 255.0;
  }
  return input;
}

async function yoloCrop() {

  const modelName = "yolov5n.onnx";
  const modelInputShape = [1, 3, 640, 640];
  const topk = 10;
  const iouThreshold = 0.45;
  const confThreshold = 0.2;
  const classThreshold = 0.2;

  // create session
  const yolov5 = await onnx.InferenceSession.create("./models/yolov5n.onnx");
  const nms = await onnx.InferenceSession.create("./models/nms-yolov5.onnx");

  // 加载标签
  const labelNames = fs.readFileSync('./models/coco.names', 'utf-8').split('\n');

  const input = await reshapeImage('./data/two_cats_old.jpg');
  console.log(input);
  const inputTensor = new onnx.Tensor('float32', input, modelInputShape);

  const config = new onnx.Tensor("float32", new Float32Array([topk, iouThreshold, confThreshold]));
  const { output0 } = await yolov5.run({ images: inputTensor });
  const { selected_idx } = await nms.run({ detection: output0, config: config });

  let [xRatio, yRatio] = [1, 1];

  // looping through output
  let boxes = [];
  selected_idx.data.forEach((idx) => {
    const data = output0.data.slice(idx * output0.dims[2], (idx + 1) * output0.dims[2]); // get rows
    const [x, y, w, h] = data.slice(0, 4);
    const confidence = data[4]; // detection confidence
    const scores = data.slice(5); // classes probability scores
    let score = Math.max(...scores); // maximum probability scores
    const label = scores.indexOf(score); // class id of maximum probability scores
    score *= confidence; // multiply score by conf

    // filtering by score thresholds
    if (score >= classThreshold)
      boxes.push({
        label: label,
        labelName: labelNames[label],
        probability: score,
        bounding: [
          Math.floor((x - 0.5 * w) * xRatio), // left
          Math.floor((y - 0.5 * h) * yRatio), //top
          Math.floor(w * xRatio), // width
          Math.floor(h * yRatio), // height
        ],
      });
  });

  console.log(boxes);
}

async function main() {
  await yoloCrop();
}

main();
