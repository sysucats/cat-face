import os
from yolo_interface import yolov5m
from tqdm import tqdm
from PIL import Image

SRC = 'fetch-data/photos'
DEST = 'crop-photos'

def main():
    print("==> loading YOLOv5 model...")
    model = yolov5m()

    numPhotos = 0
    numSkippedPhotos = 0

    if not os.path.exists(DEST):
        os.mkdir(DEST)
    else:
        print('### destination path exists, continue...')

    print("==> processing photos...")
    for dirName in tqdm(os.listdir(SRC), leave=False, desc="processing"):
        srcPath = os.path.join(SRC, dirName)
        if os.path.isdir(srcPath) == False:
            continue

        destPath = os.path.join(DEST, dirName)
        if not os.path.isdir(destPath):
            os.mkdir(destPath)
        
        for fileName in tqdm(os.listdir(srcPath), leave=False, desc=dirName):
            numPhotos += 1

            srcFilePath = os.path.join(srcPath, fileName)
            destFilePath = os.path.join(destPath, fileName)
            # 使用 YOLOv5 进行目标检测，结果为[{xmin, ymin, xmax, ymax, confidence, class, name}]格式
            results = model(srcFilePath).pandas().xyxy[0].to_dict('records')
            # 过滤非cat目标
            catResults = list(filter(lambda target: target['name'] == 'cat', results))
            # 跳过图片内检测不到cat或有多个cat的图片
            if len(catResults) != 1:
                numSkippedPhotos += 1
                continue
            # 裁剪出cat
            catResult = catResults[0]
            cropBox = catResult['xmin'], catResult['ymin'], catResult['xmax'], catResult['ymax']
            im = Image.open(srcFilePath).crop(cropBox)
            if im.mode in ("RGBA", 'P'):
                im = im.convert('RGB')
            im.save(destFilePath, format='JPEG')
    
    print(f"==> done. {numPhotos - numSkippedPhotos} photos cropped, {numSkippedPhotos} photos skipped.")

if __name__ == "__main__":
    main()