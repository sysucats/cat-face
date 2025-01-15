from PIL import Image
import os
from sklearn.model_selection import train_test_split
import argparse
import json
import random
import shutil

parser = argparse.ArgumentParser(description="Cat Recognize Data Preprocessor")
parser.add_argument(
    "--source",
    default="data/crop_photos",
    type=str,
    help="photo data directory (default: data/crop_photos)",
)
parser.add_argument("--name", default="cat", type=str, help="name of dataset")
parser.add_argument("--size", default=256, type=int, help="image size (default: 256)")
parser.add_argument(
    "--filter",
    default=10,
    type=int,
    help="cats whose number of photos is less than this value will be filtered (default: 10)",
)
args = parser.parse_args()

# 定义数据集目录和图片源目录
dataset_dir = f"data/dataset-{args.name}"
source_dir = args.source
image_size = (args.size, args.size)  # 目标图片尺寸

# 创建dataset目录
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# 创建train, test, val目录
train_dir = os.path.join(dataset_dir, "train")
test_dir = os.path.join(dataset_dir, "test")
val_dir = os.path.join(dataset_dir, "val")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取所有种类的文件夹
categories = [
    d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))
]
ids = []


# 定义一个函数来缩放图片
def resize_image(image_path, output_path, size):
    try:
        with Image.open(image_path) as img:
            img = img.resize(size, Image.LANCZOS)  # 使用Image.LANCZOS
            img.save(output_path)
    except Exception as e:
        print(e)


# 遍历每个种类的文件夹
for category in categories:
    # 获取该种类下的所有图片
    category_path = os.path.join(source_dir, category)
    images = [
        f
        for f in os.listdir(category_path)
        if os.path.isfile(os.path.join(category_path, f))
    ]

    # 如果图片数量不足 args.filter，则进行补充
    if len(images) != 0 and len(images) < args.filter:
        print(f"Category '{category}' has less than {args.filter} images. Augmenting...")
        while len(images) < args.filter:
            # 随机选择一张图片进行复制
            random_image = random.choice(images)
            new_image_name = f"copy_{len(images)}_{random_image}"
            shutil.copy(
                os.path.join(category_path, random_image),
                os.path.join(category_path, new_image_name),
            )
            images.append(new_image_name)

    # 检查图片数量是否至少为args.filter张
    if len(images) < args.filter:
        print(f"Skipping category '{category}' with less than {args.filter} images.")
        continue  # 跳过该类别

    ids.append(category)

    # 划分训练集和剩余集
    train_images, remaining_images = train_test_split(
        images, test_size=0.25, random_state=42
    )

    # 划分测试集和验证集
    test_images, val_images = train_test_split(
        remaining_images, test_size=0.5, random_state=42
    )

    # 创建种类对应的train, test, val目录
    for dataset_dir, dataset_images in zip(
        [train_dir, test_dir, val_dir], [train_images, test_images, val_images]
    ):
        dataset_category_dir = os.path.join(
            dataset_dir, category, dataset_dir.split("/")[-1]
        )
        os.makedirs(dataset_category_dir, exist_ok=True)

        # 遍历图片并进行缩放和复制
        for image in dataset_images:
            input_image_path = os.path.join(category_path, image)
            output_image_path = os.path.join(dataset_category_dir, image)
            resize_image(input_image_path, output_image_path, image_size)  # 缩放图片

if not os.path.exists("export/"):
    os.mkdir("export/")

ids.sort()
with open(f"export/{args.name}.json", "w") as fp:
    json.dump(ids, fp)

print(
    f"Dataset {args.name} preparation completed with image resizing and splitting into train and test sets."
)
