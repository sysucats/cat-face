python data_preprocess.py
python data_split.py
python data_split.py --name fallback --source data/photos --size 512
python train.py
python train.py --data data/dataset-fallback --name fallback --size 512
rm -rf data/crop_photos
rm -rf data/dataset-fallback
rm -rf data/dataset-cat
# 遍历 data/photos 下所有文件夹，删除以 copy 开头的文件
find data/photos -type f -name "copy*" -exec rm -f {} \;