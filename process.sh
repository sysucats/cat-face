python3 data_preprocess.py
python3 data_split.py
python3 data_split.py --name fallback --source data/photos --size 512
python3 train.py
python3 train.py --data data/dataset-fallback --name fallback --size 512