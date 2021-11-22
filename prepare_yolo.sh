#!/bin/bash

git submodule update --init
wget https://ultralytics.com/assets/Arial.ttf -O $HOME/.config/Ultralytics/Arial.ttf
wget https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt -O yolov5m.pt