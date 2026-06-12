#!/usr/bin/env bash

CONFIG=$1
EPOCHS=$2
IMGSZ=$3
BATCH=$4
NAME=$5

python3 ./tools/train_detection/train_detect.py "$CONFIG" --epochs="$EPOCHS" --imgsz="$IMGSZ" --batch="$BATCH" --name="$NAME"