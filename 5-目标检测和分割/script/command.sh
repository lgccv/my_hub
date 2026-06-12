# 复制数据--从本机到服务器
scp -P 22 -r  ./dataset_0325  std@10.10.16.58:/home/std/workspace-hub/lgc/dataset/PersonDetetionDataset   st
# 复制数据-- 从服务器到本机
scp -P 22 -r  std@10.10.16.58:/home/std/workspace-hub/lgc/dataset/PersonDetetionDataset . st
# 一键停止训练
pkill -KILL -f "train_detect.py|torch.distributed.run|/home/std/.config/Ultralytics/DDP/_temp_"

watch -n 0.5 nvidia-smi

# 行人检测项目
conda activate yolov11
nohup bash -c "python3 make_dataset/make_person_detect_dataset.py && ./tools/dist_train.sh ultralytics/cfg/datasets/person_detect.yaml 1 320 128 person_detect" > runs/log/person_detect.log 2>&1 & 
cp -r ./runs/detect/person_detect "models/person_detect_$(date +%Y%m%d%H%M)"
./tools/dist_pt_onnx.sh
./tools/dist_onnx_rknn.sh

# 行人检测项目s模型
nohup bash -c "python3 make_dataset/make_person_detect_dataset_s.py && ./tools/dist_train.sh ultralytics/cfg/datasets/person_detect_s.yaml 600 320 128 person_detect" > runs/log/person_detect.log 2>&1 & disown
cp -r ./runs/detect/person_detect "models/person_detect_$(date +%Y%m%d%H%M)"
./tools/dist_pt_onnx.sh
./tools/dist_onnx_rknn.sh
# 查看推理效果
python3 tools/train_detection/infer_by_pth.py
