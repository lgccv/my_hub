# conda activate rknn-toolkit2-2.3.2
# python3 ./deploy/onnx_to_rknn.py
import sys
from rknn.api import RKNN

if __name__ == "__main__":
    platform = "rk3588"
    onnx_path = "/home/std/workspace-hub/lgc/yolov26/runs/segment/train-5/weights/last.onnx"
    rknn_path = "/home/std/workspace-hub/lgc/yolov26/runs/segment/train-5/weights/last.rknn"
    do_quant = False
    DATASET_PATH = ""
    rknn = RKNN(verbose=False)
    rknn.config(target_platform=platform)
    ret = rknn.load_onnx(model=onnx_path)
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
