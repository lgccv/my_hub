import sys
from rknn.api import RKNN

DATASET_PATH = '../../../datasets/COCO/coco_subset_20.txt'
DEFAULT_RKNN_PATH = '../model/yolo11.rknn'
DEFAULT_QUANT = False



if __name__ == '__main__':
    platform ="rk3588"
    model_path = r"/home/std/workspace-hub/lgc/yolov11/yolo11n-cls-embed.onnx"
    do_quant=False
    output_path =r"/home/std/workspace-hub/lgc/yolov11/yolo11n-cls-embed.rknn"

    rknn = RKNN(verbose=False)

    # Pre-process config
    print('--> Config model')
    rknn.config(target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
