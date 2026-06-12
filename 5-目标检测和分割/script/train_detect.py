from ultralytics import YOLO
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detetor")
    parser.add_argument('data', help='train config file path')
    parser.add_argument('--epochs', type=int,default=600,help='train epoch')
    parser.add_argument('--imgsz', type=int,default=320,help='train image size')
    parser.add_argument('--batch', type=int,default=128,help='batch size')
    parser.add_argument('--name', type=str,default='train',help='fold name')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # Load a pretrained YOLO11n model
    model = YOLO("/home/std/workspace-hub/lgc/yolov11/yolov8s.pt")
    print("args_data:",args.data)

    train_results = model.train(
        data = args.data,  # Path to dataset configuration file
        epochs= args.epochs,  # Number of training epochs
        imgsz= args.imgsz,  # Image size for training
        batch = args.batch,
        name = args.name,  # 输出文件夹名字
        exist_ok = True,     # 覆盖原文件
        device=[0,1]  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )


if __name__ == "__main__":
    main()