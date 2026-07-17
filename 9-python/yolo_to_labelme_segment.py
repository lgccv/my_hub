
import os
import numpy as np
import cv2
import json
import shutil
from tqdm import tqdm

yolo_path = r"/Users/jodocls/Desktop/123/autolabel/real/yolo"
labelme_path = r"/Users/jodocls/Desktop/123/autolabel/real/labelme_bake"
image_path = r"/Users/jodocls/Desktop/123/autolabel/real/images"
label_names = [
    "wheel-8cWi",
    "cardboard_smallbox",
    "gray_box",
    "guiding",
    "material_platform",
    "tag_code_m",
    "tray",
]

def read_image_chinese_path(img_path):
    # 以二进制模式读取文件
    with open(img_path, 'rb') as f:
        img_data = np.frombuffer(f.read(), dtype=np.uint8)
    # 解码图像
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    return img

for name in tqdm(os.listdir(yolo_path)):
    image_name = name.replace('.txt','.jpg')

    image = read_image_chinese_path(os.path.join(image_path,image_name))
    h, w = image.shape[:2]

    shapes = []

    # 开始组织shape
    with open(os.path.join(yolo_path,name),"r",encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        label = label_names[class_id]

        points = []
        for i in range(1,len(parts),2):
            x = float(parts[i]) *w
            y = float(parts[i+1]) *h
            points.append([x,y])
        shapes.append({
            "label": label,
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags":{},
            "mask":None
        })

    ann_json = {
        "version": "5.3.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }


    json_save_path = os.path.join(labelme_path, image_name.replace('.jpg', '.json'))
    with open(json_save_path, 'w') as f:
        json.dump(ann_json, f, indent=4)
    f.close()

    shutil.copy2(os.path.join(image_path,image_name),os.path.join(labelme_path,image_name))