import os
import numpy as np
import cv2
import json
import shutil
from tqdm import tqdm

def read_image_chinese_path(img_path):
    # 以二进制模式读取文件
    with open(img_path, 'rb') as f:
        img_data = np.frombuffer(f.read(), dtype=np.uint8)
    # 解码图像
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    return img


def clamp(value,min_value,max_value):
    return max(min_value, min(value, max_value))

def yolo_bbox_to_rectangle(values, image_width, image_height):
    x_center, y_center, width, height = values

    x_min = (x_center - width / 2) * image_width
    y_min = (y_center - height / 2) * image_height
    x_max = (x_center + width / 2) * image_width
    y_max = (y_center + height / 2) * image_height

    x_min = clamp(x_min,0,image_width)
    y_min = clamp(y_min,0,image_height)
    x_max = clamp(x_max,0,image_width)
    y_max = clamp(y_max,0,image_height)

    return [[x_min,y_min], [x_max, y_max]]

def yolo_to_labelme(yolo_path,image_path,labelme_path,class_list):
    if not os.path.exists(labelme_path):
        os.mkdir(labelme_path)

    txt_files = [txt_name for txt_name in os.listdir(yolo_path) if '.txt' in txt_name]

    converted_count = 0
    for yolo_name in tqdm(txt_files):
        image = read_image_chinese_path(os.path.join(image_path,yolo_name.replace('.txt','.jpg')))
        image_height, image_width = image.shape[1],image.shape[0]
        image_height, image_width = image.shape[:2]
        image_name = yolo_name.replace(".txt",".jpg")
        shapes = []

        with open(os.path.join(yolo_path,yolo_name),"r",encoding="utf-8") as f:
            lines = f.readlines()

        for line_no, line in enumerate(lines, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            class_id = int(parts[0])
            values = [float(v) for v in parts[1:]]

            label = class_list[class_id]
            points = yolo_bbox_to_rectangle(values,image_width,image_height)
            shape_type = "rectangle"

            shapes.append({
                "label": label,
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": shape_type,
                "flags": {},
                "mask": None
            })

        labelme_data ={
            "version": "5.3.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": yolo_name.replace('.txt','.jpg'),
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width
        }

        json_file = os.path.join(labelme_path,image_name.replace(".jpg",".json"))
        with open(json_file,"w",encoding="utf-8") as f:
            json.dump(labelme_data,f,ensure_ascii=False,indent=4)

        shutil.copy2(os.path.join(image_path,image_name), os.path.join(labelme_path,image_name))
        converted_count +=1

    
    print(f"转换完成! 共处理 {converted_count} 个文件")
    print(f"输出目录: {labelme_path}")





if __name__ == "__main__":
    yolo_path = r"/Users/jodocls/Desktop/result/0626/labels"
    image_path = r"/Users/jodocls/Desktop/result/0626/images"
    labelme_path =r"/Users/jodocls/Desktop/result/0626/labelme"

    class_list = ['bucket', 'circle', 'handcart', 'pallet', 'two_piers', 'workbin']

    yolo_to_labelme(yolo_path,image_path,labelme_path,class_list)