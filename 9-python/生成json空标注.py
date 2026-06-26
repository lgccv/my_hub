import os
import os.path as osp
import cv2
import json
from tqdm import tqdm
import numpy as np
import shutil

def read_image_chinese_path(img_path):
    # 以二进制模式读取文件
    with open(img_path, 'rb') as f:
        img_data = np.frombuffer(f.read(), dtype=np.uint8)
    # 解码图像
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    return img

if __name__ == '__main__':
    img_dir = r"/Users/jodocls/Desktop/123/conver_image/4"
    img_list = os.listdir(img_dir)
    img_list = [img for img in img_list if img.endswith('.jpg')]
    for img in tqdm(img_list):
        try:
            # print('img:',img)
            img_path = osp.join(img_dir, img)
            # image = cv2.imdecode(np.fromfile(img_path), 1)
            image = read_image_chinese_path(img_path)
            # image = cv2.imread(img_path)
            h, w = image.shape[:2]

            shapes = []
            ann_json = {
                "version": "5.3.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": img,
                "imageData": None,
                "imageHeight": h,
                "imageWidth": w
            }

            json_save_path = osp.join(img_dir, img.replace('.jpg', '.json'))
            if osp.exists(json_save_path):
                continue
            with open(json_save_path, 'w') as f:
                json.dump(ann_json, f, indent=4)
            f.close()
        except:
            print(os.path.join(img_dir,img))
            shutil.move(os.path.join(img_dir,img),os.path.join(r'C:\Users\61082\Desktop\新建文件夹 (2)',img))
            # os.remove(os.path.join(img_dir,img))

