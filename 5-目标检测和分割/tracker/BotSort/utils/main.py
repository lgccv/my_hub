import os
import cv2
import sys
import argparse

# add path
import numpy as np
from rknn_executor import RKNN_model_container 

IMG_SIZE = (224, 224)  # (width, height), such as (1280, 736)

class Preprocess:
    def __init__(self,img_size):
        self.imgsz = img_size
        
    def __call__(self,crops,mean=(0,0,0),std=(1,1,1)):
        batch = []
        for crop in crops:
            if crop is None or crop.size == 0:
                raise ValueError("Empty crop found.")

            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            h, w = img.shape[:2]
            if h < w:
                new_h = self.imgsz
                new_w = int(round(w * self.imgsz / h))
            else:
                new_w = self.imgsz
                new_h = int(round(h * self.imgsz / w))

            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            top = (new_h - self.imgsz) // 2
            left = (new_w - self.imgsz) // 2
            img = img[top:top + self.imgsz, left:left + self.imgsz]

            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            mean = np.asarray(mean, dtype=np.float32)
            std = np.asarray(std, dtype=np.float32)
            img = (img - mean[:, None, None]) / std[:, None, None]

            batch.append(img)

        batch = np.stack(batch, axis=0).astype(np.float32)
        return batch


if __name__ == '__main__':

    # init model
    model_path = r"/home/pi/rknn/Easygo_base_0415/yolo11n-cls-embed.rknn"
    model = RKNN_model_container(model_path, "rk3588")

    img_path = r"/home/pi/rknn/Easygo_base_0415/1774939924_914488064_369.jpg"
    image = cv2.imread(img_path)
    dets = np.array([
        [150, 150, 100, 100],
    ], dtype=np.float32)

    dets[:,2:] = dets[:,2:]*1.02+10
    h, w = image.shape[:2]
    crops = []
    for det in dets:
        y1 = int(det[1] - det[3]/2)
        y2 = int(det[1] + det[3]/2)
        x1 = int(det[0] - det[2]/2)
        x2 = int(det[0] + det[2]/2)
        crops.append(image[y1:y2, x1:x2].copy())

    preprocess = Preprocess(224)

    batch = preprocess(crops)

    outputs = model.run(batch)

    print(outputs)


