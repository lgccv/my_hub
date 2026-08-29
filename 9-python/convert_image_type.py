import os
import cv2
from tqdm import tqdm
import re
import json

orig_image = r"/Users/jodocls/Desktop/result/result3/steel/save"
labelme_image = r"/Users/jodocls/Desktop/result/result3/steel/images"

def convert_image_type():
    orig_image = r"/Users/jodocls/Desktop/result/result3/steel/save"
    labelme_image = r"/Users/jodocls/Desktop/result/result3/steel/images"
    json_name = [name for name in os.listdir(orig_image) if name.endswith('.png')]
    for name in tqdm(json_name):
        image = cv2.imread(os.path.join(orig_image,name))
        if name.endswith('.png'):
            name = name.replace('.png','.jpg')
        cv2.imwrite(os.path.join(labelme_image,name),image)

def convert_image_name():
    image_path = "/Users/jodocls/Desktop/123/conver_image/4"
    image_name = [name for name in os.listdir(image_path) if '.jpg' in name]
    for name in tqdm(image_name):
        new_name = name.replace(" ", "").replace("(", "").replace(")", "")
        os.rename(os.path.join(image_path,name),os.path.join(image_path,new_name))
        os.rename(os.path.join(image_path,name.replace('.jpg','.json')),os.path.join(image_path,new_name.replace('.jpg','.json')))

def add_image_prefix():
    prefix = "20260818_"
    image_path = "/Users/jodocls/Desktop/result/0818/origin_image"
    image_name = [name for name in os.listdir(image_path) if '.jpg' in name]
    for name in tqdm(image_name):
        # new_name = name.replace('.jpg',prefix)
        new_name = prefix + name
        os.rename(os.path.join(image_path,name),os.path.join(image_path,new_name))
        # os.rename(os.path.join(image_path,name.replace('.jpg','.json')),os.path.join(image_path,new_name.replace('.jpg','.json')))

def rename_json_name():
    image_path = "/Users/jodocls/Desktop/123/autolabel/train/labelme"
    labelme_image = r"/Users/jodocls/Desktop/123/autolabel/train/labelme_jpg"
    json_name = [name for name in os.listdir(image_path) if '.json' in name]
    for name in json_name:
        print(name)
        labelme_file = json.load(open(os.path.join(image_path,name),'r',encoding='UTF-8'))
        labelme_file['imagePath'] = name.replace('.json','.jpg')
        with open(os.path.join(labelme_image,name), "w",encoding='UTF-8') as f:
            json.dump(labelme_file, f,indent=4)

def convert_image_size():
    image_path = r"/Users/jodocls/Desktop/result/0818/image"
    resize_image_path = r"/Users/jodocls/Desktop/result/0818/labelme"
    image_name = [name for name in os.listdir(image_path) if '.jpg' in name]
    for name in image_name:
        image = cv2.imread(os.path.join(image_path,name))
        size_image = cv2.resize(image,(320,240))
        cv2.imwrite(os.path.join(resize_image_path,name),size_image)


if __name__ == "__main__":
    convert_image_size()
