import os
import shutil
import json
from tqdm import tqdm

label_file = r'/Users/jodocls/Desktop/123/conver_image/concat_image/labelme'
image_dir = r'/Users/jodocls/Desktop/123/conver_image/empty'


all_file = os.listdir(label_file)

image_file = []
json_file = []


image_file = [name for name in all_file if name.endswith('.jpg')]
json_file = [name for name in all_file if name.endswith('.json')]


for name in tqdm(json_file):
    label_file_ = json.load(open(os.path.join(label_file,name), 'r', encoding='UTF-8'))
    shapes = label_file_["shapes"]
    if len(shapes) ==0:
        shutil.move(os.path.join(label_file,name),os.path.join(image_dir,name))
        shutil.move(os.path.join(label_file,name.replace('json','jpg')),os.path.join(image_dir,name.replace('json','jpg')))


