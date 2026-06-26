import json
import os
import os.path as osp
from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np
import cv2
#import mmcv
from PIL import Image
import shutil

# img_path = r'C:\Users\61082\Desktop\label\图像\-9223372036854773339.jpg'
# target_dir = r'C:\Users\61082\Desktop\label\imcode_img.jpg'
# shutil.copy2(img_path,target_dir)
# exit()


# 图片不存在的
no_image = []
# json不存在的
no_json = []

Image_Dir = []
CoCo_Dir = []
#
Image_Dir = [r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250722\images',
             r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250724\images',
             r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250725\images',
             r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250726\images',
             r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250727\images',
             r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250728\images',
             r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250729\images',
             r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250730\images',
             ]

CoCo_Dir = [r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250722\annotations\instances_annotations.json',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250724\annotations\instances_annotations.json',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250725\annotations\instances_annotations.json',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250726\annotations\instances_annotations.json',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250727\annotations\instances_annotations.json',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250728\annotations\instances_annotations.json',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250729\annotations\instances_annotations.json',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250730\annotations\instances_annotations.json',
            ]

# #
save_dir = [r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250722\labelme',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250724\labelme',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250725\labelme',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250726\labelme',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250727\labelme',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250728\labelme',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250729\labelme',
            r'D:\2-Project\8-yazi\part_no数据(1)\AI漏检-250730\labelme',
            ]

Image_Dir = [r'D:\2-Project\33-lancheng\images']
CoCo_Dir = [r'D:\2-Project\33-lancheng\annotations\instances_annotations.json']
save_dir = [r'D:\2-Project\33-lancheng\labelme']

# 删除说有的labelme文件夹
for name in save_dir:
    if os.path.exists(name):
        # 删除文件夹及其中所有内容
        shutil.rmtree(name)
        print(f"文件夹 {name} 及其内容已删除")
    else:
        print(f"文件夹 {name} 不存在")


for img_dir_index in range(len(Image_Dir)):
    img_dir = Image_Dir[img_dir_index]
    os.makedirs(save_dir[img_dir_index], exist_ok=True)

    # 判断图像和json是否存在
    # 加载COCO JSON文件
    coco = COCO(CoCo_Dir[img_dir_index])

    categories = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}
    print('categories',coco.loadCats(coco.getCatIds()))

    # 加载图像信息
    image_ids = coco.getImgIds()

    for img_id in tqdm(image_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(img_dir, img_info['file_name'])
        if not osp.exists(img_path):
            no_image.append(img_path)
            continue
        #assert osp.exists(img_path), Image.open(img_path)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        shapes = []

        lable_name = []
        for ann in annotations:
            # 将名字转化为中文
            lable_name.append(categories[ann['category_id']])
            bbox = ann['bbox']
            shape = {
                'label': categories[ann['category_id']],
                'points': [
                    [bbox[0], bbox[1]],
                    [bbox[0] + bbox[2], bbox[1] + bbox[3]]
                ],
                'group_id': None,
                "description": "",
                'shape_type': 'rectangle',
                'flags': {}
            }
            shapes.append(shape)
        ann_json = {
                "version": "5.3.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": img_info['file_name'],
                "imageData": None,
                "imageHeight": img_info['height'],
                "imageWidth": img_info['width']
        }

        image_path = os.path.join(img_dir, img_info['file_name'])
        json_save_path = os.path.join(save_dir[img_dir_index], img_info['file_name'].replace('.jpg', '.json'))
        if not os.path.exists(json_save_path):
            with open(json_save_path, 'w',encoding='utf-8') as f:
                json.dump(ann_json, f, indent=4)
            f.close()
            imagepath = json_save_path.replace('.json','.jpg')
            shutil.copy2(img_path, imagepath)
        else:
            print('new')
            new_json_save_path = json_save_path[0:-5]+'_'+str(img_id)+'.json'
            with open(new_json_save_path, 'w',encoding='utf-8') as f:
                json.dump(ann_json, f, indent=4)
            f.close()
            new_imagepath = new_json_save_path.replace('.json','.jpg')
            shutil.copy2(img_path, new_imagepath)

print("done ! ! !")
print('no_image:',no_image)
