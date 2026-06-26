import argparse
import os
import json
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import shutil
from tqdm import tqdm
import sys


def show_coco(dataset_dir, output, show, category_id):
    if not os.path.exists(output):
        os.mkdir(output)
    
    annotations_path = os.path.join(dataset_dir, 'annotations', 'instances_annotations.json')
    images_dir = os.path.join(dataset_dir, 'images')
    
    annotations_data = json.load(open(annotations_path, encoding='utf-8'))

    id2class_name = {}
    for cate_item in annotations_data['categories']:
        id2class_name[cate_item['id']] = cate_item['name']

    print('id2class_name:',id2class_name)
    if category_id == -1:
        output = os.path.join(output, 'NG')
    else:
        output = os.path.join(output, id2class_name[category_id])

    if not os.path.exists(output):
        os.mkdir(output)

    id2image = {}
    image2bbox = {}
    for img_item in annotations_data['images']:
        id2image[img_item['id']] = img_item['file_name']
        image2bbox[img_item['file_name']] = []
    for ann_item in annotations_data['annotations']:
        if category_id != -1:
            if ann_item['category_id'] != category_id:
                continue
        file_name = id2image[ann_item['image_id']]
        image2bbox[file_name].append(ann_item)

    for file_name, bboxes in tqdm(image2bbox.items()):
        if len(bboxes) == 0:
            continue
        image_path = os.path.join(images_dir, file_name)
        # print(image_path)
        # image = cv2.imread(image_path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)

        for ann_item in bboxes:
            x1, y1, w, h = map(int, ann_item['bbox'])
            x2 = x1 + w - 1
            y2 = y1 + h - 1
            cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 2)
            # cv2.putText(image, id2class_name[ann_item['category_id']], (int(x1), int(y1) - 2), 0, 2 / 3, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image)
            fontStyle = ImageFont.truetype("font/simsun.ttc", 25, encoding="utf-8")
            draw.text((int(x1), int(y1) - 25), id2class_name[ann_item['category_id']], (225, 255, 255), font=fontStyle)
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        
        
        if len(bboxes) > 0:
            # cv2.imwrite(os.path.join(output, file_name), image)
            cv2.imencode('.jpg', image)[1].tofile(os.path.join(output, file_name))
        if show:
            cv2.imshow('show', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def show_coco_ok(dataset_dir, output):
    output = os.path.join(output, 'OK')
    if not os.path.exists(output):
        os.mkdir(output)

    annotations_path = os.path.join(dataset_dir, 'annotations', 'instances_annotations.json')
    images_dir = os.path.join(dataset_dir, 'images')
    
    annotations_data = json.load(open(annotations_path, encoding='utf-8'))

    id2image = {}
    image2bbox = {}
    for img_item in annotations_data['images']:
        id2image[img_item['id']] = img_item['file_name']
        image2bbox[img_item['file_name']] = []
    for ann_item in annotations_data['annotations']:
        file_name = id2image[ann_item['image_id']]
        image2bbox[file_name].append(ann_item)

    for image_file, bbox in image2bbox.items():
        if len(bbox) > 0:
            continue
        shutil.copy(os.path.join(images_dir, image_file), output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 目录下有2个文件夹
    parser.add_argument('--coco_dataset', default=r'D:\WXWork\1688855031533875\Cache\File\2025-10\19号-25号指标漏检标注-1759044542494',type=str, help='coco dataset dir')
    parser.add_argument('--output',default=r'D:\WXWork\1688855031533875\Cache\File\2025-10\19号-25号指标漏检标注-1759044542494\show_coco', type=str, help='result output dir')
    parser.add_argument('--num_classes',default=19, type=int, help='number of classes')
    parser.add_argument('--show', action='store_true', help='show image')
    args = parser.parse_args()
    print(args.output)

    for category_id in range(1, args.num_classes+1):
        show_coco(args.coco_dataset, args.output, args.show, category_id)
    # show_coco(args.coco_dataset, args.output, args.show, -1)
    show_coco_ok(args.coco_dataset, args.output)