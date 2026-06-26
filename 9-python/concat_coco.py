import argparse
import os
import json
import shutil
import random
import math
import cv2
import numpy as np
import time
from tqdm import tqdm


def delete_nolabel_images(coco_dataset):
    annotations_path = os.path.join(coco_dataset, 'annotations', 'instances_annotations.json')
    annotations_data = json.load(open(annotations_path))
    
    id2image = {}
    image2bbox = {}
    for img_item in annotations_data['images']:
        id2image[img_item['id']] = img_item['file_name']
        image2bbox[img_item['file_name']] = []
    for ann_item in annotations_data['annotations']:
        file_name = id2image[ann_item['image_id']]
        image2bbox[file_name].append(ann_item)
    
    # 删除没有标签的图片信息
    images = []
    oldImgID2newImgID = {}
    image_id_cnt = 0
    for img_item in annotations_data['images']:
        if len(image2bbox[img_item['file_name']]) == 0:
            # 删除没有标签的图片
            os.remove(os.path.join(coco_dataset, 'images', img_item['file_name']))
            continue
        oldImgID2newImgID[img_item['id']] = image_id_cnt
        img_item['id'] = image_id_cnt
        images.append(img_item)
        image_id_cnt += 1
    annotations_data['images'] = images
    
    # 修改标签信息中的image_id
    for ann_item in annotations_data['annotations']:
        ann_item['image_id'] = oldImgID2newImgID[ann_item['image_id']]
    
    with open(annotations_path, 'w') as fp:
        json.dump(annotations_data, fp)
    print('Save annotation to {}'.format(annotations_path))


def split_train_test(ok_images, ng_images, split_ratio=0.9):
    ok_count = len(ok_images)
    ng_count = len(ng_images)
    train_images = random.sample(ok_images, math.ceil(ok_count*split_ratio)) + random.sample(ng_images, math.ceil(ng_count*split_ratio))
    test_images = list(set(ok_images + ng_images) - set(train_images))
    return train_images, test_images


def concat_coco(dataset_dirs, split_ratio, output_dir):
    t1 = time.time()
    output_images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    annotations_path = os.path.join(dataset_dirs[0], 'annotations', 'instances_annotations.json')
    all_annotations_data = json.load(open(annotations_path, encoding='utf-8'))
    # shutil.copytree(os.path.join(dataset_dirs[0], 'images'), output_images_dir, dirs_exist_ok=True)
    for img_item in all_annotations_data['images']:
        if not os.path.isfile(os.path.join(output_images_dir, img_item['file_name'])):
            shutil.copy(os.path.join(dataset_dirs[0], 'images', img_item['file_name']), output_images_dir)

    with open(os.path.join(output_dir, 'classes.txt'), 'w') as fp:
        print('classes num: ', len(all_annotations_data['categories']))
        for cate_item in all_annotations_data['categories']:
            print(cate_item['name'], " ", end='')
            fp.write('{}\n'.format(cate_item['name']))
        print()
    
    end_annotation_id = max(all_annotations_data['annotations'], key=lambda x:x['id'])['id']
    end_image_id = max(all_annotations_data['images'], key=lambda x:x['id'])['id']
    for dataset_dir in dataset_dirs[1:]:
        annotation_path = os.path.join(dataset_dir, 'annotations', 'instances_annotations.json')
        annotation_data = json.load(open(annotation_path, encoding='utf-8'))
        print(annotation_path)
        print(annotation_data['annotations'])
        c_start_annotation_id = min(annotation_data['annotations'], key=lambda x:x['id'])['id']
        c_start_image_id = min(annotation_data['images'], key=lambda x:x['id'])['id']
        for ann_item in annotation_data['annotations']:
            ann_item['id'] += (end_annotation_id - c_start_annotation_id + 1)
            ann_item['image_id'] += (end_image_id - c_start_image_id + 1)
            all_annotations_data['annotations'].append(ann_item)

        for img_item in annotation_data['images']:
            img_item['id'] += (end_image_id - c_start_image_id + 1)
            all_annotations_data['images'].append(img_item)

            shutil.copy(os.path.join(dataset_dir, 'images', img_item['file_name']), output_images_dir)
        
        # shutil.copytree(os.path.join(dataset_dir, 'images'), output_images_dir, dirs_exist_ok=True)
        end_annotation_id += len(annotation_data['annotations'])
        end_image_id += len(annotation_data['images'])
    t2 = time.time()
    print('>>> ', t2 - t1)

    id2image = {}
    image2bbox = {}
    for img_item in all_annotations_data['images']:
        id2image[img_item['id']] = img_item['file_name']
        image2bbox[img_item['file_name']] = []
    for ann_item in all_annotations_data['annotations']:
        file_name = id2image[ann_item['image_id']]
        image2bbox[file_name].append(ann_item)
    
    ok_images = []
    ng_images = []
    for image_file, bbox in image2bbox.items():
        if len(bbox) == 0:
            ok_images.append(image_file)
        else:
            ng_images.append(image_file)
    print("-"*60)
    print('ok count: ', len(ok_images))
    print('ng count: ', len(ng_images))
    print('image count: ', end_image_id + 1)
    print('bbox count: ', end_annotation_id + 1)
    print("-"*60)

    if split_ratio == 1.0:
        output_annotations_dir = os.path.join(output_dir, 'annotations')
        if not os.path.exists(output_annotations_dir):
            os.makedirs(output_annotations_dir)
        with open(os.path.join(output_annotations_dir, 'instances_annotations.json'), 'w') as fp:
            json.dump(all_annotations_data, fp)
    else:
        output_train_images_dir = os.path.join(output_dir, 'train', 'images')
        os.makedirs(output_train_images_dir, exist_ok=False)
        output_train_annotations_dir = os.path.join(output_dir, 'train', 'annotations')
        os.makedirs(output_train_annotations_dir, exist_ok=False)
        output_test_images_dir = os.path.join(output_dir, 'test', 'images')
        os.makedirs(output_test_images_dir, exist_ok=False)
        output_test_annotations_dir = os.path.join(output_dir, 'test', 'annotations')
        os.makedirs(output_test_annotations_dir, exist_ok=False)

        train_images, test_images = split_train_test(ok_images, ng_images, split_ratio)
        assert len(train_images) > 0, 'number of train image must be greater than 0'
        assert len(test_images) > 0, 'number of test image must be greater than 0'
        print('-'*60)
        print('train image count: ', len(train_images))
        print('test image count: ', len(test_images))
        print('-'*60)

        train_dataset = {'categories': [], 'annotations': [], 'images': []}
        test_dataset = {'categories': [], 'annotations': [], 'images': []}
        # 添加类别信息
        train_dataset['categories'] = all_annotations_data['categories']
        test_dataset['categories'] = all_annotations_data['categories']
        
        ann_id_cnt = 0
        for image_id, file_name in enumerate(tqdm(train_images)):
            shutil.copy(os.path.join(output_images_dir, file_name), output_train_images_dir)
            im = cv2.imdecode(np.fromfile(os.path.join(output_train_images_dir, file_name), dtype=np.uint8), cv2.IMREAD_COLOR)
            height, width, _ = im.shape
            # 添加图片信息
            train_dataset['images'].append({'file_name': file_name,
                                            'id': image_id,
                                            'width': width,
                                            'height': height})
            # 添加标注信息
            if len(image2bbox[file_name]) == 0:
                # 如没标签，跳过，只保留图片信息
                continue
            for ann_item in image2bbox[file_name]:
                ann_item['id'] = ann_id_cnt
                ann_item['image_id'] = image_id

                x1, y1, bbox_w, bbox_h = ann_item['bbox']
                x2 = x1 + bbox_w - 1
                y2 = x2 + bbox_h - 1
                x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
                if x1<0:
                    x1 = 0
                elif x1>width:
                    x1 = width-1
                if x2<0:
                    x2 = 0
                elif x2>width:
                    x2 = width-1
                if y1<0:
                    y1 = 0
                elif y1>height:
                    y1 = height-1
                if y2<0:
                    y2 = 0
                elif y2>height:
                    y2 = height-1
                bbox_w = max(0, x2 - x1)
                bbox_h = max(0, y2 - y1)
                if bbox_w * bbox_h == 0:
                    continue
                ann_item['area'] = bbox_w * bbox_h
                ann_item['bbox'] = [x1, y1, bbox_w, bbox_h]

                train_dataset['annotations'].append(ann_item)
                ann_id_cnt += 1
        with open(os.path.join(output_train_annotations_dir, 'instances_annotations.json'), 'w') as fp:
            json.dump(train_dataset, fp)
        print('Save annotation to {}'.format(output_train_annotations_dir))

        ann_id_cnt = 0
        for image_id, file_name in enumerate(tqdm(test_images)):
            shutil.copy(os.path.join(output_images_dir, file_name), output_test_images_dir)
            im = cv2.imdecode(np.fromfile(os.path.join(output_test_images_dir, file_name), dtype=np.uint8), cv2.IMREAD_COLOR)
            height, width, _ = im.shape
            # 添加图片信息
            test_dataset['images'].append({'file_name': file_name,
                                            'id': image_id,
                                            'width': width,
                                            'height': height})
            
            # 添加标注信息
            if len(image2bbox[file_name]) == 0:
                # 如没标签，跳过，只保留图片信息
                continue
            for ann_item in image2bbox[file_name]:
                ann_item['id'] = ann_id_cnt
                ann_item['image_id'] = image_id

                x1, y1, bbox_w, bbox_h = ann_item['bbox']
                x2 = x1 + bbox_w - 1
                y2 = x2 + bbox_h - 1
                x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
                if x1<0:
                    x1 = 0
                elif x1>width:
                    x1 = width-1
                if x2<0:
                    x2 = 0
                elif x2>width:
                    x2 = width-1
                if y1<0:
                    y1 = 0
                elif y1>height:
                    y1 = height-1
                if y2<0:
                    y2 = 0
                elif y2>height:
                    y2 = height-1
                bbox_w = max(0, x2 - x1)
                bbox_h = max(0, y2 - y1)
                if bbox_w * bbox_h == 0:
                    continue
                ann_item['area'] = bbox_w * bbox_h
                ann_item['bbox'] = [x1, y1, bbox_w, bbox_h]

                test_dataset['annotations'].append(ann_item)
                ann_id_cnt += 1
        with open(os.path.join(output_test_annotations_dir, 'instances_annotations.json'), 'w') as fp:
            json.dump(test_dataset, fp)
        print('Save annotation to {}'.format(output_test_annotations_dir))

        shutil.rmtree(output_images_dir)


def same_file_name(dataset_dirs):
    file_names = []
    for dataset_dir in dataset_dirs:
        file_names.extend(os.listdir(os.path.join(dataset_dir, 'images')))
    print(len(file_names))
    print(len(set(file_names)))
    same_file_names = []
    for file_name in set(file_names):
        if file_names.count(file_name) > 1:
            same_file_names.append(file_name)

    # 名字相同文件夹
    # same_images_dir = r'D:\10-dataset\jlc\otherImage\same_images'
    # os.makedirs(same_images_dir)
    # for file_name in same_file_names:
    #     for dataset_dir in dataset_dirs:
    #         if file_name in os.listdir(os.path.join(dataset_dir, 'images')):
    #             shutil.copy(os.path.join(dataset_dir, 'images', file_name), same_images_dir)
    #             break
    assert len(file_names) == len(set(file_names)), 'include same file name!!!'


# 路径下是images/annotations两个文件夹
dataset_dirs = [
    r'D:\10-dataset\jlc\otherImage\coco_20642\dataset1',
    r'D:\10-dataset\jlc\otherImage\coco_20642\dataset2\1218-1702881442604\0.1',
    r'D:\10-dataset\jlc\otherImage\coco_20642\dataset2\1218标注任务-1702892566452\0.1'
]









if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_ratio', type=float, default=1.0, help='train dataset ratio')
    # 不改变原始数据
    parser.add_argument('--output_dir', type=str,default=r'D:\10-dataset\jlc\otherImage\coco_20642\dataset', help='output coco dataset dir')

    args = parser.parse_args()
    
    same_file_name(dataset_dirs)
    
    concat_coco(dataset_dirs, args.split_ratio, args.output_dir)

    # 过滤没有标签图片
    # if args.split_ratio == 1.0:
    #     delete_nolabel_images(args.output_dir)
    # else:
    #     delete_nolabel_images(os.path.join(args.output_dir, 'train'))
    #     delete_nolabel_images(os.path.join(args.output_dir, 'test'))