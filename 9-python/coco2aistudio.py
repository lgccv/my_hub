import argparse
import os
import json
import random
import numpy as np
from lxml import etree, objectify

COCO_START_IMAGEID = 0

def create_random_color():
    rgb = [random.randint(20, 225), random.randint(20, 225), random.randint(20, 225)]
    color = '#'
    for item in rgb:
        color += hex(item)[2:]
    return color
    

def create_meta_tree(labels):
    E = objectify.ElementMaker(annotate=False)
    lables_tree = E.labels()
    for label in labels:
        lables_tree.append(
            E.label(
                E.name(label),
                E.color(create_random_color()),
                E.attributes('')
            )
        )
    task_tree = E.task(
        E.id(''),
        E.name(''),
        E.size(''),
        E.mode(''),
        E.overlap(''),
        E.bugtracker(''),
        E.created(''),
        E.start_frame(''),
        E.stop_frame(''),
        E.frame_filter(''),
        lables_tree,
        E.segments(
            E.segment(
                E.id(''),
                E.start(''),
                E.stop(''),
                E.url('')
            )
        ),
        E.owner(
            E.username(''),
            E.email('')
        ),
        E.assignee('')
    )
    meta_tree = E.meta(
        task_tree,
        E.dumped('')
    )
    return meta_tree


def create_version_tree(version=''):
    version_tree = etree.Element("version")
    version_tree.text = version
    return version_tree


def coco2aistudio(json_file, xml_file):
    xml_dirname = os.path.dirname(xml_file)
    if not os.path.exists(xml_dirname):
        os.makedirs(xml_dirname, exist_ok=True)
    
    coco_ann_data = json.load(open(json_file, 'r',encoding='UTF-8'))
    
    COCO_START_IMAGEID = min(coco_ann_data['images'], key=lambda x:x['id'])['id']

    id_map = {}
    for item in coco_ann_data['categories']:
        id_map[item['id']] = item['name']

    annotations = {}
    for ann in coco_ann_data['annotations']:
        image_id = ann['image_id']
        annotations.setdefault(image_id, []).append(ann)
    
    E = objectify.ElementMaker(annotate=False)
    annotations_tree = E.annotations()
    version_tree = create_version_tree()
    meta_tree = create_meta_tree(id_map.values())
    annotations_tree.append(version_tree)
    annotations_tree.append(meta_tree)

    for img in coco_ann_data['images']:
        img_id = img['id']
        img_file_name = img['file_name']
        img_width = img['width']
        img_height = img['height']

        attributes = {
            'id': str(img_id - COCO_START_IMAGEID),
            'name': img_file_name,
            'width': str(img_width),
            'height': str(img_height)
        }
        image_tree = etree.Element("image", attributes)

        ann_list = annotations.get(img_id, [])
        for ann in ann_list:
            bbox = ann['bbox']
            bbox[2:4] = np.sum([bbox[:2], bbox[2:4]], axis=0).tolist()
            bbox = list(map(str, bbox))
            attributes = {
                'label': id_map[ann['category_id']],
                'occluded': '0',
                'source': 'manual',
                'xtl': bbox[0],
                'ytl': bbox[1],
                'xbr': bbox[2],
                'ybr': bbox[3],
                'z_order': "0"
            }
            box_tree = etree.Element("box", attributes)
            box_tree.text = ''
            image_tree.append(box_tree)
        if len(ann_list) == 0:
            image_tree.text = ''
        annotations_tree.append(image_tree)
    # etree.ElementTree(annotations_tree).write(xml_file, encoding="utf-8", pretty_print=True)
    etree.ElementTree(annotations_tree).write(xml_file, encoding="utf-8", pretty_print=True)


# python coco2aistudio.py D:\workspace\active_learning\baseline\annotations\test.json D:\workspace\active_learning\baseline\annotations.xml
# python coco2aistudio.py D:\10-dataset\jlc\jlc_dataset\annotations\instances_annotations.json D:\10-dataset\jlc\jlc_dataset\annotations\instances_annotations.xml
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert COCO annotation to AIStudio format.'
    )

    parser.add_argument('--json_file', type=str, default =r'C:\Users\61082\Desktop\omoron\jlc\13_14_15_16_17_loujianfenxi\biaozhu_images\images\diedai_labelme\annotations\instances_annotations.json',help='COCO format dataset json file')
    parser.add_argument('--xml_file', type=str,default=r'C:\Users\61082\Desktop\omoron\jlc\13_14_15_16_17_loujianfenxi\biaozhu_images\images\diedai_labelme\annotations\annotations.xml', help='AIStudio format dataset annotations xml file')

    args = parser.parse_args()

    coco2aistudio(args.json_file, args.xml_file)