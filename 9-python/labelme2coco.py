# 各个项目标签名字:
# 精视项目:
label_name = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '短路', '立碑', '翘脚', '错件', '锡珠', '虚焊']
our_to_jingshi_label={"侧立":"侧立","偏移":"偏移","反向":"反向","反白":"翻件","少件":"少件","少锡":"少锡","异物":"异物","引脚错位":"引脚错位","损件":"损件","短路":"短路","立碑":"立碑","翘脚":"翘脚","错件":"错件","锡珠":"锡珠","虚焊":"虚焊"}
# jingshi_to_our_label={"侧立":"侧立","偏移":"偏移","反向":"反向","翻件":"反白","少件":"少件","少锡":"少锡","异物":"异物","引脚错位":"引脚错位","损件":"损件","短路":"短路","立碑":"立碑","翘脚":"翘脚","错件":"错件","锡珠":"锡珠",["虚焊","假焊"]:"虚焊"}

### 星晨项目:不要映射
label_name = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '短路', '立碑', '翘脚', '错件', '锡珠', '虚焊']

### 众诚项目:
label_name = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '短路', '立碑', '翘脚', '错件', '锡珠', '虚焊']
our_to_zhongcheng_label={"侧立":"转角","偏移":"偏移","反向":"极性","反白":"反白","少件":"缺件","少锡":"少锡","异物":"污物","引脚错位":"引脚错位","损件":"破损","短路":"桥接","立碑":"立碑","翘脚":"翘脚","错件":"错件","锡珠":"锡珠","虚焊":"虚焊"}

### 欧姆龙标签:
label_name = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '未过板', '短路', '立碑', '翘脚', '虚焊', '错件', '锡珠']
our_to_omoron_label = {"OK":"OK","侧立":"其它","偏移":"偏移","反向":"反向","反白":"翻件","少件":"漏件","少锡":"锡少","异物":"锡多","引脚错位":"偏移","损件":"损件","未过板":"未过板","短路":"短路","立碑":"立碑","翘脚":"翘脚","虚焊":"虚焊","错件":"错件","锡珠":"锡珠"}


## 嘉立创标签
#!/usr/bin/env python
import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import cv2
import random
import imgviz
import numpy as np
import labelme
from tqdm import tqdm
import shutil

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)

# 欧姆龙标签:
our_to_omoron_label = {"少锡":"锡少","偏移":"偏移","少件":"漏件","未过板":"未过板","反向":"反向","翘脚":"翘脚","错件":"错件","反白":"翻件","短路":"短路","锡珠":"锡珠","异物":"锡多","损件":"损件","立碑":"立碑","引脚错位":"引脚错位"}
omoron_to_our_label = {"锡少":"少锡","偏移":"偏移","漏件":"少件","未过板":"未过板","反向":"反向","翘脚":"翘脚","错件":"错件","翻件":"反白","短路":"短路","锡珠":"锡珠","锡多":"异物","损件":"损件","立碑":"立碑","引脚错位":"引脚错位"}

# 众诚标签
our_to_zhongcheng_label={"短路":"桥接","反向":"极性","少件":"缺件","偏位":"偏移","立碑":"立碑","侧立":"转角","异物":"污物","少锡":"少锡","损件":"破损","错件":"错件","反白":"反白","翘脚":"翘脚","引脚错位":"偏移","锡珠":"锡珠"}
zhongcheng_to_our_label={"桥接":"短路","极性":"反向","缺件":"少件","偏移":"偏位","立碑":"立碑","转角":"侧立","污物":"异物","少锡":"少锡","破损":"损件","错件":"错件","反白":"反白","翘脚":"翘脚","引脚错位":"偏移","锡珠":"锡珠"}


def walk_file(path, type=".json"):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(type):
                f_path = os.path.join(root, file)
                if osp.exists(f_path) and osp.exists(f_path.replace('.json', '.jpg')):
                    yield f_path

def check_points(points):
    x0, y0 = points[0]
    x1, y1 = points[1]
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return [[x0, y0], [x1, y1]]

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --input_dir  图片和labelme的路径
    # --output_dir coc的保存路径
    # labels:注意第一行是__ignore__
    # 要改
    startIndex = 1
    # label_name = ['损件', '露铜', '极反', '多件', '错料', '反向', '少锡', '侧立', '立碑', '反白', '翘脚', '短路','少件', '偏位','器件连接','引脚错位']
    # 嘉立创珠海
    label_name = ['短路', '器件连接', '少件', '偏位', '立碑','侧立', '异物', '少锡', '损件', '多件','反白', '翘脚', '引脚错位', '其他']

    # 嘉立创一期二期(韶关)
    label_name = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '短路', '立碑', '翘脚', '错件', '锡珠', '虚焊']
    ################## 雅智
    # label_name = ['位移', '侧立', '反贴', '少锡', '异物', '文字错误', '极性','浮起', '短路', '立碑', '缺件', '规格错误', '锡点']
    # label_name = ['极性','丝印']
    # label_name = ['器件', '器件+焊盘']
    # label_name = ['划痕', '多墨', '墨点', '条码墨桥', '条码文字白点', '脏污']
    # label_name = ['断线','浓淡不均','墨点','划痕','破损','脏污','其它']
    # label_name = ['位移','短路']
    # label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    # label_name = ['反贴','正常']
    # label_name = ['anhen', 'bainingjiao', 'ganlieningjiao', 'heidian', 'liangdian', 'lianghen', 'posunningjiao', 'qipao']
    # 欧姆龙
    label_name = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '未过板', '短路', '立碑', '翘脚', '错件', '锡珠','虚焊']
    # 美陆
    # label_name = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '短路', '立碑', '翘脚', '错件', '锡珠', '虚焊']
    # 众诚项目
    # label_name = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '短路', '立碑', '翘脚', '错件', '锡珠', '虚焊']
    #  精视项目
    # label_name = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '短路', '立碑', '翘脚', '错件', '锡珠', '虚焊']
    #  星晨项目
    # label_name = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '短路', '立碑', '翘脚', '错件', '锡珠', '虚焊']

    # 星晨定位模型
    # label_name = ['component', 'component_pin_solder_pad', 'pin', 'printed_text', 'solder_pad']
    # 晶圆检测:
    # label_name = ['崩边','划痕','脏污','探针']

    # 硅翔检测模型:
    # label_name = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '短路', '立碑', '翘脚', '错件', '锡珠', '虚焊']

    # 硅翔定位模型:
    # label_name = ['器件','焊盘','标记点']

    # 雅智二期
    label_name = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '短路', '立碑', '翘脚', '错件', '锡珠', '虚焊']

    # label_name = ['异物']
    # label_name = ['component', 'chip', 'polarity', 'printedtext', 'solder pad', 'mld', 'cae', 'led', 'bga', 'xtal', 'ocs', 'sod', 'so', 'qfp', 'qfn', 'dfn', 'lcc', 'fpc', 'to', 'sot', 'sop', 'qita']
    # label_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', 'C', 'F', 'H', 'J', 'K', 'M', 'P', 'R', 'T', 'U', 'X', 'Y', 'a', 'b', 'c', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'o', 'r', 't', 'u', 'v', 'w', 'x', 'y']
    # 一准检测模型
    # label_name = ['zangwu', 'huichen', 'bengbian', 'huahen']
    # label_name = ['脏污','崩边','划痕']
    # 一准定位模型
    # label_name = ['波导区', '光口区', 'pad区', '重点区']
    # label_name = ['芯片全面积']
    # 兰晨项目
    # label_name = ['偏位', '少件', '短路', '翘脚', '反白', '空焊', '立碑', '侧立', '少锡', '反向', '错料', '多件', '损件', '排插错PIN', '其他', '多锡','异物','偏移']
    # label_name = ['侧立', '偏移', '反向', '反白', '少件', '少锡', '异物', '引脚错位', '损件', '短路', '立碑', '翘脚', '错件', '锡珠', '虚焊']
    # 博瑞通项目
    # label_name = ['短路', '其他', '缺件', '歪斜', '器件连接', '少锡', '偏位', '包焊', '破损', '脚长', '锡洞', '异物', '多锡', '引脚错位']

    parser.add_argument("--input_dir", default=r"C:\Users\61082\Desktop\omoron\yz_erqi", help="input annotated directory")
    parser.add_argument("--output_dir", default=r"C:\Users\61082\Desktop\omoron\yz_erqi\annotations", help="output dataset directory")
    # parser.add_argument("--labels", default=r"D:\10-dataset\jlc\otherImage\coco_20642\1218-1702881442604\0.1\classes.txt", help="labels file")
    parser.add_argument(
        "--noviz", default=True, help="no visualizationd", action="store_true"
    )
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Creating dataset:", args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        licenses=[
            {
                "url":"",
                "id":0,
                "name":""
            }
        ],
        info={
            "contributor": "",
            "date_created": "",
            "description":"",
            "url":"",
            "version":"",
            "year":""
        },
        categories=[
            # supercategory, id, name
        ],

        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
    )

    class_name_to_id = {}

    for i, line in enumerate(label_name):
        class_id = i  # starts with -1
        class_name = line
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            {
                "id":class_id+startIndex,  # id从1开始
                "name":class_name,
                "supercategory":""
            }
        )

    out_ann_file = osp.join(args.output_dir, "instances_annotations.json")

    label_files = list(walk_file(args.input_dir))

    ng_images = []
    ng_labels = []
    for image_id, filename in enumerate(tqdm(label_files)):
        assert os.path.exists(filename)
        # print(filename)
        label_file = json.load(open(filename, 'r',encoding='UTF-8'))   #labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_file_name = base + ".jpg"

        img_path = filename.replace(".json", ".jpg")
        assert osp.exists(img_path), img_path
        # img = cv2.imread(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        data["images"].append(
            {
                "id":image_id,  # 图像id从1开始
                "width":img.shape[1],
                "height":img.shape[0],
                "file_name":out_file_name,
                "license":0,
                "flickr_url":"",
                "coco_url" :"",
                "date_captured":0,
            }
        )

        Area = []
        Bbox = []
        Labelid = []
        for shape in label_file['shapes']:
            points = shape["points"]
            label = shape["label"]

            # if label == '多件':
            #     label = '异物'

            # if label == '露铜':
            #     label = '异物'

            # if label == '假焊':
            #     label = '虚焊'

            # if label == '少件':
            #     label = '缺件'

            # if label == '偏移':
            #     label = '偏位'

            # if label == 'N':
            #     label = '2'
            
            # if label == 'E':
            #     label = '3'

            # if label == 'G':
            #     label = 'C'

            # if label == 'L':
            #     label = '7'
            
            # if label == 'D':
            #     label = '0'

            # if label == '9':
            #     label = '6'

            # if label == 'S':
            #     label = '5'

            # if label == 'Z':
            #     label = '2'

            # if label == 'W':
            #     label = 'M'

            # if label == '假焊':
            #     label = '虚焊'

            # if label == '引脚变形':
            #     label = '翘脚'

            # if label == '假焊':
            #     label = '虚焊'

            # if label == '缺件':
            #     label = '少件'

            # if label == '破损':
            #     label = '损件'

            try:
                labelid = label_name.index(label)+startIndex
            except Exception as e:
                ng_images.append(filename)
                labelid = 0
                ng_labels.append(label)

            group_id = shape["group_id"]
            shape_type = shape["shape_type"]

            xmin = round(points[0][0],2)
            ymin = round(points[0][1],2)
            xmax = round(points[1][0],2)
            ymax = round(points[1][1],2)

            o_width = round(xmax-xmin,2)
            o_height = round(ymax - ymin,2)
            seg_area = o_width*o_height

            Labelid.append(labelid)
            Area.append(seg_area)
            Bbox.append([xmin,ymin,o_width,o_height])

        for area, bbox,labelid in zip(Area,Bbox,Labelid):
            data["annotations"].append(
                {
                    "id":len(data["annotations"]),
                    "image_id":image_id,
                    "category_id":labelid,
                    "segmentation":[],
                    "area":area,
                    "bbox":bbox,
                    "iscrowd":0,
                    "attributes": {"occluded": False}
                }
            )

    with open(out_ann_file, "w",encoding='utf-8') as f:
        json.dump(data, f)

    print('finish')
    # print(ng_images)
    print(len(ng_images))
    print(set(ng_labels))

    for image in ng_images:
        json_name = image.split('\\')[-1]
        shutil.copy2(image,os.path.join(r'C:\Users\61082\Desktop\新建文件夹',json_name))
        shutil.copy2(image.replace('.json','.jpg'),os.path.join(r'C:\Users\61082\Desktop\新建文件夹',json_name.replace('.json','.jpg')))

if __name__ == "__main__":
    main()
    print('finish')