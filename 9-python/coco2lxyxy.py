import os
import json
import numpy as np
import pandas as pd



def convert_lxyxy(box):
    # box = [左上x，左上y，w, h]
    xmin = box[0]
    ymin = box[1]
    xmax = box[0] + box[2]
    ymax = box[1] + box[3]
    return [xmin, ymin, xmax, ymax]

def write_labels(x):
    global out_dir
    file_name,iw,ih,category_id,bboxes = x
    if len(bboxes):
        with open(out_dir+'\\'+file_name[:-3]+'txt','w',encoding='utf8') as f:
            for cate,bbox in zip(category_id,bboxes):
                x,y,w,h = convert_lxyxy(bbox)
                f.write(' '.join([str(cate),str(x),str(y),str(w),str(h)])+'\n')  

# 平台json文件的路径
tarjson = r'D:\2-Project\4-jixing\dataset\test\annotations\instances_annotations.json'
# 图片的路径,在图片目录下生成txt文件
out_dir = r"D:\2-Project\4-jixing\dataset\test\images"

with open(tarjson,'r') as f:
    ins = json.loads(f.read())

# 将图像信息保存下来
img_info = pd.DataFrame(ins['images'])
# 将标注信息保存下来
anno_info =  pd.DataFrame(ins['annotations'])
print(anno_info)
# 第几号图像有几个框,框的坐标是什么
new_anno = anno_info[['image_id','category_id','bbox']].groupby('image_id').agg(list).reset_index().rename(columns={'image_id':'id'})

infos = pd.merge(img_info,new_anno,on='id',how='left')

infos[~infos['category_id'].isnull()][['file_name','width','height','category_id','bbox']].apply(write_labels,axis=1)