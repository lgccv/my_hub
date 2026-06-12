import os
import shutil
from tqdm import tqdm
import json
import glob
from pathlib import Path


def labelme_to_yolo(path,class_list):
    """
    将Labelme标注转换为YOLO格式
    
    Args:
        labelme_dir (str): Labelme标注文件所在的目录
        output_dir (str): YOLO格式输出目录
        class_list (list): 类别列表，如果为None则自动从所有文件中提取
    """
    if os.path.exists(os.path.join(path,'labels')):
        shutil.rmtree(os.path.join(path,'labels'))
        print(f"文件夹{os.path.join(path,'labels')}及其内容已删除")
    
    # 创建输出目录
    os.makedirs(os.path.join(path,'labels'), exist_ok=True)
    
    # 获取所有json文件
    json_files = glob.glob(os.path.join(path,'labelme', "*.json"))
    

    # 创建类别映射字典
    class_to_id = {class_name: idx for idx, class_name in enumerate(class_list)}
    
    # 处理每个JSON文件
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取图像尺寸
        image_width = data['imageWidth']
        image_height = data['imageHeight']
        
        # 准备YOLO格式内容
        yolo_lines = []
        
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']
            
            # 只处理多边形和矩形
            if shape['shape_type'] not in ['polygon', 'rectangle']:
                print(f"跳过不支持的类型: {shape['shape_type']} in {json_file}")
                continue
            
            # 获取类别ID
            class_id = class_to_id.get(label)
            if class_id is None:
                print(f"警告: 跳过未在类别列表中找到的标签 '{label}' in {json_file}")
                continue
            
            # 转换坐标到YOLO格式
            if shape['shape_type'] == 'rectangle':
                # 矩形: 转换为[x_center, y_center, width, height]
                x_min = min(points[0][0], points[1][0])
                y_min = min(points[0][1], points[1][1])
                x_max = max(points[0][0], points[1][0])
                y_max = max(points[0][1], points[1][1])
                
                x_center = (x_min + x_max) / 2 / image_width
                y_center = (y_min + y_max) / 2 / image_height
                width = (x_max - x_min) / image_width
                height = (y_max - y_min) / image_height
                
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            elif shape['shape_type'] == 'polygon':
                # 多边形: 转换为YOLO多边形格式 [class_id, x1, y1, x2, y2, ...]
                normalized_points = []
                for x, y in points:
                    normalized_x = x / image_width
                    normalized_y = y / image_height
                    normalized_points.extend([normalized_x, normalized_y])
                
                points_str = " ".join([f"{p:.6f}" for p in normalized_points])
                yolo_lines.append(f"{class_id} {points_str}")
        
        # 写入YOLO格式文件
        base_name = Path(json_file).stem
        output_file = os.path.join(path,'labels', f"{base_name}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_lines))
    

# 检查yolo是否合格
def is_illega_yolo(path):

    images = [name for name in os.listdir(os.path.join(path,"images")) if name.endswith('.jpg')]
    yolo = [name.replace('.txt','.jpg') for name in os.listdir(os.path.join(path,"labels")) if name.endswith("txt")]

    if sorted(images) == sorted(yolo):
        result = "合格"
    else:
        result = "不合格"
    return result


if __name__ == "__main__":

    labelmeName = ['person','amr']

    dataset_path = [r"/home/std/workspace-hub/lgc/dataset/persondetectionDataset_s/dataset_0331_9642",
                    r"/home/std/workspace-hub/lgc/dataset/persondetectionDataset_s/dataset_0401",  # 下面应该是images和labelme
                    r"/home/std/workspace-hub/lgc/dataset/persondetectionDataset_s/dataset_040102",
                    r"/home/std/workspace-hub/lgc/dataset/persondetectionDataset_s/dataset_0402"
                    ]
    
    concat_dataset_path = r"/home/std/workspace-hub/lgc/dataset/persondetectionDataset_s/concat_dataset"

    # labelme 生成yolo
    for sub_path in dataset_path:
        if not os.path.exists(os.path.join(sub_path,"labels")):
            labelme_to_yolo(sub_path,labelmeName)
            if is_illega_yolo(sub_path) == "合格":
                print(f"dataset_path:{sub_path}:生成Yolo文件夹")
        else:
            # 检查yolo是否合格
            result = is_illega_yolo(sub_path)
            if result == '不合格':
                labelme_to_yolo(sub_path,labelmeName)
                print(f"dataset_path:{sub_path}:生成yolo文件夹")
            else:
                print(f"dataset_path:{sub_path}:yolo文件夹未变化")

    # 合并yolo
    if not os.path.exists(os.path.join(concat_dataset_path,'train','images')):
        os.makedirs(os.path.join(concat_dataset_path,'train','images'),exist_ok=True)
    if not os.path.exists(os.path.join(concat_dataset_path,'train','labels')):
        os.makedirs(os.path.join(concat_dataset_path,'train','labels'),exist_ok=True)

    print("原始训练集数量：",len(os.listdir(os.path.join(concat_dataset_path,"train","images"))))

    images = [name for name in os.listdir(os.path.join(concat_dataset_path,'train','images'))]
    txts = [name for name in os.listdir(os.path.join(concat_dataset_path,'train','labels'))]

    for sub_path in tqdm(dataset_path):
        sub_images = [name for name in os.listdir(os.path.join(sub_path,'images')) if name.endswith('.jpg')]
        sub_txt = [name for name in os.listdir(os.path.join(sub_path,'labels')) if name.endswith('.txt')]
        for index,simage in tqdm(enumerate(sub_images)):
            if simage not in images:
                shutil.copy2(os.path.join(sub_path,'images',simage),os.path.join(concat_dataset_path,'train','images',simage))
                shutil.copy2(os.path.join(sub_path,'labels',simage.replace('.jpg','.txt')),os.path.join(concat_dataset_path,'train','labels',simage.replace('.jpg','.txt')))

    # 检查labelme是否合格
    result = is_illega_yolo(concat_dataset_path+'/train')
    if result == "合格":
        print("合并数据集结果合格OK")
        print("当前训练集数量：",len(os.listdir(os.path.join(concat_dataset_path,"train","images"))))
    else:
        print("合并数据集结果不不不合格")
        exit()


