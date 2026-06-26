import json
import os
import glob
from pathlib import Path

def labelme_to_yolo(labelme_dir, output_dir, class_list=None):
    """
    将Labelme标注转换为YOLO格式
    
    Args:
        labelme_dir (str): Labelme标注文件所在的目录
        output_dir (str): YOLO格式输出目录
        class_list (list): 类别列表，如果为None则自动从所有文件中提取
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有json文件
    json_files = glob.glob(os.path.join(labelme_dir, "*.json"))
    
    # 如果没有提供类别列表，则自动收集所有类别
    if class_list is None:
        class_set = set()
        for json_file in json_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for shape in data['shapes']:
                    class_set.add(shape['label'])
        class_list = sorted(list(class_set))
    
    # 创建类别映射字典
    class_to_id = {class_name: idx for idx, class_name in enumerate(class_list)}
    
    # 保存类别文件
    # with open(os.path.join(output_dir, 'classes.txt'), 'w', encoding='utf-8') as f:
    #     for class_name in class_list:
    #         f.write(f"{class_name}\n")
    
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
        output_file = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(yolo_lines))
    
    print(f"转换完成! 共处理 {len(json_files)} 个文件")
    print(f"类别列表: {class_list}")

def main():
    # 使用示例
    labelme_dir = r"/Users/jodocls/Desktop/123/conver_image/concat_image/labelme"  # 替换为你的Labelme JSON文件目录
    output_dir = r"/Users/jodocls/Desktop/123/conver_image/concat_image/labels"         # 替换为输出目录
    
    # 可选: 指定类别列表 (如果为None则自动从文件中提取)
    # class_list = ["person", "car", "dog"]
    class_list = ['bucket', 'circle', 'handcart', 'pallet', 'two_piers', 'workbin']
    
    labelme_to_yolo(labelme_dir, output_dir, class_list)

if __name__ == "__main__":
    main()
