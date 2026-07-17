import json
import os
import cv2
from collections import defaultdict
from pathlib import Path
import shutil
import numpy as np


def decode_compressed_rle_counts(counts):
    numbers = []
    index = 0

    while index < len(counts):
        shift = 0
        value = 0

        while True:
            char_value = ord(counts[index]) - 48
            index += 1
            value |= (char_value & 0x1F) << shift
            shift += 5

            if not (char_value & 0x20):
                if char_value & 0x10:
                    value |= -1 << shift
                break

        if len(numbers) > 2:
            value += numbers[-2]
        numbers.append(value)

    return numbers


def decode_rle(segmentation):
    counts = segmentation["counts"]
    height, width = segmentation["size"]

    if isinstance(counts, str):
        counts = decode_compressed_rle_counts(counts)

    flat = np.zeros(height * width, dtype=np.uint8)
    cursor = 0
    value = 0

    for run_length in counts:
        next_cursor = cursor + int(run_length)
        if value == 1:
            flat[cursor:next_cursor] = 1
        cursor = next_cursor
        value = 1 - value

    return flat.reshape((height, width), order="F")


def rle_to_polygons(segmentation):
    mask = decode_rle(segmentation)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        contour = contour.reshape(-1, 2)
        if len(contour) < 3:
            continue
        polygons.append([[float(x), float(y)] for x, y in contour])

    return polygons


def polygon_to_points(polygon):
    return [[float(polygon[i]),float(polygon[i+1])] for i in range(0,len(polygon),2)]

def bbox_to_points(bbox):
    x, y, width, height = [float(value) for value in bbox]
    return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]

def annotation_to_shapes(annotation,category_names):
    label = category_names[annotation["category_id"]]
    segmentation = annotation.get("segmentation")
    shapes = []

    if isinstance(segmentation,list):
        polygons = [polygon_to_points(polygon) for polygon in segmentation if len(polygon) >=6]
    elif isinstance(segmentation,dict):
        polygons = rle_to_polygons(segmentation)
    else:
        polygons = []

    if not polygons and isinstance(segmentation,dict) and annotation.get("bbox"):
        polygons = [bbox_to_points(annotation["bbox"])]

    for points in polygons:
        if len(points) < 3:
            continue
        shapes.append({
            "label": label,
            "points": points,
            "group_id": None,
            "description": "",
            "shape_type": "polygon",
            "flags": {},
            "mask": None,
        })
    return shapes

def convert_coco_to_labelme(coco_path, labelme_path):
    if not os.path.exists(labelme_path):
        os.mkdir(labelme_path)

    with open(os.path.join(coco_path,"annotations","annotations.json"),"r",encoding='utf-8') as f:
        coco = json.load(f)

    category_names = {category["id"]: category["name"] for category in coco["categories"]}
    annotations_by_images = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_images[annotation["image_id"]].append(annotation)

    converted = 0 
    skipped_missing_images = 0

    for image in coco["images"]:
        if not os.path.exists(os.path.join(coco_path,'images',image["file_name"])):
            skipped_missing_images +=1
        
        labelme = {
            "version": "5.5.0",
            "flags": {},
            "shapes": [],
            "imagePath": image["file_name"],
            "imageData": None,
            "imageHeight": int(image["height"]),
            "imageWidth": int(image["width"]),
        }

        for annotation in annotations_by_images.get(image["id"],[]):
            labelme["shapes"].extend(annotation_to_shapes(annotation,category_names))

        output_path = os.path.join(labelme_path, Path(image["file_name"]).stem + ".json")
        with open(output_path,"w",encoding="utf-8") as file:
            json.dump(labelme,file,indent=4)
        shutil.copy2(os.path.join(os.path.join(coco_path,"images",image["file_name"])),os.path.join(labelme_path,image["file_name"]))      
        converted +=1

    return converted,skipped_missing_images


if __name__ == "__main__":
    labelme_path = r"/Users/jodocls/Desktop/123/StandardJNDataset.v23-jnv4.10.coco-segmentation/valid/labelme"
    coco_path = r"/Users/jodocls/Desktop/123/StandardJNDataset.v23-jnv4.10.coco-segmentation/valid"

    converted, skipped_missing_images = convert_coco_to_labelme(coco_path,labelme_path)
    print(f"已生成 {converted} 个 LabelMe JSON 文件。")
    if skipped_missing_images:
        print(f"提示：有 {skipped_missing_images} 张图片文件不存在，但仍按 COCO 记录生成了 JSON。")
