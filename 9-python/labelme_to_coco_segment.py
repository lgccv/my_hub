import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def polygon_area(points):
    """用多边形鞋带公式计算 COCO annotation 的 area。"""
    if len(points) < 3:
        return 0.0

    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def polygon_bbox(points):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    return [
        round(min_x, 3),
        round(min_y, 3),
        round(max_x - min_x, 3),
        round(max_y - min_y, 3),
    ]


def flatten_polygon(points):
    polygon = []
    for x, y in points:
        polygon.extend([round(float(x), 3), round(float(y), 3)])
    return polygon


def make_categories(label_names):
    if not label_names:
        return []

    supercategory = "none"
    return [
        {
            "id": category_id,
            "name": label,
            "supercategory": supercategory if category_id == 0 else label_names[0],
        }
        for category_id, label in enumerate(label_names)
    ]


def convert_labelme_to_coco(labelme_path, coco_path, label_names):
    labelme_dir = Path(labelme_path)
    coco_dir = Path(coco_path)
    images_dir = coco_dir / "images"
    annotations_dir = coco_dir / "annotations"
    annotations_path = annotations_dir / "annotations.json"

    if not labelme_dir.exists():
        raise FileNotFoundError(f"labelme 目录不存在: {labelme_dir}")

    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    label_to_id = {label: index for index, label in enumerate(label_names)}
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    coco = {
        "info": {
            "year": datetime.now().year,
            "version": "1",
            "description": "",
            "contributor": "",
            "url": "",
            "date_created": now,
        },
        "licenses": [{"id": 1, "url": "", "name": "Unknown"}],
        "categories": make_categories(label_names),
        "images": [],
        "annotations": [],
    }

    annotation_id = 1
    json_files = sorted(labelme_dir.glob("*.json"))

    for image_id, json_path in enumerate(json_files):
        with json_path.open("r", encoding="utf-8") as file:
            labelme = json.load(file)

        image_name = labelme["imagePath"]
        image_path = labelme_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"{json_path.name} 对应图片不存在: {image_path}")

        target_image_path = images_dir / image_name
        if image_path.resolve() != target_image_path.resolve():
            shutil.copy2(image_path, target_image_path)

        coco["images"].append(
            {
                "id": image_id,
                "license": 1,
                "file_name": image_name,
                "height": int(labelme["imageHeight"]),
                "width": int(labelme["imageWidth"]),
                "date_captured": now,
            }
        )

        for shape in labelme.get("shapes", []):
            label = shape.get("label")
            if label not in label_to_id:
                raise ValueError(f"{json_path.name} 里出现未配置类别: {label}")

            points = [(float(x), float(y)) for x, y in shape.get("points", [])]
            if len(points) < 3:
                continue

            area = polygon_area(points)
            if area <= 0:
                continue

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label_to_id[label],
                    "bbox": polygon_bbox(points),
                    "area": round(area, 3),
                    "segmentation": [flatten_polygon(points)],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

    with annotations_path.open("w", encoding="utf-8") as file:
        json.dump(coco, file, ensure_ascii=False)

    return annotations_path


if __name__ == "__main__":
    labelme_path = r"/Users/jodocls/Desktop/123/autolabel/real/labelme"
    coco_path = "/Users/jodocls/Desktop/123/autolabel/real"
    label_names = [
        "wheel-8cWi",
        "cardboard_smallbox",
        "gray_box",
        "guiding",
        "material_platform",
        "tag_code_m",
        "tray",
    ]

    result_path = convert_labelme_to_coco(labelme_path, coco_path, label_names)
    print(f"COCO annotations saved to: {result_path}")
