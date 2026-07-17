# 将分割数据集转为yolo的格式

import os
import json

labelme_path = r"/Users/jodocls/Desktop/123/autolabel/real/labelme"
yolo_path = r"/Users/jodocls/Desktop/123/autolabel/real/yolo"

label_names = [
    "wheel-8cWi",
    "cardboard_smallbox",
    "gray_box",
    "guiding",
    "material_platform",
    "tag_code_m",
    "tray",
]

json_name = [name for name in os.listdir(labelme_path) if name.endswith(".json")]
print("json_name:",len(json_name))

for name in json_name:
    with open(os.path.join(labelme_path,name),mode="r") as f:
        data = json.load(f)

    width = data["imageWidth"]
    height = data["imageHeight"]

    yolo_lines = []

    shapes = data["shapes"]
    for shape in shapes:
        class_id = label_names.index(shape["label"])
        values = [str(class_id)]
        for point in shape["points"]:
            x = point[0] / width
            y = point[1] / height

            x = max(0.0,min(1.0,x))
            y = max(0.0,min(1.0,y))
            values.append(f"{x:.6f}")
            values.append(f"{y:.6f}")
        yolo_lines.append(" ".join(values))

    txt_path = yolo_path  +'/'+ name.replace(".json",".txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(yolo_lines))


