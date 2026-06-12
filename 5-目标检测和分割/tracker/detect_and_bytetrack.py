# 先用检测，再用跟踪

import cv2
from tqdm import tqdm
import cv2
import glob
import os
from ultralytics import YOLO
import ultralytics
from ByteTrack import BYTETracker

from types import SimpleNamespace


model = YOLO("/home/std/workspace-hub/lgc/yolov11/runs/detect/person_detect/weights/last.pt")

image_path =r"/home/std/workspace-hub/lgc/yolov11/images/image3"
output_path =r"/home/std/workspace-hub/lgc/yolov11/result/track_result"
if not os.path.exists(output_path):
    os.mkdir(output_path)

tracker = BYTETracker(0.25,0,0.25,30,0.8,True)

total = 0
for name in tqdm(sorted(os.listdir(image_path),key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))):
    if name == "1774939923_218679040_352.jpg":
        print("name:", name)
    # print(name)
    frame = cv2.imread(os.path.join(image_path,name))
    results = model(os.path.join(image_path,name),conf = 0.25)
    boxes = results[0].boxes.cpu().numpy()
    tracker_objects = tracker.update(boxes,frame)

    annotated_frame = frame.copy()
    for track in tracker_objects:
        x1, y1, x2, y2, track_id, score, cls, idx = track
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)
        cls = int(cls)
        label = f"id:{track_id} cls:{cls} conf:{score:.2f}"
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_frame,
            label,
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    total += len(tracker_objects)
    print("目标框为:", len(tracker_objects))
    cv2.imwrite(os.path.join(output_path, name), annotated_frame)



print("检测的总目标框为:",total)