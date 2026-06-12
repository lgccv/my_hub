
# Python
import cv2
from tqdm import tqdm
import cv2
import glob
import os

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/home/std/workspace-hub/lgc/yolov11/runs/detect/person_detect/weights/last.pt")

# Open the video file
image_path =r"/home/std/workspace-hub/lgc/yolov11/images/image3"
output_path =r"/home/std/workspace-hub/lgc/yolov11/result/track_result"
if not os.path.exists(output_path):
    os.mkdir(output_path)


total = 0
for name in tqdm(sorted(os.listdir(image_path),key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))):
    if name == "1774939923_218679040_352.jpg":
        print("name:", name)
    print("name:",name)
    frame = cv2.imread(os.path.join(image_path,name))
    # Run YOLO11 tracking on the frame, persisting tracks between frames
    results = model.track(frame, persist=True,tracker="botsort.yaml",conf = 0.25)
    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    result = results[0]
    boxes = result.boxes

    if boxes.id is None:
        annotated_frame = frame.copy()
        track_count = 0
    else:
        annotated_frame = result.plot()
        track_count = len(boxes.id)

    total = total + track_count

    print("目标框为:",len(results[0].boxes.conf))
    cv2.imwrite(os.path.join(output_path,name),annotated_frame)

print("检测的总目标框为:",total)




result = results[0]
boxes = result.boxes

if boxes.id is None:
    annotated_frame = frame.copy()
    track_count = 0
else:
    annotated_frame = result.plot()
    track_count = len(boxes.id)

