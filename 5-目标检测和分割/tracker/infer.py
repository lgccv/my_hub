
# Python
import cv2

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/home/std/workspace-hub/lgc/yolov11/runs/detect/TrackCar/weights/best.pt")

# Open the video file
video_path = "/home/std/workspace-hub/lgc/yolov11/tools/tracker/track2.mp4"
output_path = "/home/std/workspace-hub/lgc/yolov11/tools/tracker/byte_track2.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
# 获取视频属性
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# 初始化VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True,tracker="bytetrack.yaml")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Write the annotated frame to output video
        out.write(annotated_frame)

        # Display the annotated frame
        # cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, video writer and close the display window
cap.release()
out.release()
# cv2.destroyAllWindows()
print(f"Output video saved to: {output_path}")


