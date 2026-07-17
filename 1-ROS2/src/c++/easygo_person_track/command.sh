## 编译
colcon build --packages-select orbbec_camera_msgs orbbec_camera easygo_person_track
source /opt/ros/jazzy/setup.zsh && source ~/easygo_ws/install/setup.zsh
ros2 launch easygo_person_track easygo_person_track_with_camera.launch.py


## 发送开始绑定指令
ros2 action send_goal /follow/perception/start_binding easygo_follow_msgs/action/StartBinding \
"{config: {bind_distance_min_m: 0.5, bind_distance_max_m: 5.0, camera_mount_height_mm: 124, camera_mount_pitch_deg: -18.0}}" \
--feedback

## 发送停止跟踪指令
ros2 service call /follow/perception/stop_tracking easygo_follow_msgs/srv/StopTracking "{}"

## 录制话题
ros2 bag record -o person_track_debug_bag \
  /camera/color/image_raw \
  /camera/depth/image_raw \
  /follow/perception/detection_image \
  /follow/perception/tracked_target \
  /follow/perception/events

ros2 bag record -o person_track_debug_bag /follow/perception/detection_image

## 播放录制包
ros2 bag play person_track_debug_bag