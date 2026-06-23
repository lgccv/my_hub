# 对于ros2 python
## 1、如何调roslaunch文件

## 2、如何断点main.py文件，注意是不同的深度学习环境




# 对于ros2 c++
## 1、如何调试roslaunch文件


## 2、如何断点main.cpp文件
colcon build --packages-select custom_interfaces -DCMAKE_BUILD_TYPE=Debug

# 安装moveit

# 安装gazebo

# 学习计划：
- nav2 : 导航路径规划
- moveit2: 机械臂抓取 (b站,古月居)
- 机器人学基础: b站

# 常用指令
#### 创建功能包
- ros2 pkg create --build-type ament_cmake learning_pkg_c        # C++
- ros2 pkg create --build-type ament_python learning_pkg_python  # Python


#### 录制和播放数据包
- ros2 bag record -o my_bag /camera/color/image_raw /camera/depth/image_raw
- ros2 bag play my_bag

#### 查询ros有多少个功能包
- ros2 pkg list

#### 查找功能包的路径
- echo $AMENT_PREFIX_PATH

#### 看接口有哪些
- ros2 interface list
- ros2 interface list | grep msg
- ros2 interface list | grep srv
- ros2 interface list | grep action
