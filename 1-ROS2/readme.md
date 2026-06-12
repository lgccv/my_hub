# 对于ros2 python
## 1、如何调roslaunch文件

## 2、如何断点main.py文件，注意是不同的深度学习环境




# 对于ros2 c++
## 1、如何调试roslaunch文件


## 2、如何断点main.cpp文件
colcon build --packages-select custom_interfaces -DCMAKE_BUILD_TYPE=Debug

# 安装moveit

# 安装gazebo

# 安装



# 常用指令
$ ros2 pkg create --build-type ament_cmake learning_pkg_c        # C++
$ ros2 pkg create --build-type ament_python learning_pkg_python  # Python
ros2 bag record -o my_bag /camera/color/image_raw /camera/depth/image_raw
ros2 bag play my_bag