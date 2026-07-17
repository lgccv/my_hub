# 对于ros2 python
## 1、如何调roslaunch文件
- 有两种方法：

- 第一种，直接写"调试 ROS2 Launch 文件",麻烦点是要写AMENT_PREFIX_PATH,AMENT_PREFIX_PATH,PYTHONPATH,LD_LIBRARY_PATH,LD_LIBRARY_PATH

- 第二种方法:
先用命令命令行启动，然后attach进去
1、先装debugpy:python3 -m pip install --user debugpy
2、然后启动roslaunch:
```shell
/usr/bin/python3 -m debugpy \        
  --listen 5678 \
  --wait-for-client \
  /opt/ros/humble/bin/ros2 launch learning_python_action simple_launch.py \
  enable_client:=false \
  client_delay:=2.0
```
3、然后再调动launch.json (Attach ROS2 Launch)

## 2、如何断点main.py文件，注意是不同的深度学习环境
- 1、首先添加:
```python
# debugpy.listen("0.0.0.0",5678)
# print("wait for debugger attach on port 5678.....")
# debugpy.wait_for_client()
```
- 2、然后在终端中运行  ros2 launch learning_python_action simple_launch.py
- 3、在launch.json中启动Attach Action Server



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

#### 看接口有哪些(/opt/ros/humble/share)
- ros2 interface list
- ros2 interface list | grep msg
- ros2 interface list | grep srv
- ros2 interface list | grep action
- ros2 interface show geometry_msgs/msg/Twist          查看话题通信的接口
- ros2 interface show turtlesim/srv/Spaw               查看服务通信的接口
- ros2 interface show turtlesim/action/RotateAbsolute  查看动作通信的接口
- ros2 interface package learning_interface            查看某功能包定义的所有通信接口

#### 设参数的方法
- ros2 param --help
- ros2 param set /head_camera/head_camera enable_color_auto_exposure false

#### action的指令
- ros2 action list
- ros2 action info
- ros2 action send_goal action名称  action数据类型 "{theta: 3.14}" --feedback(把反馈打开)


### 创建接口包(只能用C++的包，不能用python)
```python
ros2 pkg create learning_interfaces --build-type ament_cmake --dependencies rosidl_default_generators std_msgs
```

### 不是一定要放到src目录下
```python
cd ~/workspace/src/cplusplus
ros2 pkg create learning_interfaces --build-type ament_cmake --dependencies rosidl_default_generators std_msgs
```

### 插件
- 安装ros2插件，写接口时会有提示

### 注意点
在CMakeLists.txt中加入
```shell
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MoveCircle.action"
)
```
在package.xml中加入
```shell
<member_of_group>rosidl_interface_packages</member_of_group>
```

### from查找的位置在哪里
/home/standard/code/ros2/install/learning_interfaces/local/lib/python3.10/dist-packages/learning_interfaces/action/_move_circle.py


## action的流程



# colcon build的相关问题
- ros2 pkg list 系统能识别的功能包
- ros2 pkg prefix rclpy  想看某个包来自哪里
- colcon list   当前工作目录的能识别的包
- colcon build --packages-select learning_python_action  编译单个包



# 问题:
1、ros2的功能包需要放到哪个路径下？一定要在src目录下吗？
不是,可以cd到指定文件夹再 ros2 pkg create

2、colcon build 可以指定编译后的路径吗
可以，colcon build --packages-select learning_interfaces --install-base result

3、source install/setup.zsh到底发生了什么？
```python
# 用来查找已经安装好的ros包
export AMENT_PREFIX_PATH=/home/standard/code/ros2/result/learning_interfaces:$AMENT_PREFIX_PATH
# 用来查找.cmake文件的路径
export CMAKE_PREFIX_PATH=/home/standard/code/ros2/result/learning_interfaces:$CMAKE_PREFIX_PATH
# 用来查找python的包，也就是import
export PYTHONPATH=/home/standard/code/ros2/result/learning_interfaces/local/lib/python3.10/dist-packages:$PYTHONPATH
# 用来查找动态链接库，也就是.so文件
export LD_LIBRARY_PATH=/home/standard/code/ros2/result/learning_interfaces/lib:$LD_LIBRARY_PATH
```
可以

cd result

source setup.bash  会自动添加上面的变量

4、ros2的python装到哪里？如何证明
```python
- /usr/bin/python
- which python
```

5、pip install安装的包都到了哪里？如何证明

6、--symlink-install是什么意思：
```python
launch.py是软链接，修改以后，不用重新colcon build
```
