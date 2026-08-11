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
# debugpy.listen(("0.0.0.0",5678))
# print("wait for debugger attach on port 5678.....")
# debugpy.wait_for_client()
```
- 2、然后在终端中运行  ros2 launch learning_python_action simple_launch.py
- 3、在launch.json中启动Attach Action Server
- 手动关掉5678端口: kill -9 $(lsof -t -i :5678)


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

#### topic的相关指令
- ros2 topic echo /chatter
- ros2 topic pub /chatter std_msgs/msg/String "{data:‘123’}"

#### node的相关指令
- ros2 node info /action/sim
- ros2 node list

#### lifecycle的相关指令
- ros2 lifecycle get lifecycle_pb
- ros2 lifecycle list lifecycle_pb   可以执行哪几种状态
- ros2 lifecycle set lifecycle_pb configure  设置节点的状态

#### service的相关指令
- ros2 service list
- ros2 service type /add_two_ints
- ros2 service call /add_two_ints learning_interface/srv/AddTwoInts "{a: 1, b: 2}"



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
## 1、ros2的功能包需要放到哪个路径下？一定要在src目录下吗？
不是,可以cd到指定文件夹再 ros2 pkg create

## 2、colcon build 可以指定编译后的路径吗
可以，colcon build --packages-select learning_interfaces --install-base result

## 3、source install/setup.zsh到底发生了什么？
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

## 4、ros2的python装到哪里？如何证明
```python
- /usr/bin/python
- which python
```

## 5、pip install安装的包都到了哪里？如何证明

## 6、--symlink-install是什么意思：
```python
launch.py是软链接，修改以后，不用重新colcon build
```


## 7、sequence size exceeds remaining buffer  这个问题如何解决
方法一:
```python
pkill -9 -f "ros2 launch pkg_python_rewrite"
pkill -9 -f "install/pkg_python_rewrite/lib/pkg_python_rewrite"
ros2 daemon stop
ros2 daemon start
```

方法二:
```python
export ROS_DOMAIN_ID=77
ros2 launch pkg_python_rewrite python_rewrite.launch.py
如果你开多个终端一次测试，他们需要都要设置同一个值 export ROS_DOMAIN_ID=77
```

## LifecycleNode是什么？
负责节点生命周期管理，了解节点的生命状态
- def on_configure(self, state: LifecycleState)  准备资源
- def on_activate(self, state: LifecycleState)   开始对外服务
- def on_deactivate(self, state: LifecycleState) 停止对外服务
- def on_cleanup(self, state: LifecycleState)    释放配置资源
- def on_shutdown(self, state: LifecycleState)   退出前清理全部资源

## 检查URDF语法错误
```
sudo apt-get install liburdfdom-tools
check_urdf src/example/robot2.urdf 
```

## 查看URDF的结构
```python
urdf_to_graphiz src/example/robot2.urdf
```

## 安装和启动Gazebo
```python
sudo apt install ros-humble-gazebo-*
ros2 launch gazebo_ros gazebo.launch.py
```

## 在gazebo下添加模型
.gazebo/models
在Insert下可以看到模型

## Link在gazebo中的要求
1、为link添加惯性参数和碰撞属性(collision)
2、为link添加gazebo标签
3、为joint添加传动装置
4、添加gazebo控制器插件
ros2 launch learning_gazebo load_urdf_into_gazebo.launch.py
ros2 run teleop_twist_keyboard teleop_twist_keyboard