## 查看pcl的版本
- grep "#define PCL_VERSION" /usr/include/pcl-*/pcl/pcl_config.h

## 查找PCL的头文件和库文件在哪里
- find /usr /usr/local /opt -path '*/pcl/point_cloud.h' 2>/dev/null
- find /usr /usr/local /opt /lib -name 'libpcl*.so*' 2>/dev/null

## 如何查电脑是什么系统
- uname -a
- uname -m
- cat /etc/os-release

## 如何查OpenCV的版本，头文件，库文件在哪里
- apt list -a libopencv-dev  查询能装哪些版本的Opencv
- dpkg -l 4.5.4+dfsg-9ubuntu4  如何安装OpenCV
- sudo apt remove libopencv-dev python3-opencv opencv-data  卸载OpenCV
   sudo apt autoremove

- sudo apt install libopencv-dev
- pkg-config --modversion opencv4
- sudo find / -name "opencv.hpp" 2>/dev/null
- sudo find / -name "libopencv_core.so*" 2>/dev/null
- sudo find / -name "OpenCVConfig.cmake" 2>/dev/null

## 如何看包是否是apt安装的
- dpkg -l | grep opencv  如果看前面是ii.说明是apt/dpkg系统安装的包

## 如果想要安装指定版本的OpenCV怎么办
- 只能用git clone才能,然后编译 


## findpackage的默认路径是什么
- list(APPEND CMAKE_PREFIX_PATH "/opt/opencv-4.8.0")  自己在这个路径一下查找OpenCVConfig.cmake
- set(OpenCV_DIR "/opt/opencv-4.8.0/lib/cmake/opencv4")  或者直接指定查找路径


## 系统默认查找头文件的目录
```python
/usr/include/c++/11                          ← C++ 标准库
/usr/include/x86_64-linux-gnu/c++/11         ← 架构特定的 C++ 标准库
/usr/include/c++/11/backward                 ← 旧版兼容
/usr/lib/gcc/x86_64-linux-gnu/11/include     ← GCC 内置头文件
/usr/local/include                           ← 本地/第三方库
/usr/include/x86_64-linux-gnu                ← 架构特定系统头文件
/usr/include                                 ← 通用系统头文件 ← glog 在这里
```
## 系统默认查找库文件的目录
```python
/usr/lib/gcc/x86_64-linux-gnu/11/
/usr/lib/x86_64-linux-gnu/               
/usr/lib/
/lib/x86_64-linux-gnu/
/lib/
```