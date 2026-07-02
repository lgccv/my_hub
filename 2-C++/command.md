## 查看pcl的版本
- grep "#define PCL_VERSION" /usr/include/pcl-*/pcl/pcl_config.h

## 查找PCL的头文件和库文件在哪里
- find /usr /usr/local /opt -path '*/pcl/point_cloud.h' 2>/dev/null
- find /usr /usr/local /opt /lib -name 'libpcl*.so*' 2>/dev/null