# 查询可以装什么版本的库
apt list -a libpcl-dev

# 查库是什么版本
dpkg -l libpcl-dev

# 查询库在什么地方
ldconfig -p 
ldconfig -p | grep pcl

# 查看一个文件是什么类型
file $(which cmake)

# 如何查电脑装了哪些库
apt list --installed
apt list --installed | grep pcl

# opt目录和usr目录有什么区别？
/usr 是系统标准软件目录,包含bin/lib/include/share

/opt 是第三方厂商独立软件目录