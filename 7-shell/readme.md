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


# 排查系统硬盘占用
1、看哪个分区快满： df -hT
2、从根目录找一级大户:du -x -h --max-depth=1 / 2>/dev/null | sort -h | tail -n 30
3、从/home/standard上:du -x -h --max-depth=1 /home/standard 2>/dev/null | sort -h | tail -n 35
4、找单个超大文件:find /home/standard -xdev -type f -printf '%s\t%p\n' 2>/dev/null \
  | sort -n \
  | tail -n 50 \
  | awk -F '\t' '{printf "%.1f GiB\t%s\n", $1/1024/1024/1024, $2}'
5、单独统计docker:docker system df
