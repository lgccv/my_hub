# 建立镜像
cd /home/standard/code/my_hub
./docker/build_docker.sh vscode
source /root/.zshrc


# 解析sbag包
conda init
conda activate base
python /workspace/9-python/sbag_to_mp4.py \
  /workspace/9-python/2026-09-20-16-04-44.sbag \
  -o /workspace/9-python/2026-09-20-16-04-44-opencv \
  --view-3d

python -m http.server 8765 \
  --directory /workspace/9-python/2026-09-20-16-04-44-opencv/






# 软件目标：
1、通过grpc,T1和T2控制两个相机单针采图
  是否要做指令缓存区，是否要做图像缓存图
2、算法封装成一个类,两个相机分别走2个对象,通过grpc发送结果
3、通过全局存图函数最后存图存到同一个文件夹中,算法的中间过程日志存到一个日志文件中
4、保存的图片定时删除，只保存最后半小时的图片
5、两个相机分别发送结果要怎么做，两个相机组合结果在一起又该怎么做?