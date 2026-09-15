# 建立镜像
cd /home/standard/code/my_hub
./docker/build_docker.sh vscode
source /root/.zshrc


# 解析sbag包
python /workspace/9-python/sbag_to_mp4.py \
  /workspace/9-python/2026-09-13-21-35-13.sbag \
  -o /workspace/9-python/2026-09-13-21-35-13-opencv \
  --view-3d

python -m http.server 8765 \
  --directory /workspace/9-python/2026-09-13-21-35-13-opencv/