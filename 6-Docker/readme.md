# 安装docker
- https://docs.docker.com/engine/install/ubuntu/
```python
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl status docker
sudo systemctl start docker
sudo docker run hello-world  // 检验安装效果
```

# 建立镜像的方式有哪些
## 命令行
docker build -t ubuntu22.04_base . -f Dockerfile.base
## docker-compose生成
docker compose up --build

# 建立容器的方法有哪些
## 命令行
```bash
docker run --name my_container -it -v /home/standard/code/docker:/workspace ubuntu22.04_base:latest /bin/zsh
docker run --rm -it ubuntu22.04_base:latest
# 如果只是想用一个镜像的环境,不想进容器，只编译一次的容器
docker run --rm -v /home/standard/code/docker:/workspace -w /workspace ubuntu22.04_base:latest  g++ /workspace/src/main.cpp -o /workspace/main
```
## dockercompose
docker compose -f docker-compose.yaml up -d my_container



## 3、进入一个容器
- 命令行
docker exec -it my_container /bin/zsh
- .devcontainer
- vscode插件


# 主机和容器共享GPU
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.17.5/install-guide.html?utm_source=openai
```python
# 添加包仓库和GPG密钥
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
# 对于较旧的系统，你可能需要安装 nvidia-docker2
# sudo apt-get install -y nvidia-docker2
sudo nvidia-ctk runtime configure --runtime=docker
# 重启 Docker 守护进程
sudo systemctl restart docker

# 验证是否成功
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

```


# 从image反推dockerfile
docker history --no-trunc ros2   # 不截断输出ros2镜像

## 如何debug一个dockerfile
参考文档:

https://www.docker.com/blog/debug-docker-builds-with-visual-studio-code/

1、安装 Docker DX 插件

2、运行 docker buildx version 确认buildx版本至少是0.29.x

launch.json
```shell
{
    "version": "0.2.0",
    "configurations": [
        {
            "type": "dockerfile",
            "request": "launch",
            "name": "Docker: Build",
            "dockerfile": "Dockerfile",
            "contextPath": "${workspaceFolder}"
        }
    ]
}
```


## 为什么需要两个docker，一个用于x86-cross的交叉编译，一个用于arm64



## 如何通过dockerfile生成一个镜像？
```python
docker  build  --network host --platform linux/arm64 \
  -t nexus3.sr/sros/darwin/dev-jetson:cuda12.8-jazzy-base \
  --push \
  -f docker/dev/arm64/Dockerfile.jetson \
  .
```
- docker buildx build：用 BuildKit/buildx 构建镜像，支持跨平台和直接推送。
- --network host：构建过程里的 apt/wget 等命令使用宿主机网络。
- --platform linux/arm64：构建 arm64 镜像，给 Orin 用。
- -t nexus3.sr/sros/darwin/dev-jetson:cuda12.8-jazzy-base：给产物打基础镜像 tag。
- --push：构建完成后推到 Nexus；这样第二步 Dockerfile 才能从 registry 拉到这个基础镜像。
- -f docker/dev/arm64/Dockerfile.jetson：指定用 Jetson 基础镜像 Dockerfile。
- .：构建上下文是当前仓库根目录。

## 镜像拉不下来怎么办？
- 加入中转站名称
```python
docker pull docker.1ms.run/dustynv/pytorch:2.7-r36.4.0-cu128-24.04
```

## 如何看镜像是什么内核架构
```python
uname -a              宿主机内核 + 内核架构
uname -r              宿主机内核版本
cat /etc/os-release   容器镜像里的发行版
dpkg --print-architecture  容器用户态架构
```

## 如何在x86上构建arm的容器

## 如何在x86上运行arm的容器

## 什么是交叉编译
- 在 x86 宿主机上运行 x86/amd64 架构的编译容器，容器里安装 ARM64 交叉编译器，最终生成 ARM64 文件。