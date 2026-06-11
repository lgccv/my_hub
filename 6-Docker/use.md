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
