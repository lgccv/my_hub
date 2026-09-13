# 踩坑记录
- 由于是GPU5060显卡，所以pytorch的cuda要是12.8，跟装不装cuda没啥关系
- 由于cuda的版本要是12.8，所以pytorch的版本要跟上
- 如果要装高版本的pytorch,python的版本要高；
- 当前的选择是python(3.10)+torch(2.8_cu12.8)


## tensorRT的安装
### 下载地址(tar)：https://developer.nvidia.com/tensorrt/download
### 安装说明:https://docs.nvidia.com/deeplearning/tensorrt/latest/installing-tensorrt/install-tar.html
```python
tar -xvf TensorRT-Enterprise-${version}-Linux-${arch}-${cuda}-Release-external.tar.zst
echo 'export LD_LIBRARY_PATH=<TensorRT-${version}/lib>:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
cd TensorRT-${version}/python
python3 -m pip install tensorrt-*-cp3x-none-linux_x86_64.whl
```

## CUDA的安装:
- 下载地址：https://developer.nvidia.com/cuda-toolkit-archive
- wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
- sudo sh cuda_12.1.0_530.30.02_linux.run
- export PATH="/usr/local/cuda-12.1/bin:$PATH"
- source ~/.bashrc
- nvcc --version


## pytorch、cuda的版本
- nvidia-smi  显卡驱动信息，当前支持的最高CUDA版本
- nvcc -V    当前使用CUDA的版本
- ll /usr/local/  # 查看安装了几个CUDA，当前使用哪个版本的CUDA
- import torch;print(torch.version.cuda)  当前Pytorch的运行CUDA API版本