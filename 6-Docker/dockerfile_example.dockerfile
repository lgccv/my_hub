FROM ubuntu:22.04

# 安装软件时不要进入交互界面
ENV DEBIAN_FRONTEND=noninteractive \
    # 设置时区是UTC
    TZ=Etc/UTC \
    # 定义conda的安装目录变量                     
    CONDA_DIR=/opt/conda \
    # 将conda的可执行目录加到系统 PATH 最前面
    PATH=/opt/conda/bin:$PATH


# 基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    # --- 基础工具 ---
    git wget vim zsh curl sudo tmux tree cloc cmake g++ gcc\
    # --- 网络诊断 ---
    iputils-ping net-tools iproute2 tcpdump nmap iperf3 \
    # --- 硬件/总线调试 ---
    can-utils usbutils minicom i2c-tools mesa-utils kmod \
    # --- 系统/性能监控 ---
    htop iotop strace gdb lsof \
    # 删除 apt 的索引缓存。
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    # 按不交互的方式安装到指定文件里面去
    bash /tmp/miniconda.sh -b -p ${CONDA_DIR} && \
    rm -f /tmp/miniconda.sh && \
    conda config --system --set auto_update_conda false && \
    conda config --system --remove-key channels || true && \
    conda config --system --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    conda config --system --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r && \
    conda config --system --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2 && \
    conda config --system --set show_channel_urls yes && \
    # 清理conda索引缓存、tar包、未使用缓存等
    conda clean -afy


# 3. 配置 Oh-My-Zsh (SKEL 方式)
ENV ZSH=/opt/oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended && \
    git clone https://github.com/zsh-users/zsh-autosuggestions $ZSH/custom/plugins/zsh-autosuggestions && \
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git $ZSH/custom/plugins/zsh-syntax-highlighting && \
    git clone https://github.com/spaceship-prompt/spaceship-prompt.git $ZSH/custom/themes/spaceship-prompt --depth=1 && \
    ln -s $ZSH/custom/themes/spaceship-prompt/spaceship.zsh-theme $ZSH/custom/themes/spaceship.zsh-theme && \
    cp $ZSH/templates/zshrc.zsh-template /root/.zshrc && \
    sed -i "s|^export ZSH=.*|export ZSH=$ZSH|g" /root/.zshrc && \
    sed -i 's/^ZSH_THEME=.*/ZSH_THEME="spaceship"/g' /root/.zshrc && \
    sed -i 's/^plugins=(.*/plugins=(git zsh-autosuggestions zsh-syntax-highlighting)/g' /root/.zshrc && \
    echo 'alias lg=lazygit' >> /root/.zshrc && \
    cp /root/.zshrc /etc/skel/.zshrc && \
    chmod -R 755 $ZSH

# 4。 安装ros2
# ros2
LABEL description="perception_platform with ROS2 Humble environment"
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# ======================== 添加 ROS2 apt 源 ========================
# 使用宿主机已有密钥文件，避免访问被封锁的 raw.githubusercontent.com
COPY ros-archive-keyring.gpg /usr/share/keyrings/ros-archive-keyring.gpg

RUN apt-get update && apt-get install -y --no-install-recommends \
     lsb-release \
     && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
         https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu $(lsb_release -cs) main" \
         > /etc/apt/sources.list.d/ros2.list \
     && rm -rf /var/lib/apt/lists/*

# ======================== 安装 ROS2 核心包 ========================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-${ROS_DISTRO}-ros-base \
    ros-${ROS_DISTRO}-ament-cmake \
    ros-${ROS_DISTRO}-rclcpp \
    ros-${ROS_DISTRO}-std-msgs \
    ros-${ROS_DISTRO}-sensor-msgs \
    ros-${ROS_DISTRO}-geometry-msgs \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-pcl-conversions \
    ros-${ROS_DISTRO}-realsense2-camera-msgs \
    python3-colcon-common-extensions \
    python3-ament-package \
    && rm -rf /var/lib/apt/lists/*

# ======================== ROS2 环境配置 ========================
# 每次启动 bash 自动 source ROS2 环境
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.zshrc

ENV AMENT_PREFIX_PATH="/opt/ros/${ROS_DISTRO}"
ENV COLCON_PREFIX_PATH="/opt/ros/${ROS_DISTRO}"
ENV PATH="/opt/ros/${ROS_DISTRO}/bin:${PATH}"
ENV ROS_VERSION=2
ENV ROS_PYTHON_VERSION=3

CMD ["/bin/zsh"]