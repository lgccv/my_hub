FROM ubuntu:24.04

ARG ROS_DISTRO=jazzy
ARG ANACONDA_VERSION=2026.07-1
ARG ANACONDA_SHA256_AMD64=9f49042048e9a8220c1cddf9192054290724052d3b2a86f938f9f5edd911db8f
ARG ZSH_AUTOSUGGESTIONS_ARCHIVE=zsh-autosuggestions-0.7.1.tar.gz
ARG ZSH_SYNTAX_HIGHLIGHTING_ARCHIVE=zsh-syntax-highlighting-0.8.0.tar.gz

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=${ROS_DISTRO} \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    CONDA_DIR=/opt/anaconda3 \
    SHELL=/usr/bin/zsh \
    PATH=/opt/ros/jazzy/bin:${PATH}:/opt/anaconda3/bin \
    QT_X11_NO_MITSHM=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ---------------------------------------------------------------------------
# Base / locale / apt sources
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt_opts="-o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"; \
    apt-get ${apt_opts} update; \
    apt-get ${apt_opts} install -y --no-install-recommends \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        locales \
        lsb-release \
        software-properties-common; \
    locale-gen en_US en_US.UTF-8; \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8; \
    add-apt-repository universe; \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
        > /etc/apt/sources.list.d/ros2.list

# ---------------------------------------------------------------------------
# C++ / PCL development environment
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt_opts="-o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"; \
    apt-get ${apt_opts} update; \
    apt-get ${apt_opts} install -y --no-install-recommends \
        bash-completion \
        build-essential \
        cmake \
        dbus-x11 \
        g++ \
        gcc \
        gdb \
        git \
        iproute2 \
        iputils-ping \
        less \
        libpcl-dev \
        lsof \
        nano \
        net-tools \
        ninja-build \
        pcl-tools \
        pkg-config \
        python3-pip \
        sudo \
        tar \
        tmux \
        unzip \
        vim \
        wget \
        zsh

# ---------------------------------------------------------------------------
# ROS2 / RViz desktop and GUI runtime dependencies
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt_opts="-o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"; \
    apt-get ${apt_opts} update; \
    apt-get ${apt_opts} install -y --no-install-recommends \
        libegl1 \
        libgl1 \
        libgl1-mesa-dri \
        libglx-mesa0 \
        libxcb-xinerama0 \
        libxkbcommon-x11-0 \
        mesa-utils \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-vcstool \
        ros-dev-tools \
        ros-${ROS_DISTRO}-desktop \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-rviz2 \
        x11-apps

# ---------------------------------------------------------------------------
# MoveIt
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt_opts="-o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"; \
    apt-get ${apt_opts} update; \
    apt-get ${apt_opts} install -y --no-install-recommends \
        ros-${ROS_DISTRO}-moveit \
        ros-${ROS_DISTRO}-moveit-planners \
        ros-${ROS_DISTRO}-moveit-ros-move-group \
        ros-${ROS_DISTRO}-moveit-servo \
        ros-${ROS_DISTRO}-moveit-setup-assistant

# ---------------------------------------------------------------------------
# Nav2
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt_opts="-o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"; \
    apt-get ${apt_opts} update; \
    apt-get ${apt_opts} install -y --no-install-recommends \
        ros-${ROS_DISTRO}-nav2-bringup \
        ros-${ROS_DISTRO}-nav2-common \
        ros-${ROS_DISTRO}-navigation2

# ---------------------------------------------------------------------------
# Gazebo / GZ Sim
# ---------------------------------------------------------------------------
RUN set -eux; \
    apt_opts="-o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"; \
    apt-get ${apt_opts} update; \
    apt-get ${apt_opts} install -y --no-install-recommends \
        ros-${ROS_DISTRO}-ros-gz \
        ros-${ROS_DISTRO}-ros-gz-bridge \
        ros-${ROS_DISTRO}-ros-gz-sim

# ---------------------------------------------------------------------------
# Local package cache
# ---------------------------------------------------------------------------
COPY docker/package/ /tmp/ros2-desktop-package/

# ---------------------------------------------------------------------------
# Anaconda (amd64 only)
# ---------------------------------------------------------------------------
RUN set -eux; \
    test "$(dpkg --print-architecture)" = "amd64"; \
    installer="/tmp/ros2-desktop-package/Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh"; \
    test -f "${installer}"; \
    echo "${ANACONDA_SHA256_AMD64}  ${installer}" | sha256sum -c -; \
    bash "${installer}" -b -p "${CONDA_DIR}"; \
    "${CONDA_DIR}/bin/conda" config --system --set auto_activate_base false; \
    "${CONDA_DIR}/bin/conda" clean -afy; \
    chmod -R a+rwX "${CONDA_DIR}"

# ---------------------------------------------------------------------------
# zsh plugins
# ---------------------------------------------------------------------------
RUN set -eux; \
    mkdir -p /opt/zsh/plugins; \
    test -f "/tmp/ros2-desktop-package/${ZSH_AUTOSUGGESTIONS_ARCHIVE}"; \
    test -f "/tmp/ros2-desktop-package/${ZSH_SYNTAX_HIGHLIGHTING_ARCHIVE}"; \
    tar -xzf "/tmp/ros2-desktop-package/${ZSH_AUTOSUGGESTIONS_ARCHIVE}" -C /opt/zsh/plugins; \
    tar -xzf "/tmp/ros2-desktop-package/${ZSH_SYNTAX_HIGHLIGHTING_ARCHIVE}" -C /opt/zsh/plugins; \
    rm -rf /opt/zsh/plugins/zsh-autosuggestions /opt/zsh/plugins/zsh-syntax-highlighting; \
    mv /opt/zsh/plugins/zsh-autosuggestions-* /opt/zsh/plugins/zsh-autosuggestions; \
    mv /opt/zsh/plugins/zsh-syntax-highlighting-* /opt/zsh/plugins/zsh-syntax-highlighting

# ---------------------------------------------------------------------------
# Permissions, shell setup, compatibility commands, and build-time checks
# ---------------------------------------------------------------------------
RUN set -eux; \
    mkdir -p /workspace /sros; \
    chmod -R a+rwX /workspace /sros; \
    echo 'ALL ALL=(ALL:ALL) NOPASSWD:ALL' > /etc/sudoers.d/99-all-nopasswd; \
    chmod 0440 /etc/sudoers.d/99-all-nopasswd; \
    if [ ! -e /bin/zsh ]; then ln -s /usr/bin/zsh /bin/zsh; fi; \
    { \
        echo '# Interactive history and completion'; \
        echo 'HISTFILE=/root/.zsh_history'; \
        echo 'HISTSIZE=10000'; \
        echo 'SAVEHIST=10000'; \
        echo 'setopt APPEND_HISTORY'; \
        echo 'setopt SHARE_HISTORY'; \
        echo 'setopt HIST_IGNORE_DUPS'; \
        echo 'setopt HIST_FIND_NO_DUPS'; \
        echo "zstyle ':completion:*' menu select"; \
        echo 'typeset -g ZSH_AUTOSUGGEST_STRATEGY=(history completion)'; \
        echo "typeset -g ZSH_AUTOSUGGEST_HIGHLIGHT_STYLE='fg=244'"; \
        echo "PROMPT='%F{cyan}%n@%m%f %F{yellow}[zsh]%f %F{blue}%~%f %# '"; \
        echo 'source /opt/ros/jazzy/setup.zsh'; \
        echo 'source /opt/anaconda3/etc/profile.d/conda.sh'; \
        echo 'autoload -Uz compinit'; \
        echo 'compinit -d /root/.zcompdump'; \
        echo 'source /opt/zsh/plugins/zsh-autosuggestions/zsh-autosuggestions.zsh'; \
        echo 'source /opt/zsh/plugins/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh'; \
        echo '# Run `conda activate base` when you want the Anaconda Python environment.'; \
    } > /root/.zshrc; \
    { \
        echo '#!/usr/bin/env bash'; \
        echo 'exec gz sim "$@"'; \
    } > /usr/local/bin/gazebo; \
    { \
        echo '#!/usr/bin/env bash'; \
        echo 'exec rviz2 "$@"'; \
    } > /usr/local/bin/rviz; \
    chmod +x /usr/local/bin/gazebo; \
    chmod +x /usr/local/bin/rviz; \
    rm -rf /tmp/ros2-desktop-package /var/lib/apt/lists/*; \
    command -v gcc; \
    command -v g++; \
    command -v git; \
    command -v cmake; \
    command -v zsh; \
    command -v rviz; \
    command -v rviz2; \
    bash -lc 'source /opt/ros/jazzy/setup.bash && command -v gz'; \
    command -v gazebo; \
    command -v conda; \
    pkg-config --exists pcl_common; \
    bash -lc 'source /opt/ros/jazzy/setup.bash && ros2 --help >/dev/null'; \
    bash -lc 'source /opt/ros/jazzy/setup.bash && ros2 launch nav2_bringup --help >/dev/null'; \
    bash -lc 'source /opt/ros/jazzy/setup.bash && ros2 launch moveit_setup_assistant setup_assistant.launch.py --help >/dev/null'

USER root
WORKDIR /workspace
CMD ["zsh"]
