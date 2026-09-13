#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACKAGE_DIR="${SCRIPT_DIR}/package"


IMAGE_NAME="dev_lgc:latest"
WORKSPACE_DIR="${REPO_ROOT}"
PROJECT_NAME="$(basename "${WORKSPACE_DIR}")"
CONTAINER_NAME="${PROJECT_NAME}"

if [ ! -d "${WORKSPACE_DIR}" ]; then
    echo "工作区目录不存在: ${WORKSPACE_DIR}" >&2
    exit 1
fi

WORKSPACE_DIR="$(cd "${WORKSPACE_DIR}" && pwd)"


ANACONDA_VERSION="${ANACONDA_VERSION:-2026.07-1}"
ANACONDA_AMD64="Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh"
ANACONDA_AMD64_URL="https://repo.anaconda.com/archive/${ANACONDA_AMD64}"


ZSH_AUTOSUGGESTIONS_ARCHIVE="zsh-autosuggestions-0.7.1.tar.gz"
ZSH_AUTOSUGGESTIONS_URL="https://github.com/zsh-users/zsh-autosuggestions/archive/refs/tags/v0.7.1.tar.gz"


ZSH_SYNTAX_HIGHLIGHTING_ARCHIVE="zsh-syntax-highlighting-0.8.0.tar.gz"
ZSH_SYNTAX_HIGHLIGHTING_URL="https://github.com/zsh-users/zsh-syntax-highlighting/archive/refs/tags/0.8.0.tar.gz"


TENSORRT_VERSION="${ROS2_DESKTOP_TENSORRT_VERSION:-10.16.1.11}"
TENSORRT_ROOT="${ROS2_DESKTOP_TENSORRT_ROOT:-/home/standard/soft/tensorRT}"
TENSORRT_HOST_DIR="${ROS2_DESKTOP_TENSORRT_DIR:-${TENSORRT_ROOT}/TensorRT-${TENSORRT_VERSION}}"
TENSORRT_CONTAINER_ENV_DIR="${ROS2_DESKTOP_TENSORRT_ENV_DIR:-${TENSORRT_ROOT}/container-tensorrt-py312}"
TENSORRT_PYTHON_WHEEL="tensorrt-${TENSORRT_VERSION}-cp312-none-linux_x86_64.whl"


# 打印脚本支持的命令和使用方式。
usage() {
    cat <<EOF
用法:
  $0                 检查 package, 构建镜像, 进入容器
  $0 check           只检查 package 缓存
  $0 build           检查 package 后构建镜像
  $0 shell           检查 package, 必要时构建镜像, 进入 zsh
  $0 vscode          检查 package, 必要时构建镜像, 后台启动供 VS Code 连接
  $0 run <command>   检查 package, 必要时构建镜像, 在容器中执行命令

示例:
  # 先检查 package 目录中的 Anaconda 和 zsh 插件包是否齐全。
  bash docker/ros2-desktop-anaconda.sh check

  # 只构建镜像,不进入容器。
  bash docker/ros2-desktop-anaconda.sh build

  # 构建或复用镜像后进入容器 zsh 终端。
  bash docker/ros2-desktop-anaconda.sh shell

  # 后台启动容器,然后用 VS Code Dev Containers 插件连接。
  bash docker/ros2-desktop-anaconda.sh vscode

  # 无参数运行时会检查 package,构建镜像,然后进入容器。
  bash docker/ros2-desktop-anaconda.sh

  # 在容器内直接启动 RViz GUI。
  bash docker/ros2-desktop-anaconda.sh run rviz2

  # 在容器内直接启动 Gazebo GUI。
  bash docker/ros2-desktop-anaconda.sh run gz sim

  # 临时指定镜像名后构建。
  ROS2_DESKTOP_ANACONDA_IMAGE=my/ros2-desktop:jazzy bash docker/ros2-desktop-anaconda.sh build

  # 没有 NVIDIA GPU 的机器需要显式允许无 GPU 进入容器。
  ROS2_DESKTOP_ALLOW_NO_GPU=1 bash docker/ros2-desktop-anaconda.sh shell

  # 指定 TensorRT 解压目录。
  ROS2_DESKTOP_TENSORRT_DIR=/path/to/TensorRT-${TENSORRT_VERSION} \
    bash docker/ros2-desktop-anaconda.sh shell
EOF
}

# 确认当前宿主机是 amd64 架构,本镜像脚本暂不处理 arm。
ensure_amd64_host() {
    if [ "$(uname -m)" != "x86_64" ]; then
        echo "当前脚本只支持 amd64/x86_64 宿主架构,检测到: $(uname -m)" >&2
        return 1
    fi
}

# 检查 package 目录下的单个缓存文件是否存在,缺失时打印下载地址。
check_one_file() {
    local file="$1"
    local url="$2"
    if [ ! -f "${PACKAGE_DIR}/${file}" ]; then  # 判断文件存在性
        printf '  缺少: %s\n' "${PACKAGE_DIR}/${file}" >&2  # 适合格式化输出
        printf '  下载: %s\n' "wget ${url}" >&2
        echo "---------------------------------------------------------"
        return 1
    fi
}

# 检查宿主机 TensorRT 解压目录中的运行库和 Python wheel。
check_tensorrt() {
    local missing=0
    local required_file

    for required_file in \
        "${TENSORRT_HOST_DIR}/bin/trtexec" \
        "${TENSORRT_HOST_DIR}/lib/libnvinfer.so.10" \
        "${TENSORRT_HOST_DIR}/python/${TENSORRT_PYTHON_WHEEL}"; do
        if [ ! -e "${required_file}" ]; then
            printf '  缺少 TensorRT 文件: %s\n' "${required_file}" >&2
            missing=1
        fi
    done

    if [ "${missing}" -ne 0 ]; then
        cat >&2 <<EOF

TensorRT 缓存不完整。默认查找目录:
  ${TENSORRT_HOST_DIR}

可通过 ROS2_DESKTOP_TENSORRT_DIR 指定 TensorRT-${TENSORRT_VERSION} 解压目录。
EOF
        return 1
    fi
}

# 检查构建镜像所需的本地 package 缓存是否完整。
check_package() {
    mkdir -p "${PACKAGE_DIR}"
    local missing=0

    ensure_amd64_host

    check_one_file "${ANACONDA_AMD64}" "${ANACONDA_AMD64_URL}" || missing=1
    check_one_file "${ZSH_AUTOSUGGESTIONS_ARCHIVE}" "${ZSH_AUTOSUGGESTIONS_URL}" || missing=1
    check_one_file "${ZSH_SYNTAX_HIGHLIGHTING_ARCHIVE}" "${ZSH_SYNTAX_HIGHLIGHTING_URL}" || missing=1
    check_tensorrt || missing=1

    if [ "${missing}" -ne 0 ]; then   # 不等于0
        cat >&2 <<EOF

package 缓存不完整。请手动下载上面的文件并放到:
  ${PACKAGE_DIR}

脚本不会自动下载这些文件; 文件存在后再次运行即可。
EOF
        return 1
    fi

    printf 'package 缓存完整。\n'
}

# 使用当前仓库作为 Docker build context 构建镜像。
build_image(){
    check_package
    printf '构建镜像: %s\n' "${IMAGE_NAME}"
    printf '构建上下文: %s\n' "${REPO_ROOT}"
    docker build \
        --pull=false \
        -t "${IMAGE_NAME}" \
        -f "${SCRIPT_DIR}/Dockerfile" \
        "${REPO_ROOT}"  
}

# 判断目标镜像是否已经存在于本地 Docker。
image_exists() {
    docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1  # 检查镜像是否存在，不在终端打印任何内容
}

# 确保本地存在可运行镜像,不存在时自动执行构建。
ensure_image() {
    check_package
    if ! image_exists; then
        build_image
    fi
}

# 准备 X11 显示环境,让容器内 rviz2/gz sim 可以弹出 GUI。
prepare_x11() {
    if [ -z "${DISPLAY:-}" ]; then    # 通过判断字符串为空，如果当前没有设置 DISPLAY变量
        if [ -n "${WAYLAND_DISPLAY:-}" ]; then  # 如果检测到宿主机在用Wayland,判断变量非空
            export DISPLAY=:1
        else
            export DISPLAY=:0
        fi
    fi

    xhost +local:docker >/dev/null 2>&1 || true   # 允许本机Docker容器访问宿主机的X11图形服务，即使xhost执行失败，也不要让整个脚本退出
}

docker_xauth_args=()
# 准备 Xauthority 挂载参数,尽量复用宿主机图形认证信息。
# 找到宿主机的X11认证文件，然后挂进Docker容器里，让容器里的GUI程序有权限连接宿主机显示器,主要是解决下x11的访问权限问题
prepare_xauthority() {
    local host_xauthority="${XAUTHORITY:-}"
    if [ -z "${host_xauthority}" ] && [ -f "${HOME}/.Xauthority" ]; then
        host_xauthority="${HOME}/.Xauthority"
    elif [ -z "${host_xauthority}" ] && [ -f "/run/user/$(id -u)/gdm/Xauthority" ]; then
        host_xauthority="/run/user/$(id -u)/gdm/Xauthority"
    fi
    if [ -n "${host_xauthority}" ] && [ -f "${host_xauthority}" ]; then
        docker_xauth_args=(
            -v "${host_xauthority}:/tmp/ros2-desktop.Xauthority:ro"
            -e "XAUTHORITY=/tmp/ros2-desktop.Xauthority"
        )
    fi
}

docker_device_args=()
# 挂普通硬件设备,准备显卡渲染和声卡设备挂载参数。
prepare_devices() {
    docker_device_args=()
    if [ -d /dev/dri ]; then  # 判断目录是否为空，/dev/dri 是 Linux 图形渲染设备目录，常用于 OpenGL、Mesa、Intel/AMD 核显、部分图形显示能力。
        docker_device_args+=(--device=/dev/dri)
    fi
    if [ -d /dev/snd ]; then  # 判断主机是否存在声卡设备目录
        docker_device_args+=(--device=/dev/snd)
    fi
}

docker_gpu_args=()
# 检查 NVIDIA 容器运行时并准备 GPU 透传参数。
prepare_gpu() {
    docker_gpu_args=()
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "未检测到宿主 nvidia-smi; 无法保证容器内 nvidia-smi 显示 GPU 型号。" >&2
        echo "如确实要无 NVIDIA GPU 进入容器,设置 ROS2_DESKTOP_ALLOW_NO_GPU=1。" >&2
        [ "${ROS2_DESKTOP_ALLOW_NO_GPU:-0}" = "1" ] || return 1
        return 0
    fi

    if ! docker run --rm --gpus all "${IMAGE_NAME}" nvidia-smi -L >/dev/null 2>&1; then
        echo "Docker 的 --gpus all 测试失败; 容器内无法使用 NVIDIA GPU。" >&2
        echo "请检查 nvidia-container-toolkit、CDI 配置和 Docker daemon 状态。" >&2
        [ "${ROS2_DESKTOP_ALLOW_NO_GPU:-0}" = "1" ] || return 1
        return 0
    fi

    docker_gpu_args=(
        --gpus all       # 把所有的GPU暴露给容器
        -e NVIDIA_VISIBLE_DEVICES=all   # 容器内可见所有 NVIDIA GPU。
        -e NVIDIA_DRIVER_CAPABILITIES=all  # 开启 NVIDIA 驱动能力，包括 compute、graphics、utility 等。
        -e __GLX_VENDOR_LIBRARY_NAME=nvidia  # 让 OpenGL/GLX 优先使用 NVIDIA。
        -e __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json # 让 EGL 也优先走 NVIDIA 相关配置。
    )
}

docker_tensorrt_args=()
# 准备 TensorRT 宿主目录和容器内持久化 Python 3.12 环境。
prepare_tensorrt() {
    check_tensorrt
    mkdir -p "${TENSORRT_ROOT}"

    local marker="${TENSORRT_CONTAINER_ENV_DIR}/.tensorrt-version"
    if [ ! -f "${marker}" ] || [ "$(cat "${marker}")" != "${TENSORRT_VERSION}" ]; then
        if [ -e "${TENSORRT_CONTAINER_ENV_DIR}" ]; then
            rm -rf "${TENSORRT_CONTAINER_ENV_DIR}"
        fi
        mkdir -p "${TENSORRT_CONTAINER_ENV_DIR}"

        docker run --rm \
            --user "$(id -u):$(id -g)" \
            -e HOME=/tmp \
            -e CONDA_PKGS_DIRS=/tmp/conda-pkgs \
            -v "${TENSORRT_HOST_DIR}:/opt/tensorrt:ro" \
            -v "${TENSORRT_CONTAINER_ENV_DIR}:/opt/tensorrt-conda:rw" \
            "${IMAGE_NAME}" \
            bash -lc "
                set -eux
                conda create --yes --override-channels --channel conda-forge \
                    --prefix /opt/tensorrt-conda python=3.12 pip
                /opt/tensorrt-conda/bin/python -m pip install \
                    --no-index --no-deps --force-reinstall \
                    /opt/tensorrt/python/${TENSORRT_PYTHON_WHEEL}
            "
        printf '%s\n' "${TENSORRT_VERSION}" > "${marker}"
    fi

    docker_tensorrt_args=(
        -v "${TENSORRT_HOST_DIR}:/opt/tensorrt:ro"
        -v "${TENSORRT_CONTAINER_ENV_DIR}:/opt/tensorrt-conda:ro"
        -e TENSORRT_HOME=/opt/tensorrt
        -e PATH=/opt/tensorrt/bin:/opt/tensorrt-conda/bin:/opt/ros/jazzy/bin:/opt/anaconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
        -e LD_LIBRARY_PATH=/opt/tensorrt/lib
    )
}

# 生成交互终端和 VS Code 后台容器共用的 Docker 参数。
docker_run_args=()
prepare_docker_run() {
    prepare_x11
    prepare_xauthority
    prepare_devices
    prepare_gpu
    prepare_tensorrt
    mkdir -p "${HOME}/sros"

    docker_run_args=(
        --name "${CONTAINER_NAME}"
        --hostname "${CONTAINER_NAME}"
        --user root
        --privileged
        --cap-add=ALL
        --security-opt seccomp=unconfined
        --security-opt apparmor=unconfined
        --network host
        --ipc host
        "${docker_gpu_args[@]}" \
        "${docker_device_args[@]}" \
        "${docker_tensorrt_args[@]}"
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw
        "${docker_xauth_args[@]}" \
        -v /etc/localtime:/etc/localtime:ro
        -v /etc/timezone:/etc/timezone:ro
        -v "${WORKSPACE_DIR}:/workspace:rw"
        -v "${HOME}/sros:/sros:rw"
        -e "DISPLAY=${DISPLAY}"
        -e QT_X11_NO_MITSHM=1
        -e "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
        -w /workspace
    )
}

# 统一封装交互式 docker run 参数,以 root/privileged/host 网络方式进入容器。
docker_run_base() {
    prepare_docker_run
    remove_existing_container

    docker run --rm -it \
        "${docker_run_args[@]}" \
        "${IMAGE_NAME}" \
        "$@"
}

# 获取同名容器当前绑定到 /workspace 的宿主机目录。
container_workspace_source() {
    docker container inspect "${CONTAINER_NAME}" \
        --format '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}'
}

# 交互式运行使用 --rm；若同名旧容器存在，先移除以避免名称冲突和旧挂载配置残留。
remove_existing_container() {
    if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        return 0
    fi

    local current_workspace
    current_workspace="$(container_workspace_source 2>/dev/null || true)"
    if [ "${current_workspace}" != "${WORKSPACE_DIR}" ]; then
        printf '移除旧容器 %s（当前 /workspace=%s）\n' \
            "${CONTAINER_NAME}" "${current_workspace:-未挂载}" >&2
    fi
    docker rm -f "${CONTAINER_NAME}" >/dev/null
}

# 检查或构建镜像后进入容器内 zsh。
enter_shell() {
    ensure_image
    docker_run_base zsh -i
}

# 检查或构建镜像后在容器内执行指定命令。
run_command() {
    ensure_image
    docker_run_base zsh -lc 'source /root/.zshrc >/dev/null 2>&1 || true; exec "$@"' zsh "$@"
}

# 后台启动一个不会自行退出的容器,供 VS Code Dev Containers 插件附加。
start_vscode_container() {
    ensure_image

    if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        local current_workspace
        current_workspace="$(container_workspace_source 2>/dev/null || true)"
        if [ "$(docker container inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")" = "true" ] \
            && [ "${current_workspace}" = "${WORKSPACE_DIR}" ]; then
            printf '容器已经在运行: %s\n' "${CONTAINER_NAME}"
            printf '请在 VS Code 中执行 “Dev Containers: Attach to Running Container...”。\n'
            return 0
        fi
        if [ "${current_workspace}" != "${WORKSPACE_DIR}" ]; then
            printf '移除旧容器 %s（当前 /workspace=%s）\n' \
                "${CONTAINER_NAME}" "${current_workspace:-未挂载}" >&2
        fi
        docker rm -f "${CONTAINER_NAME}" >/dev/null
    fi

    prepare_docker_run
    docker run --rm -d \
        "${docker_run_args[@]}" \
        "${IMAGE_NAME}" \
        sleep infinity >/dev/null

    printf 'VS Code 容器已后台启动: %s\n' "${CONTAINER_NAME}"
    printf '请在 VS Code 中执行 “Dev Containers: Attach to Running Container...”，选择该容器，再选择 /workspace。\n'
    printf '停止容器: docker stop %s\n' "${CONTAINER_NAME}"
}

cmd="${1:-default}"
case "${cmd}" in
    default)
        build_image
        docker_run_base zsh -i
        ;;
    check)
        check_package
        ;;
    build)
        build_image
        ;;
    shell)
        enter_shell
        ;;
    vscode)
        start_vscode_container
        ;;
    run)
        shift
        if [ "$#" -eq 0 ]; then
            echo "run 需要跟容器内命令。" >&2
            usage >&2
            exit 2
        fi
        run_command "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "未知命令: ${cmd}" >&2
        usage >&2
        exit 2
        ;;
esac