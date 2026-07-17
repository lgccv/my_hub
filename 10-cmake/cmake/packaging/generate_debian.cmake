# ============================================================================
# generate_debian.cmake — 动态生成 deb 包的 DEBIAN 控制文件和脚本
#
# 用法（通过 cmake -P 调用）：
#   cmake -DPKG_NAME=perception-platform \
#         -DVERSION=0.5.0 \
#         -DARCH=amd64 \
#         -DOUTPUT_DIR=/path/to/staging/DEBIAN \
#         -DCONFIG_SRC_DIR=/opt/perception_platform/config \
#         -DCONFIG_DST_SUBDIR=perception_platform/config \
#         -P cmake/packaging/generate_debian.cmake
#
# 参数说明：
#   PKG_NAME          — 包名（如 perception-platform）
#   VERSION           — 版本号
#   ARCH              — 架构（amd64 / arm64）
#   OUTPUT_DIR        — 输出目录（DEBIAN 目录路径）
#   CONFIG_SRC_DIR    — 配置文件在安装包中的路径（默认 /opt/perception_platform/config）
#   CONFIG_DST_SUBDIR — 配置文件在 ~/.local/share 下的子目录（默认 perception_platform/config）
# ============================================================================
cmake_minimum_required(VERSION 3.16)

# 参数校验
foreach(_var PKG_NAME VERSION ARCH OUTPUT_DIR)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "Required variable ${_var} is not defined")
    endif()  # if NOT DEFINED
endforeach()  # foreach _var

# 默认值
if(NOT DEFINED CONFIG_SRC_DIR)
    set(CONFIG_SRC_DIR "/opt/perception_platform/config")
endif()  # if NOT DEFINED CONFIG_SRC_DIR

if(NOT DEFINED CONFIG_DST_SUBDIR)
    set(CONFIG_DST_SUBDIR "perception_platform/config")
endif()  # if NOT DEFINED CONFIG_DST_SUBDIR

# ============================================================================
# 1. 生成 control 文件（从 control.in 模板替换占位符）
# ============================================================================
get_filename_component(_TEMPLATE_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
set(_CONTROL_IN "${_TEMPLATE_DIR}/${PKG_NAME}.control.in")

if(NOT EXISTS "${_CONTROL_IN}")
    message(FATAL_ERROR "Control template not found: ${_CONTROL_IN}")
endif()  # if NOT EXISTS

file(READ "${_CONTROL_IN}" _CONTROL_CONTENT)
string(REPLACE "@VERSION@" "${VERSION}" _CONTROL_CONTENT "${_CONTROL_CONTENT}")
string(REPLACE "@ARCH@"    "${ARCH}"    _CONTROL_CONTENT "${_CONTROL_CONTENT}")

file(MAKE_DIRECTORY "${OUTPUT_DIR}")
file(WRITE "${OUTPUT_DIR}/control" "${_CONTROL_CONTENT}")
message(STATUS "Generated: ${OUTPUT_DIR}/control")

# ============================================================================
# 2. 生成 postinst 脚本
# ============================================================================
set(_POSTINST_CONTENT "#!/bin/bash
set -e
ldconfig

# ============================================================================
# Install config files to user's ~/.local directory
# For each yaml file, skip if it already exists at the destination.
#
# User detection logic:
#   - SUDO_USER is set when a normal user runs 'sudo dpkg -i ...'
#   - If SUDO_USER is unset or is 'root', abort config installation
#     (direct root login is not a valid deployment scenario)
#
# Examples:
#   $ sudo dpkg -i xxx.deb          # SUDO_USER=nanzj -> /home/nanzj/.local/share/...
#   $ sudo -u deploy sudo dpkg ...  # SUDO_USER=deploy -> /home/deploy/.local/share/...
#   # root login, dpkg -i xxx.deb   # SUDO_USER unset, USER=root -> skip
#   # sudo su; dpkg -i xxx.deb      # SUDO_USER=root -> skip
# ============================================================================
CONFIG_SRC=\"${CONFIG_SRC_DIR}\"
REAL_USER=\"\${SUDO_USER:-}\"

# Reject root user: config must be installed to a normal user's home
if [[ -z \"\$REAL_USER\" || \"\$REAL_USER\" == \"root\" ]]; then
    echo \"[WARN] Cannot determine non-root user (SUDO_USER='\${SUDO_USER:-}'). Skipping config installation.\"
    echo \"[WARN] Please run: sudo dpkg -i <package>.deb (as a normal user with sudo).\"
    exit 0
fi  # if REAL_USER is empty or root

REAL_HOME=\$(getent passwd \"\$REAL_USER\" | cut -d: -f6)
if [[ -z \"\$REAL_HOME\" || ! -d \"\$REAL_HOME\" ]]; then
    echo \"[WARN] Home directory for user '\$REAL_USER' not found. Skipping config installation.\"
    exit 0
fi  # if REAL_HOME not found

CONFIG_DST=\"\${REAL_HOME}/.local/share/${CONFIG_DST_SUBDIR}\"

if [[ -d \"\$CONFIG_SRC\" ]]; then
    find \"\$CONFIG_SRC\" -name '*.yaml' -type f | while read -r src_file; do
        rel_path=\"\${src_file#\$CONFIG_SRC/}\"
        dst_file=\"\$CONFIG_DST/\$rel_path\"
        dst_dir=\$(dirname \"\$dst_file\")
        mkdir -p \"\$dst_dir\"
        if [[ -f \"\$dst_file\" ]]; then
            echo \"Config file already exists, skipping: \$dst_file\"
        else
            cp \"\$src_file\" \"\$dst_file\"
            echo \"Installed config: \$dst_file\"
        fi  # if dst_file exists
    done
    # Fix ownership and ensure read-write permissions for the user
    # chown/chmod the top-level parent dir as well (mkdir -p creates it as root)
    chown -R \"\${REAL_USER}:\$(id -gn \"\$REAL_USER\")\" \"\${REAL_HOME}/.local/share/perception_platform\"
    chmod -R u+rw \"\${REAL_HOME}/.local/share/perception_platform\"
fi  # if CONFIG_SRC exists
")

file(WRITE "${OUTPUT_DIR}/postinst" "${_POSTINST_CONTENT}")
message(STATUS "Generated: ${OUTPUT_DIR}/postinst")

# ============================================================================
# 3. 生成 postrm 脚本
# ============================================================================
set(_POSTRM_CONTENT "#!/bin/bash
set -e
ldconfig
")

file(WRITE "${OUTPUT_DIR}/postrm" "${_POSTRM_CONTENT}")
message(STATUS "Generated: ${OUTPUT_DIR}/postrm")
