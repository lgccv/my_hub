# 如何断点调试cmake
1、安装cmake tools插件
2、参考.vscode中的写法

# 如何给cmake文件传参数

# install是怎么写的，怎么设置路径
build_dir="./build"
INSTALL_PREFIX="/opt/perception-platform"
DESTDIR="$staging" cmake --install "$build_dir" --prefix "$INSTALL_PREFIX"  暂时存放到$staging/$INSTALL_PREFIX


# 常见变量是什么意思？



# 如何用cmake生成debian的文件
```python
    # --- 2. ldconfig 配置 ---
    cat > "$staging/etc/ld.so.conf.d/perception-platform.conf" << LDEOF
${INSTALL_PREFIX}/lib
${INSTALL_PREFIX}/lib64
${INSTALL_PREFIX}/lib/plugins
LDEOF

    # --- 3. 环境变量配置（可选，供 shell 用户使用） ---
    cat > "$staging/etc/profile.d/perception-platform.sh" << 'ENVEOF'
# perception_platform 环境变量
export PERCEPTION_PLATFORM_DIR="/opt/perception_platform"
export PERCEPTION_PLATFORM_PLUGIN_DIR="/opt/perception_platform/lib/plugins"
# 将 CMAKE_PREFIX_PATH 追加到环境中，便于 CMake 查找
if [[ ":${CMAKE_PREFIX_PATH:-}:" != *":${PERCEPTION_PLATFORM_DIR}:"* ]]; then
    export CMAKE_PREFIX_PATH="${PERCEPTION_PLATFORM_DIR}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
fi
ENVEOF
```
output_dir="/workspace/src/perception-platform/dist/staging/perception-platform/DEBIAN"  # 生成的文件放到这里
GENERATE_DEBIAN_CMAKE="/workspace/src/perception-platform/cmake/packaging/generate_debian.cmake"
cmake \
    -DPKG_NAME="$pkg_name" \
    -DVERSION="$version" \
    -DARCH="$arch" \
    -DOUTPUT_DIR="$output_dir" \
    -DCONFIG_SRC_DIR="${INSTALL_PREFIX}/config" \
    -DCONFIG_DST_SUBDIR="perception_platform/config" \
    -P "$GENERATE_DEBIAN_CMAKE"
chmod 755 "$output_dir/postinst" "output_dir/postrm"

staging="/workspace/src/perception-platform/dist/staging/perception-platform"
dpkg-deb --build --root-owner-group "${staging}" "$OUTPUT_DIR/$deb_name"

### 安装
sudo dpkg -i xxx.deb


# CMAKE_EXTRA_ARGS是什么？

# COLCON_EXTRA_ARGS是什么？