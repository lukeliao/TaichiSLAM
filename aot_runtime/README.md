# TaichiSLAM AOT Runtime

高性能3D建图算法AOT（Ahead-of-Time）部署方案，支持在ARM Mali GPU上通过Taichi C-API运行。

## 特性

- **跨平台AOT部署**：支持ARM Mali GPU（Vulkan/OpenGL ES）、桌面GPU（CUDA/Vulkan）
- **高性能建图算法**：Octomap、TSDF（Truncated Signed Distance Function）
- **C++接口**：简洁易用的C++ API，方便集成到其他算法
- **实时性能**：GPU加速，支持实时3D建图

## 系统要求

### 开发环境
- C++17兼容编译器（GCC 7+, Clang 5+, MSVC 2017+）
- CMake 3.14+
- Taichi C-API库

### 目标平台
- **ARM SoC with Mali GPU**
  - Mali-G系列（G31, G51, G52, G71, G72, G76, G77等）
  - 支持Vulkan 1.1+ 或 OpenGL ES 3.1+
  - 推荐：至少2GB RAM

- **桌面平台**
  - NVIDIA GPU with CUDA 10.0+
  - 或支持Vulkan 1.1+的GPU

## 快速开始

### 1. 安装Taichi C-API

下载Taichi C-API库：

```bash
# 方法1：从Taichi官方发布下载
wget https://github.com/taichi-dev/taichi/releases/download/v1.7.0/taichi-1.7.0-linux.zip
unzip taichi-1.7.0-linux.zip -d /opt/taichi

# 方法2：使用pip安装的taichi
pip install taichi==1.7.0
export TAICHI_C_API_PATH=$(python -c "import taichi; print(taichi.__path__[0])")
```

### 2. 构建AOT运行时库

```bash
cd /path/to/TaichiSLAM/aot_runtime

# 创建构建目录
mkdir build && cd build

# 配置（桌面平台）
cmake .. -DTAICHI_C_API_PATH=/opt/taichi/c_api

# 或者交叉编译（ARM平台）
cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/arm64-toolchain.cmake \
         -DTAICHI_C_API_PATH=/path/to/taichi/c_api/arm64

# 构建
make -j$(nproc)

# 安装（可选）
sudo make install
```

### 3. 导出AOT模块

在开发机上导出AOT模块：

```bash
cd /path/to/TaichiSLAM

# 导出Octomap AOT模块（Vulkan后端）
python aot_export.py \
    --method octo \
    --arch vulkan \
    --output ./aot_modules

# 导出TSDF AOT模块（OpenGL ES后端，适合ARM Mali）
python aot_export.py \
    --method tsdf \
    --arch gles \
    --output ./aot_modules
```

### 4. 运行示例程序

```bash
# 设置库路径
export LD_LIBRARY_PATH=/opt/taichi/c_api/lib:$LD_LIBRARY_PATH

# 运行基本示例
./build/examples/example_basic_octo ./aot_modules/taichislam_octo

# 运行ARM Mali示例
./build/examples/example_arm_mali ./aot_modules/taichislam_octo
```

## C++ API使用指南

### 基本用法

```cpp
#include <taichislam_aot.h>
#include <iostream>

int main() {
    using namespace taichislam;

    // 1. 初始化运行时（Vulkan后端）
    TaichiSLAMRuntime runtime(TI_ARCH_VULKAN, 0);

    // 2. 加载AOT模块
    MappingConfig config;
    config.map_size_xy = 20.0f;    // 20m x 20m map
    config.map_size_z = 5.0f;      // 5m height
    config.voxel_scale = 0.1f;     // 10cm voxels

    auto mapping = runtime.loadOctomap(
        "./aot_modules/taichislam_octo", config);

    // 3. 准备点云数据
    std::vector<float> points = {
        1.0f, 0.0f, 0.5f,   // Point 1
        2.0f, 0.0f, 0.5f,   // Point 2
        3.0f, 0.0f, 0.5f,   // Point 3
        // ... more points
    };
    std::vector<uint8_t> colors = {
        255, 0, 0,    // Red
        0, 255, 0,    // Green
        0, 0, 255,    // Blue
        // ... more colors
    };

    // 4. 设置传感器位姿
    Pose pose;
    // Rotation matrix (identity - facing forward)
    pose.R[0] = 1.0f; pose.R[1] = 0.0f; pose.R[2] = 0.0f;
    pose.R[3] = 0.0f; pose.R[4] = 1.0f; pose.R[5] = 0.0f;
    pose.R[6] = 0.0f; pose.R[7] = 0.0f; pose.R[8] = 1.0f;
    // Translation (at origin, 1m above ground)
    pose.T[0] = 0.0f;  // X
    pose.T[1] = 0.0f;  // Y
    pose.T[2] = 1.0f;  // Z

    // 5. 插入点云到地图
    std::cout << "Inserting " << points.size() / 3 << " points..." << std::endl;
    mapping->insertPointCloud(
        points.data(),
        colors.data(),
        points.size() / 3,
        pose);

    // 6. 提取占据体素
    std::cout << "Extracting voxels..." << std::endl;
    VoxelGrid voxels = mapping->extractVoxels(0);
    std::cout << "Extracted " << voxels.num_voxels << " occupied voxels" << std::endl;

    // 7. 获取地图统计信息
    size_t total_voxels;
    float map_size_mb;
    mapping->getStatistics(&total_voxels, &map_size_mb);
    std::cout << "Map statistics:" << std::endl;
    std::cout << "  Total voxels: " << total_voxels << std::endl;
    std::cout << "  Memory usage: " << map_size_mb << " MB" << std::endl;

    // 8. 重置地图
    mapping->reset();
    std::cout << "Map reset" << std::endl;

    std::cout << std::endl;
    std::cout << "Example completed successfully!" << std::endl;

    return 0;
}
