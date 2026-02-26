# TaichiSLAM AOT (Ahead-of-Time) 部署方案

本项目提供完整的TaichiSLAM AOT部署方案，支持将3D建图算法（Octomap/TSDF）编译为AOT模块，在ARM Mali GPU等目标设备上通过Taichi C-API高效运行。

## 架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TaichiSLAM AOT Workflow                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐      AOT Export      ┌──────────────────┐     │
│  │  Python/Taichi   │  ─────────────────►  │   AOT Module     │     │
│  │  (Development)   │                     │  (GPU Shaders)   │     │
│  │                  │                     │                  │     │
│  │  - Octomap       │                     │  - metadata.json │     │
│  │  - TSDF          │                     │  - kernels.spv   │     │
│  │  - Kernels       │                     │  - graphs.tcb    │     │
│  └──────────────────┘                     └────────┬─────────┘     │
│                                                   │                 │
│                                                   │ Load & Execute │
│                                                   ▼                 │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    Target Device (ARM SoC)                    │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │ │
│  │  │  C++ App   │──►│ Taichi C-API │──►│   GPU (Mali/Vulkan)  │ │ │
│  │  │            │   │  Runtime     │   │  - Execute kernels   │ │ │
│  │  │ - PointCloud │ │  - Load AOT  │   │  - Update voxels     │ │ │
│  │  │ - Pose       │   │  - Memory    │   │  - Extract surface   │ │ │
│  │  │ - Extract    │   │  - Synchronization  │                    │ │ │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## 目录结构

```
TaichiSLAM/
├── aot_export.py              # AOT模块导出脚本
├── aot_runtime/               # C++运行时库
│   ├── include/
│   │   └── taichislam_aot.h   # C++接口头文件
│   ├── src/
│   │   └── taichislam_aot.cpp # 实现文件
│   ├── examples/
│   │   ├── basic_octo.cpp     # 基础示例
│   │   ├── basic_tsdf.cpp     # TSDF示例
│   │   └── arm_mali_example.cpp # ARM Mali部署示例
│   ├── cmake/
│   │   ├── arm64-toolchain.cmake    # ARM64交叉编译工具链
│   │   └── TaichiSLAM_AOTConfig.cmake.in
│   └── CMakeLists.txt
└── AOT_README.md              # 本文件
```

## 快速开始

### 1. 环境准备

#### 开发主机（x86_64 Linux）

```bash
# 安装依赖
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget

# 安装Python和Taichi
pip install taichi==1.7.0 numpy

# 下载Taichi C-API
mkdir -p ~/taichi && cd ~/taichi
wget https://github.com/taichi-dev/taichi/releases/download/v1.7.0/taichi-1.7.0-linux.zip
unzip taichi-1.7.0-linux.zip
export TAICHI_C_API_PATH=~/taichi/c_api
```

#### 目标设备（ARM SoC with Mali GPU）

```bash
# 在目标设备上安装Taichi C-API
# 方法1：使用预编译的ARM版本
wget https://github.com/taichi-dev/taichi/releases/download/v1.7.0/taichi-1.7.0-linux-arm64.zip
unzip taichi-1.7.0-linux-arm64.zip -d /opt/taichi

# 方法2：从pip安装（如果可用）
pip install taichi==1.7.0

# 检查GPU支持
ls /dev/mali*  # Mali GPU设备
ls /dev/dri/*  # DRI设备

# 设置权限
sudo usermod -aG render $USER  # 添加用户到render组
```

### 2. 导出AOT模块

```bash
cd /path/to/TaichiSLAM

# 导出Octomap AOT模块（Vulkan，适合ARM Mali-G系列）
python aot_export.py \
    --method octo \
    --arch vulkan \
    --output ./aot_modules

# 导出TSDF AOT模块（OpenGL ES，适合旧版Mali GPU）
python aot_export.py \
    --method tsdf \
    --arch gles \
    --output ./aot_modules

# 查看导出的模块
ls -la ./aot_modules/
```

### 3. 构建运行时库

#### 主机编译（用于测试）

```bash
cd /path/to/TaichiSLAM/aot_runtime
mkdir -p build && cd build

cmake .. \
    -DTAICHI_C_API_PATH=$TAICHI_C_API_PATH \
    -DBUILD_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

#### 交叉编译（用于ARM目标）

```bash
cd /path/to/TaichiSLAM/aot_runtime
mkdir -p build_arm && cd build_arm

# 配置交叉编译
cmake .. \
    -DCMAKE_TOOLCHAIN_FILE=../cmake/arm64-toolchain.cmake \
    -DTAICHI_C_API_PATH=/path/to/taichi/arm64/c_api \
    -DBUILD_EXAMPLES=ON \
    -DCMAKE_BUILD_TYPE=Release

# 编译
make -j$(nproc)

# 部署到目标设备
scp -r examples/aot_modules user@target_device:/path/to/deploy
scp examples/example_arm_mali user@target_device:/path/to/deploy
```

### 4. 运行

#### 主机运行

```bash
cd /path/to/TaichiSLAM/aot_runtime/build

export LD_LIBRARY_PATH=$TAICHI_C_API_PATH/lib:$LD_LIBRARY_PATH

./examples/example_basic_octo \
    ../../aot_modules/taichislam_octo
```

#### ARM目标设备运行

```bash
# 在目标设备上
export LD_LIBRARY_PATH=/opt/taichi/c_api/lib:$LD_LIBRARY_PATH

# 运行示例
./example_arm_mali ./aot_modules/taichislam_octo

# 带参数运行
./example_arm_mali \
    --file /path/to/pointcloud.xyz \
    ./aot_modules/taichislam_octo
```

## C++ API 使用指南

### 基本用法

```cpp
#include <taichislam_aot.h>

using namespace taichislam;

int main() {
    // 1. 初始化运行时（选择后端）
    TaichiSLAMRuntime runtime(TI_ARCH_VULKAN, 0);

    // 2. 配置建图参数
    MappingConfig config;
    config.map_size_xy = 20.0f;    // 20m x 20m
    config.map_size_z = 5.0f;       // 5m height
    config.voxel_scale = 0.1f;      // 10cm voxels

    // 3. 加载AOT模块
    auto mapping = runtime.loadOctomap(
        "./aot_modules/taichislam_octo", config);

    // 4. 准备点云数据
    std::vector<float> points = { /* ... */ };
    std::vector<uint8_t> colors = { /* ... */ };

    // 5. 设置传感器位姿
    Pose pose;
    // ... 设置旋转矩阵R[9]和平移向量T[3]

    // 6. 插入点云
    mapping->insertPointCloud(
        points.data(),
        colors.data(),
        points.size() / 3,
        pose);

    // 7. 提取体素
    VoxelGrid voxels = mapping->extractVoxels(0);

    return 0;
}
```

### 完整示例

参见 `aot_runtime/examples/` 目录下的示例代码：

- `basic_octo.cpp` - 基础Octomap使用示例
- `basic_tsdf.cpp` - TSDF建图示例
- `arm_mali_example.cpp` - ARM Mali GPU部署完整示例

## 性能优化

### 1. 内存优化

```cpp
// 限制最大点云数量
config.max_disp_particles = 100000;  // 根据设备内存调整

// 合理设置地图大小
config.map_size_xy = 20.0f;  // 20m x 20m
config.map_size_z = 5.0f;    // 5m height
config.voxel_scale = 0.1f;   // 10cm resolution
```

### 2. 后端选择

```cpp
// 优先使用Vulkan（Mali-G系列）
TiArch arch = ti::is_arch_available(TI_ARCH_VULKAN)
    ? TI_ARCH_VULKAN
    : TI_ARCH_GLES;  // 回退到OpenGL ES
```

### 3. 批处理

```cpp
// 批量处理多个帧
const int BATCH_SIZE = 10;
for (int batch = 0; batch < num_frames; batch += BATCH_SIZE) {
    for (int i = 0; i < BATCH_SIZE && (batch + i) < num_frames; ++i) {
        // Insert frame
        mapping->insertPointCloud(...);
    }
    // Sync after each batch
    runtime.getNativeRuntime()->wait();
}
```

## 故障排除

### 问题1: "Target architecture is not available"

**原因**: 目标GPU不支持选定的后端

**解决**:
```bash
# 检查可用的后端
# 在目标设备上运行测试程序

# 对于Mali GPU，检查Vulkan支持
vulkaninfo | grep "deviceName"

# 如果不支持Vulkan，使用OpenGL ES
python aot_export.py --arch gles --method octo
```

### 问题2: "Failed to load AOT module"

**原因**: AOT模块与运行时架构不匹配

**解决**:
```bash
# 确保使用正确的AOT模块
# Vulkan后端 -> 使用vulkan导出的模块
# OpenGL ES后端 -> 使用gles导出的模块

# 重新导出匹配后端的模块
python aot_export.py --arch vulkan --method octo --output ./aot_modules
```

### 问题3: 内存不足

**原因**: 地图太大或分辨率太高

**解决**:
```cpp
// 减小地图大小或增加体素尺寸
MappingConfig config;
config.map_size_xy = 10.0f;  // 减小到10m
config.map_size_z = 3.0f;
config.voxel_scale = 0.2f;     // 增大到20cm
config.max_disp_particles = 50000;  // 减小点云限制
```

### 问题4: 性能低下

**原因**: 后端选择不当或批处理不合理

**解决**:
```cpp
// 1. 使用最适合的后端
TiArch arch;
if (ti::is_arch_available(TI_ARCH_VULKAN)) {
    arch = TI_ARCH_VULKAN;  // 首选Vulkan
} else if (ti::is_arch_available(TI_ARCH_GLES)) {
    arch = TI_ARCH_GLES;
}

// 2. 使用异步批处理
// 参见性能优化部分
```

## 贡献

欢迎贡献代码和报告问题！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

## 致谢

- [Taichi](https://github.com/taichi-dev/taichi) - 高性能计算框架
- [TaichiSLAM](https://github.com/yourusername/TaichiSLAM) - 基于Taichi的SLAM库

## 相关文档

- [Taichi C-API 文档](https://docs.taichi-lang.org/docs/taichi_core)
- [Vulkan for Mali GPU](https://developer.arm.com/documentation/101897/latest/)
- [OpenGL ES 3.1 规范](https://www.khronos.org/registry/OpenGL/specs/es/3.1/es_spec_3.1.pdf)
