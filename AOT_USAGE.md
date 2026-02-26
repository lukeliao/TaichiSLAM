# TaichiSLAM AOT 使用指南

## 概述

TaichiSLAM AOT (Ahead-of-Time) 版本提供了预编译的 Octomap 和 TSDF 映射算法，可以在 ARM Mali G52 等嵌入式 GPU 上实时运行，无需 Python 环境。

## 应用场景

- **桌面机械臂实时环境感知**
- **机器人路径规划**
- **实时点云处理与 3D 重建**
- **低功耗嵌入式设备部署**

## 硬件要求

- GPU: ARM Mali G52 或其他支持 Vulkan 1.2 的 GPU
- 内存: 至少 100MB 可用内存
- 操作系统: Linux

## 算法配置

### Octomap

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 网格大小 | 128³ | 体素网格维度 |
| 体素尺寸 | 3cm | 可根据需求调整（用户要求 6cm） |
| 覆盖范围 | 3.84m³ | ±1.92m 从中心 |
| 批处理大小 | 4096 点 | 每次插入的点云数量 |
| 内存占用 | 8.00 MB | 仅占用网格 |

### TSDF (Truncated Signed Distance Function)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 网格大小 | 128³ | 体素网格维度 |
| 体素尺寸 | 3cm | 可根据需求调整 |
| 截断距离 | 9cm | 3 × 体素尺寸 |
| 覆盖范围 | 3.84m³ | ±1.92m 从中心 |
| 内存占用 | 16.00 MB | TSDF + Weight |

## AOT 模块使用方法

### 1. 导出 AOT 模块

#### Octomap (Vulkan)

```bash
python3 test_aot_export_v2.py
```

这将生成 `./aot_modules/octo_vulkan/` 目录，包含：
- `.spv` 文件: Vulkan SPIR-V 着色器
- `metadata.json`: 元数据
- `__content__`: 内容清单

#### Octomap (CPU)

```bash
python3 test_aot_export_cpu.py
```

生成 `./aot_modules/octo_cpu/` 目录，包含 `.ll` LLVM IR 文件。

#### TSDF (Vulkan)

```bash
python3 test_aot_tsdf_vulkan.py
```

生成 `./aot_modules/tsdf_vulkan/` 目录。

### 2. C++ 运行时 API

#### 基本使用流程

```cpp
#include <taichi/taichi_core.h>

// 1. 创建运行时
TiRuntime runtime = ti_create_runtime(TI_ARCH_VULKAN, 0);
if (!runtime) {
    // 错误处理
}

// 2. 加载 AOT 模块
TiAotModule module = ti_load_aot_module(runtime, "./aot_modules/octo_vulkan");

// 3. 获取 kernels
TiKernel clear_kernel = ti_get_aot_module_kernel(module, "clear_grid");
TiKernel insert_kernel = ti_get_aot_module_kernel(module, "insert_points_kernel");
TiKernel extract_kernel = ti_get_aot_module_kernel(module, "extract_occupancy_kernel");

// 4. 启动 kernel
ti_launch_kernel(runtime, clear_kernel, 0, nullptr);

// 5. 清理
ti_destroy_runtime(runtime);
```

#### 带参数的 kernel 启动

```cpp
// 设置 float 参数
float threshold = 1.0f;
TiArgument arg;
arg.type = TI_ARGUMENT_TYPE_F32;
arg.value.f32 = threshold;

ti_launch_kernel(runtime, extract_kernel, sizeof(arg), &arg);
```

### 3. 编译和运行 C++ 程序

```bash
# 编译
g++ -std=c++17 \
    -I/home/liao/.local/lib/python3.12/site-packages/taichi/_lib/c_api/include \
    your_program.cpp \
    -L/home/liao/.local/lib/python3.12/site-packages/taichi/_lib/c_api/lib \
    -ltaichi_c_api \
    -Wl,-rpath,/home/liao/.local/lib/python3.12/site-packages/taichi/_lib/c_api/lib \
    -o your_program

# 设置环境变量
export TI_LIB_DIR=/home/liao/.local/lib/python3.12/site-packages/taichi/_lib/runtime

# 运行
./your_program
```

### 4. 运行测试

项目包含几个预编译的测试程序：

```bash
# CPU 测试
export TI_LIB_DIR=/home/liao/.local/lib/python3.12/site-packages/taichi/_lib/runtime
./test_cpp_aot

# Vulkan Octomap 测试
./test_cpp_aot_vulkan

# Vulkan TSDF 测试
./test_cpp_aot_tsdf_vulkan

# 集成测试
./test_cpp_aot_integration
```

## 算法细节

### Octomap 算法

Octomap 使用占用概率网格表示 3D 空间。

#### 核心 Kernels

1. **`clear_grid()`**
   - 清空占用网格
   - 时间复杂度: O(n³), n=128
   - 复杂度: 约 2M 次操作

2. **`insert_points_kernel(points)`**
   - 插入点云到网格
   - 使用原子加法保证线程安全
   - 将世界坐标转换为网格索引

3. **`extract_occupancy_kernel(threshold)`**
   - 提取占用体素（occupancy > threshold）
   - 返回 (x, y, z, occupancy) 数组

### TSDF 算法

TSDF (Truncated Signed Distance Function) 用于表面重建。

#### 核心 Kernels

1. **`clear_tsdf()`**
   - 清空 TSDF 网格为截断距离
   - 清空权重网格为 0

2. **`update_tsdf_kernel(points)`**
   - 更新 TSDF 值
   - 使用加权平均融合
   - 截断距离: ±9cm (3 × 体素尺寸)

3. **`extract_esdf_kernel()`**
   - 提取 ESDF (Euclidean Signed Distance Field)
   - 用于路径规划

4. **`extract_surface_kernel(threshold)`**
   - 提取表面体素（|TSDF| < threshold）
   - 用于网格重建

## 部署到 ARM Mali G52

### 交叉编译

1. 使用 ARM 交叉编译器重新编译 C++ 代码
2. 确保 Vulkan 驱动在目标设备上可用
3. 将 AOT 模块目录复制到设备

### 运行

```bash
# 在 ARM 设备上
export TI_LIB_DIR=/path/to/taichi/runtime
./your_arm_program
```

## 性能优化建议

1. **调整体素尺寸**
   - 用户要求: 6cm（修改 `VOXEL_SCALE = 0.06`）
   - 权衡: 精度 vs 内存占用 vs 计算速度

2. **批处理大小**
   - 根据可用内存调整 `MAX_POINTS`
   - 更大的批处理 = 更少的 kernel 启动

3. **Vulkan vs CPU**
   - Vulkan: 更好的并行性能
   - CPU: 更好的调试支持

## 故障排除

### 常见问题

**Q: 找不到 `libtaichi_c_api.so`**
```bash
export LD_LIBRARY_PATH=/path/to/taichi/_lib/c_api/lib:$LD_LIBRARY_PATH
```

**Q: `TI_LIB_DIR` 未设置**
```bash
export TI_LIB_DIR=$(python3 -c "import taichi; import os; print(os.path.join(taichi.__path__[0], '_lib', 'runtime'))")
```

**Q: Vulkan 初始化失败**
- 检查 GPU 驱动是否正确安装
- 尝试使用 CPU 后端 (`TI_ARCH_X64`)

## 文件结构

```
TaichiSLAM/
├── aot_modules/
│   ├── octo_cpu/          # CPU 版本 Octomap
│   │   ├── clear_grid.ll
│   │   ├── insert_points_kernel.ll
│   │   ├── extract_occupancy_kernel.ll
│   │   └── metadata.json
│   ├── octo_vulkan/       # Vulkan 版本 Octomap
│   │   ├── *.spv
│   │   └── metadata.json
│   └── tsdf_vulkan/       # Vulkan 版本 TSDF
│       ├── *.spv
│       └── metadata.json
├── taichi_slam/mapping/
│   ├── aot_octomap.py     # Python 导出脚本
│   └── aot_tsdf.py        # Python 导出脚本
├── test_cpp_aot.cpp       # C++ 测试程序
├── test_aot_export_*.py    # Python 导出测试
└── AOT_USAGE.md           # 本文档
```

## 参考资料

- [Taichi C API 文档](https://docs.taichi-lang.org/)
- [OctoMap 论文](https://octomap.github.io/)
- [TSDF 重建](https://www.cs.tau.ac.il/~dcor/Graphics/cg-slides/TSDF.pdf)
