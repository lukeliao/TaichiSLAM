# TaichiSLAM AOT Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement TaichiSLAM AOT (Ahead-of-Time) export and C++ runtime for real-time robotic arm environment perception on ARM Mali G52.

**Architecture:** Minimal Viable AOT approach with 128³ dense grid @ 3cm resolution (3.84m³ coverage, 8.4MB memory). Three core kernels (clear_grid, insert_points, extract_occupancy) pre-compiled to SPIR-V for Vulkan 1.2 backend.

**Tech Stack:** Python 3.8+, Taichi 1.7+, C++17, CMake 3.14+, Vulkan Loader, ARM Mali G52

---

## Phase 1: Python AOT Export (Days 1-2)

### Task 1: Create AOT-Friendly Octomap Class

**Files:**
- Create: `taichi_slam/mapping/aot_octomap.py`

**Step 1: Write class structure with 128³ dense grid**

```python
import taichi as ti
import numpy as np

@ti.data_oriented
class AOTOctomap:
    """AOT-friendly Octomap for real-time robotic arm planning.

    Configuration:
    - Grid: 128³ dense (8.4MB)
    - Resolution: 3cm per voxel
    - Coverage: 3.84m³ (±1.92m from center)
    - Batch: 4096 points
    """

    def __init__(self, grid_size=128, voxel_scale=0.06, max_points=4096):
        self.grid_size = grid_size  # 128
        self.voxel_scale = voxel_scale  # 3cm
        self.max_points = max_points  # 4096

        # World bounds: ±(128 * 0.03 / 2) = ±1.92m
        self.world_size = grid_size * voxel_scale  # 3.84m
        self.world_min = -self.world_size / 2  # -1.92m

        # Dense occupancy grid: 128³ float32 = 8.4MB
        self.occupancy = ti.field(ti.f32, shape=(grid_size, grid_size, grid_size))

        # Point cloud input buffer (device memory)
        self.point_buffer = ti.Vector.field(3, ti.f32, shape=(max_points,))

        # Number of valid points in current batch
        self.num_points = ti.field(ti.i32, shape=())

        # Output buffer for extracted occupied voxels
        # Each voxel: (x, y, z, occupancy_value)
        max_outputs = grid_size ** 3  # ~2M voxels max
        self.output_buffer = ti.Vector.field(4, ti.f32, shape=(max_outputs,))
        self.num_output = ti.field(ti.i32, shape=())

    @ti.kernel
    def clear_grid(self):
        """Clear the occupancy grid to zero."""
        for i, j, k in ti.ndrange(self.grid_size, self.grid_size, self.grid_size):
            self.occupancy[i, j, k] = 0.0

    @ti.kernel
    def insert_points_kernel(self, points: ti.types.ndarray(ndim=2)):
        """Insert points from numpy array into occupancy grid."""
        for idx in range(self.num_points[None]):
            x = points[idx, 0]
            y = points[idx, 1]
            z = points[idx, 2]

            # World to grid coordinate transformation
            gx = int((x - self.world_min) / self.voxel_scale)
            gy = int((y - self.world_min) / self.voxel_scale)
            gz = int((z - self.world_min) / self.voxel_scale)

            # Bounds check and atomic increment
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size and 0 <= gz < self.grid_size:
                ti.atomic_add(self.occupancy[gx, gy, gz], 1.0)

    @ti.kernel
    def extract_occupancy_kernel(self, threshold: ti.f32):
        """Extract voxels with occupancy > threshold."""
        self.num_output[None] = 0

        for i, j, k in ti.ndrange(self.grid_size, self.grid_size, self.grid_size):
            occ = self.occupancy[i, j, k]
            if occ > threshold:
                x = self.world_min + (i + 0.5) * self.voxel_scale
                y = self.world_min + (j + 0.5) * self.voxel_scale
                z = self.world_min + (k + 0.5) * self.voxel_scale

                idx = ti.atomic_add(self.num_output[None], 1)
                if idx < self.grid_size ** 3:
                    self.output_buffer[idx] = ti.Vector([x, y, z, occ])


def export_aot_module(output_dir: str, arch=ti.vulkan):
    """Export AOT module for target architecture."""
    ti.init(arch=arch)

    octomap = AOTOctomap(grid_size=128, voxel_scale=0.03, max_points=4096)

    module = ti.aot.Module(arch)

    # Add kernels
    module.add_kernel(octomap.clear_grid, template_args={})
    module.add_kernel(
        octomap.insert_points_kernel,
        template_args={"points": ti.types.ndarray(ndim=2, dtype=ti.f32)}
    )
    module.add_kernel(
        octomap.extract_occupancy_kernel,
        template_args={"threshold": ti.f32}
    )

    # Add fields
    module.add_field("occupancy", octomap.occupancy)
    module.add_field("point_buffer", octomap.point_buffer)
    module.add_field("num_points", octomap.num_points)
    module.add_field("output_buffer", octomap.output_buffer)
    module.add_field("num_output", octomap.num_output)

    import os
    os.makedirs(output_dir, exist_ok=True)
    module.save(output_dir)

    print(f"AOT module exported: {output_dir}")


if __name__ == "__main__":
    export_aot_module("./aot_modules/vulkan", arch=ti.vulkan)
```

设计文档已完成！包含了完整的架构设计、数据流图、内存布局、内核规范、C++ API 设计和实施路线图。</think>完成！我已经创建了完整的 **TaichiSLAM AOT Implementation Design** 文档，位于 `docs/plans/2025-02-25-aot-implementation-design.md`（约 800 行详细设计，涵盖完整架构、数据流、内存布局、3 个核心内核的详细实现、C++ API 设计以及 5 天的实施路线图）。文档包括 128³ 密集网格配置、3 个 AOT 内核、C++ API 设计，以及针对 ARM Mali G52 和桌面机械臂实时环境感知的使用场景。现在我可以继续执行实施阶段，准备开始 Phase 1 的 Python AOT Export 实现。是否需要继续执行 Phase 1？或者需要先对设计文档进行任何调整？或者是否应该先用 `superpowers:executing-plans` 来执行实施计划？让我先检查一下当前的任务状态，看看是否有任何阻塞。</think>我已经完成了完整的 **TaichiSLAM AOT Implementation Design** 文档，包含详细架构设计、内核实现、C++ API 和 5 天实施路线图。接下来我将开始实施 Phase 1: Python AOT Export，首先创建 AOT 友好的 Octomap 类。<|tool_calls_section_begin|><|tool_call_begin|>functions.TaskCreate:322<|tool_call_argument_begin|>{