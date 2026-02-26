"""
AOT-friendly Octomap for real-time robotic arm planning.

This implementation uses module-level kernels for AOT compatibility.
All kernels must be pre-compiled before adding to the AOT module.

Configuration:
    - Grid: 128³ dense (8.4MB)
    - Resolution: 3cm per voxel
    - Coverage: 3.84m³ (±1.92m from center)
    - Batch: 4096 points
"""

import taichi as ti
import numpy as np
import os


# ============================================================================
# Global Configuration
# ============================================================================
GRID_SIZE = 128
VOXEL_SCALE = 0.03  # 3cm
MAX_POINTS = 4096
WORLD_SIZE = GRID_SIZE * VOXEL_SCALE
WORLD_MIN = -WORLD_SIZE / 2


# ============================================================================
# Fields (created at module level for AOT)
# ============================================================================
occupancy = ti.field(dtype=ti.f32, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE))
point_buffer = ti.Vector.field(n=3, dtype=ti.f32, shape=(MAX_POINTS,))
num_points = ti.field(dtype=ti.i32, shape=())
max_outputs = GRID_SIZE ** 3
output_buffer = ti.Vector.field(n=4, dtype=ti.f32, shape=(max_outputs,))
num_output = ti.field(dtype=ti.i32, shape=())


# ============================================================================
# Kernels (module-level for AOT compatibility)
# ============================================================================

@ti.kernel
def clear_grid():
    """Clear the occupancy grid to zero."""
    for i, j, k in ti.ndrange(GRID_SIZE, GRID_SIZE, GRID_SIZE):
        occupancy[i, j, k] = 0.0


@ti.kernel
def insert_points_kernel(points: ti.types.ndarray(ndim=2)):
    """Insert points from numpy array into occupancy grid."""
    for idx in range(num_points[None]):
        x = points[idx, 0]
        y = points[idx, 1]
        z = points[idx, 2]

        gx = int((x - WORLD_MIN) / VOXEL_SCALE)
        gy = int((y - WORLD_MIN) / VOXEL_SCALE)
        gz = int((z - WORLD_MIN) / VOXEL_SCALE)

        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE and 0 <= gz < GRID_SIZE:
            ti.atomic_add(occupancy[gx, gy, gz], 1.0)


@ti.kernel
def extract_occupancy_kernel(threshold: ti.f32):
    """Extract voxels with occupancy value greater than threshold."""
    num_output[None] = 0
    for i, j, k in ti.ndrange(GRID_SIZE, GRID_SIZE, GRID_SIZE):
        occ = occupancy[i, j, k]
        if occ > threshold:
            x = WORLD_MIN + (i + 0.5) * VOXEL_SCALE
            y = WORLD_MIN + (j + 0.5) * VOXEL_SCALE
            z = WORLD_MIN + (k + 0.5) * VOXEL_SCALE
            idx = ti.atomic_add(num_output[None], 1)
            if idx < GRID_SIZE ** 3:
                output_buffer[idx] = ti.Vector([x, y, z, occ])


def precompile_kernels():
    """Pre-compile all kernels to materialize them for AOT."""
    sample_points = np.zeros((1, 3), dtype=np.float32)
    sample_points_dev = ti.ndarray(shape=(1, 3), dtype=ti.f32)
    sample_points_dev.from_numpy(sample_points)

    num_points[None] = 0
    clear_grid()
    insert_points_kernel(sample_points_dev)
    extract_occupancy_kernel(1.0)


def export_aot_module(output_dir: str, arch=ti.vulkan):
    """
    Export AOT module for target architecture.

    Args:
        output_dir: Directory to save the AOT module files
        arch: Target architecture (default: ti.vulkan for ARM Mali)

    Example:
        # Export for ARM Mali G52 (Vulkan)
        export_aot_module("./aot_modules/vulkan", ti.vulkan)
    """
    # Pre-compile kernels first
    precompile_kernels()

    # Create AOT module
    module = ti.aot.Module(arch)

    # Create sample input for template args
    sample_points = np.zeros((1, 3), dtype=np.float32)
    sample_points_dev = ti.ndarray(shape=(1, 3), dtype=ti.f32)
    sample_points_dev.from_numpy(sample_points)

    # Add kernels with template arguments
    module.add_kernel(clear_grid)
    module.add_kernel(insert_points_kernel, template_args={"points": sample_points_dev})
    module.add_kernel(extract_occupancy_kernel)

    # Add fields for host-device data exchange
    module.add_field("occupancy", occupancy)
    module.add_field("point_buffer", point_buffer)
    module.add_field("num_points", num_points)
    module.add_field("output_buffer", output_buffer)
    module.add_field("num_output", num_output)

    # Create output directory and save module
    os.makedirs(output_dir, exist_ok=True)
    module.save(output_dir)

    # Print summary
    print(f"=" * 60)
    print(f"AOT Octomap Module Exported Successfully")
    print(f"=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Target architecture: {arch}")
    print(f"Grid configuration: {GRID_SIZE}³ @ {VOXEL_SCALE*100:.0f}cm")
    print(f"Coverage: {WORLD_SIZE:.2f}m³")
    print(f"Memory: {GRID_SIZE**3 * 4 / 1024 / 1024:.2f} MB")
    print(f"Batch size: {MAX_POINTS} points")
    print(f"=" * 60)


if __name__ == "__main__":
    # Export for Vulkan (ARM Mali G52)
    export_aot_module("./aot_modules/octo_vulkan", arch=ti.vulkan)
