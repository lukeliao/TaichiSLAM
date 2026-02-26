"""
AOT-friendly TSDF (Truncated Signed Distance Function) for real-time surface reconstruction.

This implementation uses module-level kernels for AOT compatibility.

Configuration:
    - Grid: 128³ dense
    - Resolution: 3cm per voxel
    - Coverage: 3.84m³ (±1.92m from center)
    - Truncation distance: 3 * voxel_scale
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
TRUNC_DIST = VOXEL_SCALE * 3  # Truncation distance


# ============================================================================
# Fields (created at module level for AOT)
# ============================================================================
# TSDF values: negative = behind surface, positive = in front, 0 = on surface
tsdf_grid = ti.field(dtype=ti.f32, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE))
# Weight for TSDF fusion
weight_grid = ti.field(dtype=ti.f32, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE))

# Input buffers
point_buffer = ti.Vector.field(n=3, dtype=ti.f32, shape=(MAX_POINTS,))
num_points = ti.field(dtype=ti.i32, shape=())

# ESDF output buffer
esdf_grid = ti.field(dtype=ti.f32, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE))

# Output buffer for surface voxels
max_outputs = GRID_SIZE ** 3
output_buffer = ti.Vector.field(n=4, dtype=ti.f32, shape=(max_outputs,))  # x, y, z, tsdf
num_output = ti.field(dtype=ti.i32, shape=())


# ============================================================================
# Kernels (module-level for AOT compatibility)
# ============================================================================

@ti.kernel
def clear_tsdf():
    """Clear the TSDF grid."""
    for i, j, k in ti.ndrange(GRID_SIZE, GRID_SIZE, GRID_SIZE):
        tsdf_grid[i, j, k] = TRUNC_DIST  # Initialize to "in front" of any surface
        weight_grid[i, j, k] = 0.0


@ti.kernel
def update_tsdf_kernel(points: ti.types.ndarray(ndim=2)):
    """Update TSDF from point cloud (simplified ray marching)."""
    for idx in range(num_points[None]):
        x = points[idx, 0]
        y = points[idx, 1]
        z = points[idx, 2]

        # Convert to grid coordinates
        gx = int((x - WORLD_MIN) / VOXEL_SCALE)
        gy = int((y - WORLD_MIN) / VOXEL_SCALE)
        gz = int((z - WORLD_MIN) / VOXEL_SCALE)

        # Bounds check
        if 0 <= gx < GRID_SIZE and 0 <= gy < GRID_SIZE and 0 <= gz < GRID_SIZE:
            # Set TSDF to 0 (on surface)
            old_tsdf = tsdf_grid[gx, gy, gz]
            old_weight = weight_grid[gx, gy, gz]

            # Simple fusion: set to 0 for surface points
            new_weight = old_weight + 1.0
            new_tsdf = (old_tsdf * old_weight + 0.0 * 1.0) / new_weight  # 0 = on surface

            tsdf_grid[gx, gy, gz] = new_tsdf
            weight_grid[gx, gy, gz] = new_weight


@ti.kernel
def extract_esdf_kernel():
    """Extract ESDF from TSDF (approximate)."""
    for i, j, k in ti.ndrange(GRID_SIZE, GRID_SIZE, GRID_SIZE):
        tsdf = tsdf_grid[i, j, k]
        weight = weight_grid[i, j, k]

        if weight < 0.1:  # Unknown voxel
            esdf_grid[i, j, k] = -1.0  # Mark as unknown
        else:
            # Approximate Euclidean distance from TSDF
            if tsdf < 0:  # Inside surface
                esdf_grid[i, j, k] = -tsdf  # Distance to surface
            else:  # Outside surface
                esdf_grid[i, j, k] = tsdf


@ti.kernel
def extract_surface_kernel(threshold: ti.f32):
    """Extract voxels near the surface (where TSDF ≈ 0)."""
    num_output[None] = 0

    for i, j, k in ti.ndrange(GRID_SIZE, GRID_SIZE, GRID_SIZE):
        tsdf = tsdf_grid[i, j, k]
        weight = weight_grid[i, j, k]

        # Extract voxels near surface
        if weight > 0.1 and ti.abs(tsdf) < threshold:
            x = WORLD_MIN + (i + 0.5) * VOXEL_SCALE
            y = WORLD_MIN + (j + 0.5) * VOXEL_SCALE
            z = WORLD_MIN + (k + 0.5) * VOXEL_SCALE

            idx = ti.atomic_add(num_output[None], 1)

            if idx < GRID_SIZE ** 3:
                output_buffer[idx] = ti.Vector([x, y, z, tsdf])


def precompile_kernels():
    """Pre-compile all kernels to materialize them for AOT."""
    sample_points = np.zeros((1, 3), dtype=np.float32)
    points_dev = ti.ndarray(shape=(1, 3), dtype=ti.f32)
    points_dev.from_numpy(sample_points)

    num_points[None] = 0
    clear_tsdf()
    update_tsdf_kernel(points_dev)
    extract_esdf_kernel()
    extract_surface_kernel(1.0)


def export_aot_module(output_dir: str, arch=ti.vulkan):
    """
    Export AOT TSDF module for target architecture.

    Args:
        output_dir: Directory to save the AOT module files
        arch: Target architecture (default: ti.vulkan for ARM Mali)
    """
    # Pre-compile kernels first
    precompile_kernels()

    # Create AOT module
    module = ti.aot.Module(arch)

    # Create sample input for template args
    sample_points = np.zeros((1, 3), dtype=np.float32)
    points_dev = ti.ndarray(shape=(1, 3), dtype=ti.f32)
    points_dev.from_numpy(sample_points)

    # Add kernels
    module.add_kernel(clear_tsdf)
    module.add_kernel(update_tsdf_kernel, template_args={"points": points_dev})
    module.add_kernel(extract_esdf_kernel)
    module.add_kernel(extract_surface_kernel)

    # Add fields
    module.add_field("tsdf_grid", tsdf_grid)
    module.add_field("weight_grid", weight_grid)
    module.add_field("esdf_grid", esdf_grid)
    module.add_field("point_buffer", point_buffer)
    module.add_field("num_points", num_points)
    module.add_field("output_buffer", output_buffer)
    module.add_field("num_output", num_output)

    # Create output directory and save module
    os.makedirs(output_dir, exist_ok=True)
    module.save(output_dir)

    # Print summary
    print(f"=" * 60)
    print(f"AOT TSDF Module Exported Successfully")
    print(f"=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Target architecture: {arch}")
    print(f"Grid configuration: {GRID_SIZE}³ @ {VOXEL_SCALE*100:.0f}cm")
    print(f"Coverage: {WORLD_SIZE:.2f}m³")
    print(f"Truncation distance: {TRUNC_DIST:.3f}m")
    print(f"Memory: {GRID_SIZE**3 * 4 * 2 / 1024 / 1024:.2f} MB (tsdf+weight)")
    print(f"=" * 60)


if __name__ == "__main__":
    # Export for Vulkan (ARM Mali G52)
    export_aot_module("./aot_modules/tsdf_vulkan", arch=ti.vulkan)
