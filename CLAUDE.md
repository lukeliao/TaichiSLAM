# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TaichiSLAM is a 3D Dense mapping backend library for SLAM based on Taichi-Lang, designed for aerial swarm robotics. It implements mapping algorithms like Octomap, TSDF (Truncated Signed Distance Function), and ESDF (Euclidean Signed Distance Function) using Taichi for GPU acceleration.

**Note: This is a ROS-free version. All ROS dependencies have been removed.**

## Architecture

### Core Mapping Modules (`taichi_slam/mapping/`)

- **`taichi_octomap.py`** - Octree-based occupancy grid mapping (OctoMap implementation)
- **`dense_tsdf.py`** - TSDF (Truncated Signed Distance Function) mapping with Marching Cubes mesh extraction
- **`dense_esdf.py`** - ESDF (Euclidean Signed Distance Function) for motion planning
- **`submap_mapping.py`** - Submap-based mapping for large-scale environments
- **`marching_cube_mesher.py`** - Marching Cubes algorithm for surface mesh generation
- **`topo_graph.py`** - Topological skeleton graph generation from TSDF

### Utilities (`taichi_slam/utils/`)

- **`pointcloud_transfer.py`** - Point cloud data conversion utilities (replaces old ROS-dependent version)
- **`visualization.py`** - Visualization helpers

## Development Commands

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add project to PYTHONPATH (optional)
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Running Tests

```bash
# Run all tests
cd tests
python test.py

# Run specific test (e.g., marching cubes)
python marching_cube_test.py

# Run topology graph generation test
python gen_topo_graph.py
```

### Performance Benchmark

```bash
# Quick benchmark (recommended)
python quick_benchmark.py --method octo --frames 30

# Test TSDF
python quick_benchmark.py --method tsdf --frames 30

# Use CUDA
python quick_benchmark.py --method octo --cuda --frames 50

# Custom parameters
python quick_benchmark.py --method octo --frames 50 --points 10000 --cuda
```

### Running Demos

```bash
# Basic demo with Octomap
python TaichiSLAM_demo.py -m octo --cuda

# TSDF demo
python TaichiSLAM_demo.py -m esdf --cuda

# CPU mode
python TaichiSLAM_demo.py -m octo

# Show all options
python TaichiSLAM_demo.py --help
```

## Key Dependencies

- **taichi >= 1.0.0** - Core DSL for high-performance computing
- **numpy** - Numerical computing
- **scipy** - Scientific computing
- **matplotlib** - Plotting
- **transformations** - 3D transformation matrices

**Note: All ROS dependencies (rospy, roscpp, std_msgs, etc.) have been removed.**

## Code Patterns

### Taichi Kernel Structure

Mapping classes use Taichi's `@ti.data_oriented` decorator and define fields with `ti.field`. Key patterns:

```python
@ti.data_oriented
class MappingClass:
    def __init__(self, ...):
        # Define Taichi fields
        self.voxels = ti.field(ti.f32, shape=(n, m, k))

    @ti.kernel
    def update_map(self, ...):
        # Parallel computation on GPU/CPU
        for i, j, k in ti.ndrange(n, m, k):
            ...
```

### Coordinate Systems

- World coordinates: metric units (meters)
- Voxel coordinates: integer indices into the voxel grid
- All mappings handle the transform between these coordinate spaces

## Important Notes

- The codebase heavily uses **Taichi** for GPU acceleration - changes to kernel functions should be tested with both CPU and CUDA backends
- **ROS integration has been removed** - this is now a pure Python library
- Large-scale mapping uses **submap** architecture to manage memory
- The Marching Cubes implementation is used for mesh extraction from TSDF

## Performance Notes

Based on benchmark tests (CPU mode, 30 frames, 5000 points/frame):

- **Octomap**: ~X ms average insertion time
- **TSDF**: ~X ms average insertion time

For best performance:
- Use CUDA mode (`--cuda` flag) for GPU acceleration
- Adjust voxel size based on your application needs
- Use appropriate map sizes to balance memory usage and coverage
