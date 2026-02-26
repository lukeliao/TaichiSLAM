# TaichiSLAM

This project is a 3D Dense mapping backend library of SLAM based on Taichi-Lang, designed for the aerial swarm.

![](./docs/marchingcubes.png)

[Demo video](https://www.bilibili.com/video/BV1yu41197Q4/)

## Intro

[Taichi](https://github.com/taichi-dev/taichi) is an efficient domain-specific language (DSL) designed for computer graphics (CG), which can be adopted for high-performance computing on mobile devices.
Thanks to the connection between CG and robotics, we can adopt this powerful tool to accelerate the development of robotics algorithms.

In this project, I am trying to take advantages of Taichi, including parallel optimization, sparse computing, advanced data structures and CUDA acceleration.
The original purpose of this project is to reproduce dense mapping papers, including [Octomap](https://octomap.github.io/), [Voxblox](https://github.com/ethz-asl/voxblox), [Voxgraph](https://github.com/ethz-asl/voxgraph) etc.

Note: This project is only backend of 3d dense mapping. For full SLAM features including real-time state estimation, pose graph optimization, depth generation, please take a look on [VINS](https://github.com/HKUST-Aerial-Robotics/VINS-Fisheye) and my fisheye fork of [VINS](https://github.com/xuhao1/VINS-Fisheye).

## Demos

Octomap/Occupy[1] map at different accuacy:

<img src="./docs/octomap1.png" alt="drawing" style="width:400px;"/>
<img src="./docs/octomap2.png" alt="drawing" style="width:400px;"/>
<img src="./docs/octomap3.png" alt="drawing" style="width:400px;"/>

Truncated signed distance function (TSDF) [2]:
Surface reconstruct by TSDF (not refined)
![](./docs/TSDF_reconstruct.png)
Occupy map and slice of original TSDF
![](./docs/TSDF.png)

## Installation

```bash
pip install -r requirements.txt
```

Requirements:
- Python >= 3.8
- Taichi >= 1.0.0
- NumPy
- SciPy
- Matplotlib

## Quick Start

### Basic Mapping Demo

```bash
# Octomap demo
python TaichiSLAM_demo.py -m octo --cuda

# TSDF demo
python TaichiSLAM_demo.py -m esdf --cuda

# CPU mode
python TaichiSLAM_demo.py -m octo
```

### Performance Benchmark

```bash
# Run performance benchmark
python quick_benchmark.py --method octo --frames 30

# Test TSDF
python quick_benchmark.py --method tsdf --frames 30

# Use CUDA
python quick_benchmark.py --method octo --cuda --frames 50
```

### Topology Graph Generation

```bash
# Generate topological skeleton graph from TSDF
python tests/gen_topo_graph.py
```

## Generation topology skeleton graph [4]

This demo generate [topological skeleton graph](https://arxiv.org/abs/2208.04248) from TSDF.
Nvidia GPU is recommend for better performance.

```
pip install -r requirements.txt
python tests/gen_topo_graph.py
```
This shows the polyhedron
![](./docs/topo_graph_gen.png)

De-select the mesh in the options to show the skeleton
![](./docs/topo_graph_gen_skeleton.png)

## Bundle Adjustment (In development)
![](./docs/gradient_descent_ba.gif)

## Architecture

### Core Mapping Modules (`taichi_slam/mapping/`)

- **`taichi_octomap.py`** - Octree-based occupancy grid mapping (OctoMap implementation)
- **`dense_tsdf.py`** - TSDF (Truncated Signed Distance Function) mapping with Marching Cubes mesh extraction
- **`dense_esdf.py`** - ESDF (Euclidean Signed Distance Function) for motion planning
- **`submap_mapping.py`** - Submap-based mapping for large-scale environments
- **`marching_cube_mesher.py`** - Marching Cubes algorithm for surface mesh generation
- **`topo_graph.py`** - Topological skeleton graph generation from TSDF

### Utilities (`taichi_slam/utils/`)

- **`pointcloud_transfer.py`** - Point cloud data conversion utilities (replaces ROS-dependent version)
- **`visualization.py`** - Visualization helpers

## Roadmap

### Paper Reproduction
- [x] Octomap
- [x] Voxblox
- [ ] Voxgraph

### Mapping
- [x] Octotree occupancy map [1]
- [x] TSDF [2]
- [x] Incremental ESDF [2]
- [x] Submap [3]
  - [ ] Octomap
  - [x] TSDF
  - [ ] ESDF
- [x] Topology skeleton graph generation [4]
  - [x] TSDF
  - [ ] Pointcloud/Octomap
- [ ] Loop Detection

### MISC
- [x] 3D occupancy map visualizer
- [x] 3D TSDF/ESDF map visualizer
- [ ] Export to C/C++
- [ ] Benchmark

## Known Issues
Memory issue on ESDF generation, debugging...

## References

[1] Hornung, Armin, et al. "OctoMap: An efficient probabilistic 3D mapping framework based on octrees." Autonomous robots 34.3 (2013): 189-206.

[2] Oleynikova, Helen, et al. "Voxblox: Incremental 3d euclidean signed distance fields for on-board mav planning." 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2017.

[3] Reijgwart, Victor, et al. "Voxgraph: Globally consistent, volumetric mapping using signed distance function submaps." IEEE Robotics and Automation Letters 5.1 (2019): 227-234.

[4] Chen, Xinyi, et al. "Fast 3D Sparse Topological Skeleton Graph Generation for Mobile Robot Global Planning." arXiv preprint arXiv:2208.04248 (2022).

## LICENSE
LGPL
