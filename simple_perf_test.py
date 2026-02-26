#!/usr/bin/env python3
"""
Simple performance test for TaichiSLAM mapping algorithms
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import taichi as ti
import numpy as np
import time

print("=" * 60)
print("TaichiSLAM Simple Performance Test")
print("=" * 60)

# Initialize Taichi
print("\n[1/4] Initializing Taichi...")
ti.init(arch=ti.cpu)
print("      Taichi initialized (CPU mode)")

# Import mapping classes
print("\n[2/4] Importing mapping modules...")
from taichi_slam.mapping.taichi_octomap import Octomap
print("      Octomap imported")

# Create Octomap instance
print("\n[3/4] Creating Octomap instance...")
mapping = Octomap(
    texture_enabled=False,
    max_disp_particles=100000,
    min_occupy_thres=2,
    map_scale=[20, 5],
    voxel_scale=0.1,
    K=2
)
print("      Octomap created successfully")

# Generate test data
print("\n[4/4] Running performance test...")
num_frames = 20
num_points = 5000

# Generate base point cloud
xyz_base = np.random.randn(num_points, 3).astype(np.float32) * 2.0
xyz_base[:, 2] = np.abs(xyz_base[:, 2]) + 0.5
rgb = np.random.randint(0, 255, (num_points, 3), dtype=np.uint8)

# Identity pose
R = np.eye(3, dtype=np.float32)
T = np.array([0.0, 0.0, 0.0], dtype=np.float32)

insert_times = []

for i in range(num_frames):
    # Move point cloud slightly each frame
    xyz = xyz_base + np.array([i * 0.05, 0.0, 0.0], dtype=np.float32)

    # Time the insertion
    start = time.time()
    mapping.recast_pcl_to_map(R, T, xyz, rgb, num_points)
    elapsed = (time.time() - start) * 1000  # ms

    insert_times.append(elapsed)

    if (i + 1) % 5 == 0:
        avg_time = np.mean(insert_times[-5:])
        print(f"      Frame {i+1}/{num_frames}: {avg_time:.2f} ms avg")

# Print results
print("\n" + "=" * 60)
print("Benchmark Results (Octomap)")
print("=" * 60)
print(f"  Configuration:")
print(f"    - Device: CPU")
print(f"    - Frames: {num_frames}")
print(f"    - Points per frame: {num_points}")
print(f"  Performance:")
print(f"    - Average insertion: {np.mean(insert_times):.2f} ms")
print(f"    - Median: {np.median(insert_times):.2f} ms")
print(f"    - Min: {np.min(insert_times):.2f} ms")
print(f"    - Max: {np.max(insert_times):.2f} ms")
print(f"    - Throughput: {num_frames * num_points / (sum(insert_times)/1000):,.0f} points/sec")
print("=" * 60)
