#!/usr/bin/env python3
"""
Quick benchmark for TaichiSLAM mapping algorithms - No ROS dependencies
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import taichi as ti
import numpy as np
import time
import argparse

def generate_random_pointcloud(num_points=10000, center=None, radius=5.0):
    """Generate random point cloud data"""
    if center is None:
        center = np.array([0.0, 0.0, 2.0])

    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    r = np.random.uniform(0, radius, num_points)

    x = center[0] + r * np.sin(phi) * np.cos(theta)
    y = center[1] + r * np.sin(phi) * np.sin(theta)
    z = center[2] + r * np.cos(phi)

    xyz = np.column_stack([x, y, z]).astype(np.float32)
    rgb = np.random.randint(0, 255, (num_points, 3), dtype=np.uint8)

    return xyz, rgb

def benchmark_octomap(num_frames=30, points_per_frame=5000, use_cuda=False):
    """Benchmark Octomap performance"""
    from taichi_slam.mapping.taichi_octomap import Octomap

    print("\n" + "="*60)
    print("Octomap Performance Benchmark")
    print("="*60)
    print(f"Device: {'CUDA' if use_cuda else 'CPU'}")
    print(f"Frames: {num_frames}")
    print(f"Points per frame: {points_per_frame}")

    # Initialize Taichi
    if use_cuda:
        ti.init(arch=ti.cuda)
    else:
        ti.init(arch=ti.cpu)

    # Create Octomap
    mapping = Octomap(
        texture_enabled=False,
        max_disp_particles=500000,
        min_occupy_thres=2,
        map_scale=[30, 10],
        voxel_scale=0.1,
        K=2
    )

    insert_times = []

    for i in range(num_frames):
        # Generate point cloud
        xyz, rgb = generate_random_pointcloud(
            num_points=points_per_frame,
            center=np.array([i*0.05, 0.0, 2.0])
        )

        # Set pose
        R = np.eye(3, dtype=np.float32)
        T = np.array([i*0.05, 0.0, 0.0], dtype=np.float32)

        for j in range(3):
            mapping.input_T[None][j] = T[j]
            for k in range(3):
                mapping.input_R[None][j, k] = R[j, k]

        # Insert point cloud
        start = time.time()
        mapping.recast_pcl_to_map(xyz, rgb, len(xyz))
        elapsed = (time.time() - start) * 1000

        insert_times.append(elapsed)

    print(f"\nResults:")
    print(f"  Average insertion time: {np.mean(insert_times):.2f} ms")
    print(f"  Median: {np.median(insert_times):.2f} ms")
    print(f"  Min: {np.min(insert_times):.2f} ms")
    print(f"  Max: {np.max(insert_times):.2f} ms")
    print(f"  Processing speed: {num_frames * points_per_frame / (sum(insert_times)/1000):,.0f} points/sec")

def benchmark_tsdf(num_frames=30, points_per_frame=5000, use_cuda=False):
    """Benchmark TSDF performance"""
    from taichi_slam.mapping.dense_tsdf import DenseTSDF

    print("\n" + "="*60)
    print("DenseTSDF Performance Benchmark")
    print("="*60)
    print(f"Device: {'CUDA' if use_cuda else 'CPU'}")
    print(f"Frames: {num_frames}")
    print(f"Points per frame: {points_per_frame}")

    # Initialize Taichi
    if use_cuda:
        ti.init(arch=ti.cuda)
    else:
        ti.init(arch=ti.cpu)

    # Create DenseTSDF
    mapping = DenseTSDF(
        texture_enabled=False,
        max_disp_particles=500000,
        min_occupy_thres=2,
        map_scale=[30, 10],
        voxel_scale=0.1,
        num_voxel_per_blk_axis=8
    )

    insert_times = []

    for i in range(num_frames):
        # Generate point cloud
        xyz, rgb = generate_random_pointcloud(
            num_points=points_per_frame,
            center=np.array([i*0.05, 0.0, 2.0])
        )

        # Set pose
        R = np.eye(3, dtype=np.float32)
        T = np.array([i*0.05, 0.0, 0.0], dtype=np.float32)

        for j in range(3):
            mapping.input_T[None][j] = T[j]
            for k in range(3):
                mapping.input_R[None][j, k] = R[j, k]

        # Insert point cloud
        start = time.time()
        mapping.recast_pcl_to_map(xyz, rgb, len(xyz))
        elapsed = (time.time() - start) * 1000

        insert_times.append(elapsed)

    print(f"\nResults:")
    print(f"  Average insertion time: {np.mean(insert_times):.2f} ms")
    print(f"  Median: {np.median(insert_times):.2f} ms")
    print(f"  Min: {np.min(insert_times):.2f} ms")
    print(f"  Max: {np.max(insert_times):.2f} ms")
    print(f"  Processing speed: {num_frames * points_per_frame / (sum(insert_times)/1000):,.0f} points/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TaichiSLAM Mapping Performance Benchmark')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--frames', type=int, default=30, help='Number of frames')
    parser.add_argument('--points', type=int, default=5000, help='Points per frame')
    parser.add_argument('--method', type=str, default='all', choices=['octo', 'tsdf', 'all'], help='Mapping method')
    args = parser.parse_args()

    print("="*60)
    print("TaichiSLAM Performance Benchmark (No ROS)")
    print("="*60)

    if args.method in ['octo', 'all']:
        benchmark_octomap(args.frames, args.points, args.cuda)

    if args.method in ['tsdf', 'all']:
        benchmark_tsdf(args.frames, args.points, args.cuda)

    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)
