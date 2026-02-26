#!/usr/bin/env python3
"""
Complete performance benchmark for TaichiSLAM
Tests Octomap and DenseTSDF with various configurations
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import taichi as ti
import numpy as np
import time

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

    print("\n" + "="*70)
    print("OCTOMAP PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"Configuration:")
    print(f"  Device: {'CUDA' if use_cuda else 'CPU'}")
    print(f"  Frames: {num_frames}")
    print(f"  Points per frame: {points_per_frame:,}")
    print(f"  Total points: {num_frames * points_per_frame:,}")

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

    # Generate base point cloud
    xyz_base = np.random.randn(points_per_frame, 3).astype(np.float32) * 2.0
    xyz_base[:, 2] = np.abs(xyz_base[:, 2]) + 0.5
    rgb = np.random.randint(0, 255, (points_per_frame, 3), dtype=np.uint8)

    # Identity pose
    R = np.eye(3, dtype=np.float32)
    T = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    insert_times = []

    print(f"\nRunning benchmark...")
    for i in range(num_frames):
        xyz = xyz_base + np.array([i * 0.05, 0.0, 0.0], dtype=np.float32)

        start = time.time()
        mapping.recast_pcl_to_map(R, T, xyz, rgb, points_per_frame)
        elapsed = (time.time() - start) * 1000

        insert_times.append(elapsed)

        if (i + 1) % 10 == 0:
            avg_time = np.mean(insert_times[-10:])
            print(f"  Progress: {i+1}/{num_frames} frames, last 10 avg: {avg_time:.2f} ms")

    print("\n" + "="*70)
    print("OCTOMAP BENCHMARK RESULTS")
    print("="*70)
    print(f"  Average insertion time: {np.mean(insert_times):.2f} ms")
    print(f"  Median:               {np.median(insert_times):.2f} ms")
    print(f"  Standard deviation:   {np.std(insert_times):.2f} ms")
    print(f"  Min:                  {np.min(insert_times):.2f} ms")
    print(f"  Max:                  {np.max(insert_times):.2f} ms")
    print(f"\n  Processing speed:     {num_frames * points_per_frame / (sum(insert_times)/1000):,.0f} points/sec")
    print(f"  Real-time capability: {'YES ✓' if np.mean(insert_times) < 1000/30 else 'NO ✗'} (30 FPS target)")
    print("="*70)

def benchmark_tsdf(num_frames=30, points_per_frame=5000, use_cuda=False):
    """Benchmark TSDF performance"""
    from taichi_slam.mapping.dense_tsdf import DenseTSDF

    print("\n" + "="*70)
    print("DENSE TSDF PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"Configuration:")
    print(f"  Device: {'CUDA' if use_cuda else 'CPU'}")
    print(f"  Frames: {num_frames}")
    print(f"  Points per frame: {points_per_frame:,}")
    print(f"  Total points: {num_frames * points_per_frame:,}")

    # Initialize Taichi
    if use_cuda:
        ti.init(arch=ti.cuda)
    else:
        ti.init(arch=ti.cpu)

    # Create DenseTSDF
    mapping = DenseTSDF(
        texture_enabled=False,
        max_disp_particles=500000,
        map_scale=[30, 10],
        voxel_scale=0.1,
        num_voxel_per_blk_axis=8
    )

    # Generate base point cloud
    xyz_base = np.random.randn(points_per_frame, 3).astype(np.float32) * 2.0
    xyz_base[:, 2] = np.abs(xyz_base[:, 2]) + 0.5
    rgb = np.random.randint(0, 255, (points_per_frame, 3), dtype=np.uint8)

    # Identity pose
    R = np.eye(3, dtype=np.float32)
    T = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    insert_times = []

    print(f"\nRunning benchmark...")
    for i in range(num_frames):
        xyz = xyz_base + np.array([i * 0.05, 0.0, 0.0], dtype=np.float32)

        start = time.time()
        mapping.recast_pcl_to_map(R, T, xyz, rgb, points_per_frame)
        elapsed = (time.time() - start) * 1000

        insert_times.append(elapsed)

        if (i + 1) % 10 == 0:
            avg_time = np.mean(insert_times[-10:])
            print(f"  Progress: {i+1}/{num_frames} frames, last 10 avg: {avg_time:.2f} ms")

    print("\n" + "="*70)
    print("DENSE TSDF BENCHMARK RESULTS")
    print("="*70)
    print(f"  Average insertion time: {np.mean(insert_times):.2f} ms")
    print(f"  Median:               {np.median(insert_times):.2f} ms")
    print(f"  Standard deviation:   {np.std(insert_times):.2f} ms")
    print(f"  Min:                  {np.min(insert_times):.2f} ms")
    print(f"  Max:                  {np.max(insert_times):.2f} ms")
    print(f"\n  Processing speed:     {num_frames * points_per_frame / (sum(insert_times)/1000):,.0f} points/sec")
    print(f"  Real-time capability: {'YES ✓' if np.mean(insert_times) < 1000/30 else 'NO ✗'} (30 FPS target)")
    print("="*70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='TaichiSLAM Performance Benchmark')
    parser.add_argument('--method', type=str, default='octo', choices=['octo', 'tsdf', 'both'],
                        help='Mapping method to benchmark')
    parser.add_argument('--frames', type=int, default=30, help='Number of frames')
    parser.add_argument('--points', type=int, default=5000, help='Points per frame')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("TAICHISLAM PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Taichi Version: {ti.__version__}")
    print(f"NumPy Version: {np.__version__}")

    if args.method == 'octo' or args.method == 'both':
        benchmark_octomap(args.frames, args.points, args.cuda)

    if args.method == 'tsdf' or args.method == 'both':
        benchmark_tsdf(args.frames, args.points, args.cuda)

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70 + "\n")
