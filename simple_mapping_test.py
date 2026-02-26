#!/usr/bin/env python3
"""
Simple mapping performance test without ROS
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import taichi as ti
import numpy as np
import time
import math

# Initialize Taichi
print("[初始化] 正在初始化 Taichi...")
ti.init(arch=ti.cpu, debug=False)
print("[初始化] Taichi 初始化完成")

# Import mapping classes
from taichi_slam.mapping.taichi_octomap import Octomap
from taichi_slam.mapping.dense_tsdf import DenseTSDF

def generate_random_pointcloud(num_points=10000, center=None, radius=5.0):
    """Generate random point cloud"""
    if center is None:
        center = np.array([0.0, 0.0, 2.0])

    # Random points in sphere
    theta = np.random.uniform(0, 2*np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)
    r = np.random.uniform(0, radius, num_points)

    x = center[0] + r * np.sin(phi) * np.cos(theta)
    y = center[1] + r * np.sin(phi) * np.sin(theta)
    z = center[2] + r * np.cos(phi)

    xyz = np.column_stack([x, y, z]).astype(np.float32)
    rgb = np.random.randint(0, 255, (num_points, 3), dtype=np.uint8)

    return xyz, rgb

def test_octomap():
    """Test Octomap performance"""
    print("\n" + "="*60)
    print("Octomap 性能测试")
    print("="*60)

    # Create Octomap instance
    print("[建图器] 初始化 Octomap...")
    mapping = Octomap(
        texture_enabled=False,
        max_disp_particles=1000000,
        min_occupy_thres=2,
        map_scale=[50, 10],
        voxel_scale=0.05,
        K=2
    )
    print("[建图器] Octomap 初始化完成")

    # Test parameters
    num_frames = 50
    points_per_frame = 10000

    print(f"\n[性能测试] 开始测试")
    print(f"  - 帧数: {num_frames}")
    print(f"  - 每帧点数: {points_per_frame}")
    print(f"  - 总点数: {num_frames * points_per_frame}")

    insert_times = []
    total_start = time.time()

    for i in range(num_frames):
        # Generate point cloud
        xyz, rgb = generate_random_pointcloud(
            num_points=points_per_frame,
            center=np.array([i*0.1, 0.0, 2.0])
        )

        # Set pose (identity)
        R = np.eye(3, dtype=np.float32)
        T = np.array([i*0.1, 0.0, 0.0], dtype=np.float32)

        for j in range(3):
            mapping.input_T[None][j] = T[j]
            for k in range(3):
                mapping.input_R[None][j, k] = R[j, k]

        # Insert point cloud
        insert_start = time.time()
        mapping.recast_pcl_to_map(xyz, rgb, len(xyz))
        insert_time = (time.time() - insert_start) * 1000  # ms

        insert_times.append(insert_time)

        if (i + 1) % 10 == 0:
            avg_time = np.mean(insert_times[-10:])
            print(f"  帧 {i+1}/{num_frames}: 平均插入时间 = {avg_time:.2f} ms")

    total_time = (time.time() - total_start) * 1000

    print(f"\n[测试结果] Octomap 性能测试完成")
    print(f"  - 总时间: {total_time:.2f} ms ({total_time/1000:.2f} s)")
    print(f"  - 平均每帧: {np.mean(insert_times):.2f} ms")
    print(f"  - 中位数: {np.median(insert_times):.2f} ms")
    print(f"  - 最小值: {np.min(insert_times):.2f} ms")
    print(f"  - 最大值: {np.max(insert_times):.2f} ms")

    # Calculate processing speed
    total_points = num_frames * points_per_frame
    processing_speed = total_points / (total_time / 1000)
    print(f"  - 处理速度: {processing_speed:,.0f} 点/秒")

    return {
        'insert_times': insert_times,
        'total_time': total_time,
        'avg_time': np.mean(insert_times),
        'processing_speed': processing_speed
    }

def test_tsdf():
    """Test TSDF performance"""
    print("\n" + "="*60)
    print("DenseTSDF 性能测试")
    print("="*60)

    # Create DenseTSDF instance
    print("[建图器] 初始化 DenseTSDF...")
    mapping = DenseTSDF(
        texture_enabled=False,
        max_disp_particles=1000000,
        min_occupy_thres=2,
        map_scale=[50, 10],
        voxel_scale=0.05,
        num_voxel_per_blk_axis=16
    )
    print("[建图器] DenseTSDF 初始化完成")

    # Test parameters
    num_frames = 50
    points_per_frame = 10000

    print(f"\n[性能测试] 开始测试")
    print(f"  - 帧数: {num_frames}")
    print(f"  - 每帧点数: {points_per_frame}")
    print(f"  - 总点数: {num_frames * points_per_frame}")

    insert_times = []
    total_start = time.time()

    for i in range(num_frames):
        # Generate point cloud
        xyz, rgb = generate_random_pointcloud(
            num_points=points_per_frame,
            center=np.array([i*0.1, 0.0, 2.0])
        )

        # Set pose (identity)
        R = np.eye(3, dtype=np.float32)
        T = np.array([i*0.1, 0.0, 0.0], dtype=np.float32)

        for j in range(3):
            mapping.input_T[None][j] = T[j]
            for k in range(3):
                mapping.input_R[None][j, k] = R[j, k]

        # Insert point cloud
        insert_start = time.time()
        mapping.recast_pcl_to_map(xyz, rgb, len(xyz))
        insert_time = (time.time() - insert_start) * 1000  # ms

        insert_times.append(insert_time)

        if (i + 1) % 10 == 0:
            avg_time = np.mean(insert_times[-10:])
            print(f"  帧 {i+1}/{num_frames}: 平均插入时间 = {avg_time:.2f} ms")

    total_time = (time.time() - total_start) * 1000

    print(f"\n[测试结果] DenseTSDF 性能测试完成")
    print(f"  - 总时间: {total_time:.2f} ms ({total_time/1000:.2f} s)")
    print(f"  - 平均每帧: {np.mean(insert_times):.2f} ms")
    print(f"  - 中位数: {np.median(insert_times):.2f} ms")
    print(f"  - 最小值: {np.min(insert_times):.2f} ms")
    print(f"  - 最大值: {np.max(insert_times):.2f} ms")

    # Calculate processing speed
    total_points = num_frames * points_per_frame
    processing_speed = total_points / (total_time / 1000)
    print(f"  - 处理速度: {processing_speed:,.0f} 点/秒")

    return {
        'insert_times': insert_times,
        'total_time': total_time,
        'avg_time': np.mean(insert_times),
        'processing_speed': processing_speed
    }

def main():
    print("\n" + "="*60)
    print("TaichiSLAM 建图算法性能测试")
    print("="*60)

    # Run Octomap test
    try:
        octo_results = test_octomap()
    except Exception as e:
        print(f"\n[错误] Octomap 测试失败: {e}")
        import traceback
        traceback.print_exc()

    # Run TSDF test
    try:
        tsdf_results = test_tsdf()
    except Exception as e:
        print(f"\n[错误] DenseTSDF 测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
