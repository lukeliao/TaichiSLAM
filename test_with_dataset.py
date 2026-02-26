#!/usr/bin/env python3
"""
使用实际3D数据集测试TaichiSLAM建图算法性能
支持从NPY文件加载点云数据
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import taichi as ti
import numpy as np
import time

def load_npy_pointcloud(filepath):
    """从NPY文件加载点云数据"""
    print(f"  加载点云数据: {filepath}")
    data = np.load(filepath)
    print(f"  数据形状: {data.shape}")
    print(f"  数据类型: {data.dtype}")

    # 解析数据格式
    if data.shape[1] >= 6:
        xyz = data[:, :3].astype(np.float32)
        rgb = data[:, 3:6].astype(np.uint8)
    elif data.shape[1] == 3:
        xyz = data[:, :3].astype(np.float32)
        rgb = np.random.randint(0, 255, (len(xyz), 3), dtype=np.uint8)
    else:
        raise ValueError(f"不支持的数据格式: 形状为 {data.shape}")

    print(f"  点云数量: {len(xyz):,}")
    print(f"  XYZ范围: X[{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}], "
          f"Y[{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}], "
          f"Z[{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}]")

    return xyz, rgb

def benchmark_with_dataset(dataset_path, method='octo', use_cuda=False):
    """使用实际数据集进行性能测试"""
    print("\n" + "="*70)
    print(f"使用数据集测试: {os.path.basename(dataset_path)}")
    print("="*70)

    # 加载数据
    xyz_data, rgb_data = load_npy_pointcloud(dataset_path)
    total_points = len(xyz_data)

    # 分批处理参数
    batch_size = 5000
    num_batches = (total_points + batch_size - 1) // batch_size

    print(f"\n处理参数:")
    print(f"  总点数: {total_points:,}")
    print(f"  批大小: {batch_size:,}")
    print(f"  批次数: {num_batches}")

    # 初始化Taichi
    print(f"\n初始化Taichi (设备: {'CUDA' if use_cuda else 'CPU'})...")
    if use_cuda:
        ti.init(arch=ti.cuda)
    else:
        ti.init(arch=ti.cpu)

    # 创建建图实例
    if method == 'octo':
        from taichi_slam.mapping.taichi_octomap import Octomap
        mapping = Octomap(
            texture_enabled=False,
            max_disp_particles=500000,
            min_occupy_thres=2,
            map_scale=[50, 20],
            voxel_scale=0.1,
            K=2
        )
        print("  使用: Octomap")
    else:  # tsdf
        from taichi_slam.mapping.dense_tsdf import DenseTSDF
        mapping = DenseTSDF(
            texture_enabled=False,
            max_disp_particles=500000,
            map_scale=[50, 20],
            voxel_scale=0.1,
            num_voxel_per_blk_axis=8
        )
        print("  使用: DenseTSDF")

    # 准备姿态（使用恒等变换）
    R = np.eye(3, dtype=np.float32)
    T = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    # 分批处理点云
    insert_times = []
    print("\n开始处理...")

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_points)

        xyz_batch = xyz_data[start_idx:end_idx]
        rgb_batch = rgb_data[start_idx:end_idx]
        n_points = len(xyz_batch)

        # 插入点云
        start = time.time()
        mapping.recast_pcl_to_map(R, T, xyz_batch, rgb_batch, n_points)
        elapsed = (time.time() - start) * 1000

        insert_times.append(elapsed)

        if (i + 1) % 10 == 0 or i == num_batches - 1:
            avg_time = np.mean(insert_times[-10:]) if len(insert_times) >= 10 else np.mean(insert_times)
            print(f"  Batch {i+1}/{num_batches}: {n_points:,} points, {elapsed:.2f} ms "
                  f"(avg: {avg_time:.2f} ms)")

    # 打印结果
    print("\n" + "="*70)
    print(f"数据集测试结果 ({method.upper()})")
    print("="*70)
    print(f"  数据集: {os.path.basename(dataset_path)}")
    print(f"  总点数: {total_points:,}")
    print(f"  批次数: {num_batches}")
    print(f"\n  性能指标:")
    print(f"    - 平均插入时间: {np.mean(insert_times):.2f} ms")
    print(f"    - 中位数:       {np.median(insert_times):.2f} ms")
    print(f"    - 标准差:       {np.std(insert_times):.2f} ms")
    print(f"    - 最小值:       {np.min(insert_times):.2f} ms")
    print(f"    - 最大值:       {np.max(insert_times):.2f} ms")
    print(f"\n  处理能力:")
    total_time = sum(insert_times) / 1000  # seconds
    throughput = total_points / total_time if total_time > 0 else 0
    print(f"    - 总处理时间:   {total_time:.2f} s")
    print(f"    - 处理速度:     {throughput:,.0f} points/sec")
    print(f"    - 实时性能:     {'✓ 满足' if np.mean(insert_times) < 1000/30 else '✗ 不足'} (30 FPS)")
    print("="*70)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='TaichiSLAM Dataset Benchmark')
    parser.add_argument('--dataset', type=str, default='data/ri_tsdf.npy',
                        help='Path to NPY dataset file')
    parser.add_argument('--method', type=str, default='octo', choices=['octo', 'tsdf'],
                        help='Mapping method')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    args = parser.parse_args()

    # 检查数据集文件
    if not os.path.exists(args.dataset):
        print(f"\n错误: 找不到数据集文件: {args.dataset}")
        print("\n可用的数据集:")
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith('.npy'):
                    print(f"  - data/{f}")
        print("\n请使用 --dataset 参数指定数据集路径")
        return

    # 运行测试
    benchmark_with_dataset(args.dataset, args.method, args.cuda)

if __name__ == "__main__":
    main()
