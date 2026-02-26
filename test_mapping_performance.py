#!/usr/bin/env python3
"""
纯净建图算法性能测试 - 无ROS依赖
测试 Octomap 和 TSDF 建图性能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import taichi as ti
import numpy as np
import time
import argparse

from taichi_slam.mapping.taichi_octomap import Octomap
from taichi_slam.mapping.dense_tsdf import DenseTSDF


class MappingPerformanceTest:
    def __init__(self, method='octo', cuda=False, voxel_size=0.05, map_size_xy=50, map_size_z=10):
        self.method = method
        self.cuda = cuda
        self.voxel_size = voxel_size
        self.map_size_xy = map_size_xy
        self.map_size_z = map_size_z

        # 初始化Taichi
        if cuda:
            ti.init(arch=ti.cuda, debug=False)
            print(f"[初始化] 使用CUDA加速")
        else:
            ti.init(arch=ti.cpu, debug=False)
            print(f"[初始化] 使用CPU模式")

        # 创建建图实例
        if method == 'octo':
            self.mapping = Octomap(
                texture_enabled=False,
                max_disp_particles=1000000,
                min_occupy_thres=2,
                map_scale=[map_size_xy, map_size_z],
                voxel_scale=voxel_size,
                K=2
            )
            print(f"[建图器] Octomap 初始化完成")
        elif method == 'tsdf':
            self.mapping = DenseTSDF(
                texture_enabled=False,
                max_disp_particles=1000000,
                min_occupy_thres=2,
                map_scale=[map_size_xy, map_size_z],
                voxel_scale=voxel_size,
                num_voxel_per_blk_axis=16
            )
            print(f"[建图器] DenseTSDF 初始化完成")
        else:
            raise ValueError(f"未知建图方法: {method}")

        self.times = {
            'insert': [],
            'voxelize': [],
            'total': []
        }

    def generate_random_pointcloud(self, num_points=10000, center=None, radius=5.0):
        """生成随机点云数据"""
        if center is None:
            center = np.array([0.0, 0.0, 2.0])

        # 生成球体内的随机点
        theta = np.random.uniform(0, 2*np.pi, num_points)
        phi = np.random.uniform(0, np.pi, num_points)
        r = np.random.uniform(0, radius, num_points)

        x = center[0] + r * np.sin(phi) * np.cos(theta)
        y = center[1] + r * np.sin(phi) * np.sin(theta)
        z = center[2] + r * np.cos(phi)

        xyz = np.column_stack([x, y, z]).astype(np.float32)
        rgb = np.random.randint(0, 255, (num_points, 3), dtype=np.uint8)

        return xyz, rgb

    def run_insertion_test(self, num_frames=100, points_per_frame=10000):
        """运行点云插入性能测试"""
        print(f"\n[性能测试] 开始点云插入测试")
        print(f"  - 帧数: {num_frames}")
        print(f"  - 每帧点数: {points_per_frame}")
        print(f"  - 总点数: {num_frames * points_per_frame}")

        total_start = time.time()

        for i in range(num_frames):
            # 生成点云
            xyz, rgb = self.generate_random_pointcloud(
                num_points=points_per_frame,
                center=np.array([i*0.1, 0.0, 2.0])  # 逐渐移动
            )

            # 设置位姿 (简单的恒等变换)
            R = np.eye(3, dtype=np.float32)
            T = np.array([i*0.1, 0.0, 0.0], dtype=np.float32)

            for j in range(3):
                self.mapping.input_T[None][j] = T[j]
                for k in range(3):
                    self.mapping.input_R[None][j, k] = R[j, k]

            # 插入点云
            insert_start = time.time()
            self.mapping.recast_pcl_to_map(xyz, rgb, len(xyz))
            insert_time = (time.time() - insert_start) * 1000  # ms

            self.times['insert'].append(insert_time)

            if (i + 1) % 10 == 0:
                avg_time = np.mean(self.times['insert'][-10:])
                print(f"  帧 {i+1}/{num_frames}: 平均插入时间 = {avg_time:.2f} ms")

        total_time = (time.time() - total_start) * 1000
        self.times['total'].append(total_time)

        print(f"\n[测试结果] 插入测试完成")
        print(f"  - 总时间: {total_time:.2f} ms ({total_time/1000:.2f} s)")
        print(f"  - 平均每帧: {np.mean(self.times['insert']):.2f} ms")
        print(f"  - 中位数: {np.median(self.times['insert']):.2f} ms")
        print(f"  - 最小值: {np.min(self.times['insert']):.2f} ms")
        print(f"  - 最大值: {np.max(self.times['insert']):.2f} ms")

        # 计算处理速度 (点/秒)
        total_points = num_frames * points_per_frame
        processing_speed = total_points / (total_time / 1000)
        print(f"  - 处理速度: {processing_speed:,.0f} 点/秒")

        return self.times

    def run_voxelize_test(self):
        """运行体素化性能测试"""
        print(f"\n[性能测试] 开始体素化测试")

        voxelize_times = []

        for i in range(10):
            start = time.time()

            if self.method == 'octo':
                self.mapping.cvt_occupy_to_voxels(level=1)
            else:  # tsdf
                self.mapping.cvt_TSDF_surface_to_voxels()

            elapsed = (time.time() - start) * 1000
            voxelize_times.append(elapsed)

        self.times['voxelize'] = voxelize_times

        print(f"[测试结果] 体素化测试完成")
        print(f"  - 平均时间: {np.mean(voxelize_times):.2f} ms")
        print(f"  - 中位数: {np.median(voxelize_times):.2f} ms")
        print(f"  - 最小值: {np.min(voxelize_times):.2f} ms")
        print(f"  - 最大值: {np.max(voxelize_times):.2f} ms")

        return voxelize_times

    def save_results(self, filename=None):
        """保存测试结果"""
        if filename is None:
            filename = f"test_results_{self.method}_{'cuda' if self.cuda else 'cpu'}.txt"

        with open(filename, 'w') as f:
            f.write("="*60 + "\n")
            f.write("TaichiSLAM 性能测试结果\n")
            f.write("="*60 + "\n\n")
            f.write(f"测试配置:\n")
            f.write(f"  - 建图方法: {self.method}\n")
            f.write(f"  - 运行设备: {'CUDA' if self.cuda else 'CPU'}\n")
            f.write(f"  - 体素大小: {self.voxel_size} m\n")
            f.write(f"  - 地图大小: {self.map_size_xy}m x {self.map_size_xy}m x {self.map_size_z}m\n")
            f.write("\n")

            if self.times['insert']:
                f.write("点云插入性能:\n")
                f.write(f"  - 平均时间: {np.mean(self.times['insert']):.2f} ms\n")
                f.write(f"  - 中位数: {np.median(self.times['insert']):.2f} ms\n")
                f.write(f"  - 最小值: {np.min(self.times['insert']):.2f} ms\n")
                f.write(f"  - 最大值: {np.max(self.times['insert']):.2f} ms\n")
                f.write("\n")

            if self.times['voxelize']:
                f.write("体素化性能:\n")
                f.write(f"  - 平均时间: {np.mean(self.times['voxelize']):.2f} ms\n")
                f.write(f"  - 中位数: {np.median(self.times['voxelize']):.2f} ms\n")
                f.write(f"  - 最小值: {np.min(self.times['voxelize']):.2f} ms\n")
                f.write(f"  - 最大值: {np.max(self.times['voxelize']):.2f} ms\n")
                f.write("\n")

        print(f"\n[结果保存] 测试结果已保存到: {filename}")


def main():
    parser = argparse.ArgumentParser(description='TaichiSLAM 建图算法性能测试')
    parser.add_argument('-m', '--method', type=str, default='octo',
                        choices=['octo', 'tsdf'], help='建图方法 (octo 或 tsdf)')
    parser.add_argument('-c', '--cuda', action='store_true',
                        help='使用CUDA加速')
    parser.add_argument('-f', '--frames', type=int, default=100,
                        help='测试帧数')
    parser.add_argument('-p', '--points', type=int, default=10000,
                        help='每帧点数')
    parser.add_argument('-v', '--voxel-size', type=float, default=0.05,
                        help='体素大小 (米)')
    parser.add_argument('-s', '--map-size', type=float, default=50,
                        help='地图XY平面大小 (米)')
    parser.add_argument('-z', '--map-height', type=float, default=10,
                        help='地图高度 (米)')
    parser.add_argument('--no-insertion-test', action='store_true',
                        help='跳过插入测试')
    parser.add_argument('--no-voxelize-test', action='store_true',
                        help='跳过体素化测试')
    args = parser.parse_args()

    print("="*60)
    print("TaichiSLAM 建图算法性能测试")
    print("="*60)
    print(f"\n测试配置:")
    print(f"  - 建图方法: {args.method}")
    print(f"  - 运行设备: {'CUDA' if args.cuda else 'CPU'}")
    print(f"  - 测试帧数: {args.frames}")
    print(f"  - 每帧点数: {args.points}")
    print(f"  - 体素大小: {args.voxel_size} m")
    print(f"  - 地图大小: {args.map_size}m x {args.map_size}m x {args.map_height}m")

    # 创建测试实例
    test = MappingPerformanceTest(
        method=args.method,
        cuda=args.cuda,
        voxel_size=args.voxel_size,
        map_size_xy=args.map_size,
        map_size_z=args.map_height
    )

    # 运行插入测试
    if not args.no_insertion_test:
        test.run_insertion_test(
            num_frames=args.frames,
            points_per_frame=args.points
        )

    # 运行体素化测试
    if not args.no_voxelize_test:
        test.run_voxelize_test()

    # 保存结果
    test.save_results()

    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
