#!/usr/bin/env python3
"""
TaichiSLAM AOT (Ahead-of-Time) Export Script

This script exports TaichiSLAM mapping algorithms (Octomap/TSDF) to AOT modules
that can be loaded and executed on ARM Mali GPU using Vulkan/OpenGL ES backend.

Usage:
    python aot_export.py --method octo --arch vulkan --output ./aot_modules
    python aot_export.py --method tsdf --arch gles --output ./aot_modules
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import taichi as ti
from taichi.aot import Module

# Import TaichiSLAM mapping modules
sys.path.insert(0, os.path.dirname(__file__))
from taichi_slam.mapping.taichi_octomap import Octomap
from taichi_slam.mapping.dense_tsdf import DenseTSDF


class AOTExporter:
    """
    AOT Module Exporter for TaichiSLAM

    Exports Taichi kernels to AOT modules that can be loaded on target devices
    using Taichi C-API.
    """

    def __init__(self, arch, output_dir, method="octo"):
        """
        Initialize AOT Exporter

        Args:
            arch: Taichi architecture (ti.vulkan, ti.gles, ti.metal, etc.)
            output_dir: Output directory for AOT modules
            method: Mapping method ("octo" or "tsdf")
        """
        self.arch = arch
        self.output_dir = Path(output_dir)
        self.method = method

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Taichi
        ti.init(arch=arch, log_level=ti.INFO)

        print(f"[AOT Exporter] Initialized with arch={arch}, method={method}")
        print(f"[AOT Exporter] Output directory: {self.output_dir.absolute()}")

    def create_mapping_instance(self):
        """Create mapping instance based on method"""
        if self.method == "octo":
            return Octomap(
                texture_enabled=False,
                max_disp_particles=500000,
                min_occupy_thres=2,
                map_scale=[30, 10],
                voxel_scale=0.1,
                K=2
            )
        elif self.method == "tsdf":
            return DenseTSDF(
                texture_enabled=False,
                max_disp_particles=500000,
                map_scale=[30, 10],
                voxel_scale=0.1,
                num_voxel_per_blk_axis=8
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def export_octomap_kernels(self, mapping, module):
        """
        Export Octomap kernels to AOT module

        Args:
            mapping: Octomap instance
            module: Taichi AOT Module
        """
        print("[AOT Exporter] Exporting Octomap kernels...")

        # Export kernels that are independent of ndarray shapes
        # Note: Kernels with dynamic ndarray inputs need special handling in AOT

        # Export init_fields kernel
        module.add_kernel(mapping.init_fields, template_args={})

        # Export cvt_occupy_to_voxels kernel
        module.add_kernel(mapping.cvt_occupy_to_voxels, template_args={'level': 0})

        # Export process_point kernel (if it's a kernel, not func)
        # Note: @ti.func cannot be exported directly, only @ti.kernel

        print("[AOT Exporter] Octomap kernels exported")

    def export_tsdf_kernels(self, mapping, module):
        """
        Export TSDF kernels to AOT module

        Args:
            mapping: DenseTSDF instance
            module: Taichi AOT Module
        """
        print("[AOT Exporter] Exporting TSDF kernels...")

        # Export init_fields kernel
        module.add_kernel(mapping.init_fields, template_args={})

        # Export cvt_TSDF_surface_to_voxels_kernel
        module.add_kernel(
            mapping.cvt_TSDF_surface_to_voxels_kernel,
            template_args={}
        )

        print("[AOT Exporter] TSDF kernels exported")

    def export(self):
        """
        Main export function

        Creates AOT module and exports all kernels
        """
        print(f"\n{'='*60}")
        print(f"TaichiSLAM AOT Export - Method: {self.method}")
        print(f"{'='*60}\n")

        # Create AOT module
        module = Module(self.arch)

        # Create mapping instance
        mapping = self.create_mapping_instance()

        # Export kernels based on method
        if self.method == "octo":
            self.export_octomap_kernels(mapping, module)
        elif self.method == "tsdf":
            self.export_tsdf_kernels(mapping, module)

        # Build and save AOT module
        output_path = self.output_dir / f"taichislam_{self.method}"
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n[AOT Exporter] Building AOT module...")
        module.save(str(output_path))

        # Generate metadata
        metadata = {
            "method": self.method,
            "arch": str(self.arch),
            "taichi_version": ti.__version__,
            "output_path": str(output_path),
            "kernels": self._get_kernel_list(mapping),
            "fields": self._get_field_info(mapping)
        }

        metadata_path = output_path / "aot_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print(f"AOT Export Complete!")
        print(f"{'='*60}")
        print(f"Output: {output_path}")
        print(f"Metadata: {metadata_path}")
        print(f"\nTo use in C++:")
        print(f"  1. Include taichi C-API headers")
        print(f"  2. Load AOT module: ti_load_aot_module(runtime, \"{output_path}\")")
        print(f"  3. Get kernels and launch")
        print(f"{'='*60}\n")

        return output_path

    def _get_kernel_list(self, mapping):
        """Get list of exportable kernels from mapping"""
        kernels = []
        for attr_name in dir(mapping):
            attr = getattr(mapping, attr_name)
            if hasattr(attr, '_is_taichi_kernel'):
                kernels.append(attr_name)
        return kernels

    def _get_field_info(self, mapping):
        """Get information about fields in mapping"""
        fields = {}
        for attr_name in dir(mapping):
            attr = getattr(mapping, attr_name)
            if isinstance(attr, ti.Field) or isinstance(attr, ti.VectorField):
                try:
                    fields[attr_name] = {
                        "type": str(attr.dtype),
                        "shape": attr.shape if hasattr(attr, 'shape') else "dynamic"
                    }
                except:
                    pass
        return fields


def main():
    parser = argparse.ArgumentParser(
        description='TaichiSLAM AOT Export',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export Octomap for Vulkan (desktop/ARM with Vulkan support)
    python aot_export.py --method octo --arch vulkan --output ./aot_modules

    # Export TSDF for OpenGL ES (ARM Mali GPU)
    python aot_export.py --method tsdf --arch gles --output ./aot_modules

    # Export for Metal (Apple devices)
    python aot_export.py --method octo --arch metal --output ./aot_modules
        """
    )

    parser.add_argument('--method', type=str, default='octo',
                        choices=['octo', 'tsdf'],
                        help='Mapping method to export (default: octo)')
    parser.add_argument('--arch', type=str, default='vulkan',
                        choices=['vulkan', 'gles', 'metal', 'cuda', 'opengl'],
                        help='Target architecture (default: vulkan)')
    parser.add_argument('--output', type=str, default='./aot_modules',
                        help='Output directory for AOT modules (default: ./aot_modules)')

    args = parser.parse_args()

    # Map arch string to Taichi arch
    arch_map = {
        'vulkan': ti.vulkan,
        'gles': ti.gles,
        'metal': ti.metal,
        'cuda': ti.cuda,
        'opengl': ti.opengl
    }

    arch = arch_map.get(args.arch, ti.vulkan)

    # Create exporter and run
    exporter = AOTExporter(arch, args.output, args.method)
    exporter.export()


if __name__ == '__main__':
    main()
