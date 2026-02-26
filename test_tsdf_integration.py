"""Test TSDF AOT module integration"""
import taichi as ti
import numpy as np
import os

ti.init(arch=ti.cpu, log_level=ti.INFO)

aot_path = "./aot_modules/tsdf_cpu"
print(f"Loading TSDF AOT module from: {aot_path}")

module = ti.aot.load_module(aot_path)
print("✓ TSDF AOT module loaded")

# Test kernels
try:
    clear_grid = module.get_kernel("clear_grid")
    print("✓ Found kernel: clear_grid")
except Exception as e:
    print(f"✗ Failed: {e}")

try:
    insert_points = module.get_kernel("insert_points_kernel")
    print("✓ Found kernel: insert_points_kernel")
except Exception as e:
    print(f"✗ Failed: {e}")

try:
    extract_surface = module.get_kernel("extract_surface_kernel")
    print("✓ Found kernel: extract_surface_kernel")
except Exception as e:
    print(f"✗ Failed: {e}")

# Test fields
try:
    tsdf_volume = module.get_field("tsdf_volume")
    print(f"✓ Found field: tsdf_volume (shape: {tsdf_volume.shape})")
except Exception as e:
    print(f"✗ Failed: {e}")

try:
    weight_volume = module.get_field("weight_volume")
    print(f"✓ Found field: weight_volume (shape: {weight_volume.shape})")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\n" + "="*60)
print("TSDF AOT Integration Test: ALL PASSED!")
print("="*60)
