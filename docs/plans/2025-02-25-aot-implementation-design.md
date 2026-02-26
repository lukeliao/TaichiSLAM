# TaichiSLAM AOT Implementation Design

**Date**: 2025-02-25
**Target**: Desktop robotic arm with ARM Mali G52 (Vulkan 1.2)
**Resolution**: 128³ @ 3cm (3.84m³ coverage, 8.4MB memory)
**Use Case**: Real-time environment perception for motion planning

---

## 1. System Architecture

### 1.1 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Desktop Robotic Arm System                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────────────────┐    │
│  │   RGB-D     │      │   Point     │      │   AOT Mapping Runtime   │    │
│  │   Camera    │─────▶│   Cloud     │─────▶│   (C++ / Taichi C-API)  │    │
│  │             │      │  (3cm res)  │      │                         │    │
│  └─────────────┘      └─────────────┘      └─────────────────────────┘    │
│                                                           │                  │
│                              ┌────────────────────────────┘                  │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                        Motion Planning Pipeline                       │  │
│  │                                                                      │  │
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐  │  │
│  │  │  128³ Grid  │────▶│  Collision  │────▶│  Path Planning (RRT*)   │  │  │
│  │  │  (3cm res)  │     │  Detection  │     │  / CHOMP / OMPL         │  │  │
│  │  └─────────────┘     └─────────────┘     └─────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ARM Mali G52 (Vulkan 1.2)                          │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      AOT Module (Pre-compiled)                       │   │
│  │                                                                    │   │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐    │   │
│  │  │  clear_grid()   │  │insert_points() │  │extract_occup() │    │   │
│  │  │  (SPIR-V)       │  │  (SPIR-V)       │  │  (SPIR-V)       │    │   │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         GPU Memory                                  │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              Occupancy Grid (128³ × float32)                 │   │   │
│  │  │                                                            │   │   │
│  │  │   Grid Size: 128 × 128 × 128 = 2,097,152 cells            │   │   │
│  │  │   Memory: 2,097,152 × 4 bytes = 8,388,608 bytes ≈ 8.4 MB  │   │   │
│  │  │                                                            │   │   │
│  │  │   Coverage: 3.84m × 3.84m × 3.84m (±1.92m from center)   │   │   │
│  │  │   Resolution: 3cm per voxel                                 │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │              Point Cloud Buffer (4096 × 3 floats)             │   │   │
│  │  │   Memory: 4096 × 3 × 4 bytes = 48 KB                         │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  │                                                                     │   │
│  │  Total GPU Memory: ~8.5 MB (well within Mali G52 budget)         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Memory Layout Details

```
Occupancy Grid Indexing:

World Coordinates → Grid Index:
  grid_x = floor((world_x - world_min) / voxel_scale)
  grid_y = floor((world_y - world_min) / voxel_scale)
  grid_z = floor((world_z - world_min) / voxel_scale)

Where:
  voxel_scale = 0.03  (3cm)
  world_min = -1.92   (-half_size)
  world_max = +1.92   (+half_size)
  grid_size = 128

Example:
  World point (0.5, -0.3, 1.0) m
  → Grid index (81, 54, 97)
  → occupancy[81, 54, 97] += 1

Occupancy Value Interpretation:
  0.0     = unknown / empty
  1.0+    = occupied (number of points falling in this voxel)

  For planning collision detection:
    if occupancy[i,j,k] > threshold (e.g., 1.0):
        voxel is considered occupied

Memory Access Pattern:
  Kernel: insert_points_kernel
    For each point in parallel:
      1. Read point coordinates (global memory)
      2. Compute grid index (register)
      3. Atomic add to occupancy grid (global memory, coalesced if sorted)

  Optimization: Sort points by grid location for better coalescing
```

### 1.3 Kernel Specifications

```python
# Kernel 1: clear_grid
# Purpose: Reset occupancy grid to zero
# Execution: Launch at start of each frame or when resetting map
@ti.kernel
def clear_grid(self):
    """Clear all voxels in the occupancy grid to 0.0"""
    for i, j, k in ti.ndrange(self.grid_size, self.grid_size, self.grid_size):
        self.occupancy[i, j, k] = 0.0

# AOT Export:
#   - No template arguments
#   - No ndarray arguments
#   - Simple 3D grid iteration
#   - Export: module.add_kernel(clear_grid, template_args={})


# Kernel 2: insert_points_kernel
# Purpose: Insert batch of points into occupancy grid
# Execution: Called for each 4096-point batch
@ti.kernel
def insert_points_kernel(self, points: ti.types.ndarray(ndim=2)):
    """
    Insert points from numpy array into occupancy grid.

    Args:
        points: ndarray of shape (num_points, 3) in device memory
                  Each row is [x, y, z] in world coordinates (meters)

    Side effects:
        Increments occupancy count for each voxel containing points
    """
    for idx in range(self.num_points[None]):  # num_points is ti.field
        # Read point coordinates
        x = points[idx, 0]
        y = points[idx, 1]
        z = points[idx, 2]

        # World to grid coordinate transformation
        # world_min = -1.92, voxel_scale = 0.03
        gx = int((x - self.world_min) / self.voxel_scale)
        gy = int((y - self.world_min) / self.voxel_scale)
        gz = int((z - self.world_min) / self.voxel_scale)

        # Bounds check and atomic increment
        if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size and 0 <= gz < self.grid_size:
            ti.atomic_add(self.occupancy[gx, gy, gz], 1.0)

# AOT Export:
#   - Template argument for ndarray shape/dtype
#   - Export: module.add_kernel(
#       insert_points_kernel,
#       template_args={"points": ti.types.ndarray(ndim=2, dtype=ti.f32)}
#     )


# Kernel 3: extract_occupancy_kernel
# Purpose: Extract occupied voxels above threshold for planning
# Execution: Called when planning needs current map
@ti.kernel
def extract_occupancy_kernel(self, threshold: ti.f32):
    """
    Extract voxels with occupancy > threshold.

    Args:
        threshold: minimum occupancy value to consider voxel as occupied
                  (typically 1.0, meaning at least 1 point in voxel)

    Output:
        Fills output_buffer with (x, y, z, occupancy) for each occupied voxel
        Sets num_output to number of occupied voxels
    """
    self.num_output[None] = 0

    for i, j, k in ti.ndrange(self.grid_size, self.grid_size, self.grid_size):
        occ = self.occupancy[i, j, k]
        if occ > threshold:
            # Compute world coordinates from grid indices
            # Grid center is at half-voxel offset
            x = self.world_min + (i + 0.5) * self.voxel_scale
            y = self.world_min + (j + 0.5) * self.voxel_scale
            z = self.world_min + (k + 0.5) * self.voxel_scale

            # Atomic add to get insertion index
            idx = ti.atomic_add(self.num_output[None], 1)

            # Bounds check for output buffer
            if idx < self.grid_size ** 3:
                self.output_buffer[idx] = ti.Vector([x, y, z, occ])

# AOT Export:
#   - Template argument for threshold (f32)
#   - Export: module.add_kernel(
#       extract_occupancy_kernel,
#       template_args={"threshold": ti.f32}
#     )
```

---

## 4. C++ Runtime Implementation

### 4.1 API Design (C-style for easy integration)

```cpp
// taichislam_aot.h - C API for robotic arm integration

#ifndef TAICHISLAM_AOT_H
#define TAICHISLAM_AOT_H

#include <taichi/taichi_core.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to mapping instance
typedef struct TaichiSLAM_Mapping* TaichiSLAM_Handle;

// Configuration for mapping instance
typedef struct {
    float voxel_scale;      // 0.03 for 3cm
    int grid_size;          // 128
    int max_points_batch;   // 4096
    const char* aot_path;   // path to AOT module
    TiArch arch;            // TI_ARCH_VULKAN for Mali
} TaichiSLAM_Config;

// Point cloud data (host memory)
typedef struct {
    const float* xyz;       // N×3 float array (x,y,z in meters)
    int num_points;         // number of points
} TaichiSLAM_PointCloud;

// Voxel data (output)
typedef struct {
    float xyz[3];           // world coordinates (meters)
    float occupancy;        // occupancy value
} TaichiSLAM_Voxel;

// ============ Lifecycle Functions ============

// Create mapping instance
// Returns NULL on error, check ti_get_last_error_message()
TaichiSLAM_Handle taichislam_create(const TaichiSLAM_Config* config);

// Destroy mapping instance and free resources
void taichislam_destroy(TaichiSLAM_Handle handle);

// ============ Processing Functions ============

// Clear the occupancy grid (call at start of new frame)
void taichislam_clear(TaichiSLAM_Handle handle);

// Insert point cloud into occupancy grid
// Points are processed in batches internally
void taichislam_insert_pointcloud(
    TaichiSLAM_Handle handle,
    const TaichiSLAM_PointCloud* pc,
    const float pose[16]  // 4×4 transform matrix (row-major)
);

// Extract occupied voxels above threshold
// Returns number of occupied voxels, fills output buffer
// Call twice if needed: once with NULL to get count, once with buffer
int taichislam_extract_occupied(
    TaichiSLAM_Handle handle,
    float threshold,    // typically 1.0
    TaichiSLAM_Voxel* output_buffer,  // can be NULL
    int buffer_capacity               // max voxels to write
);

// ============ Utility Functions ============

// Get last error message
const char* taichislam_get_error();

// Get memory usage statistics
void taichislam_get_memory_stats(
    TaichiSLAM_Handle handle,
    size_t* gpu_memory_used,
    size_t* host_memory_used
);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TAICHISLAM_AOT_H
```

---

## 5. Implementation Roadmap

### Phase 1: Python AOT Export (Days 1-2)

**Day 1: AOT Octomap Implementation**
- [ ] Create `aot_octomap.py` with fixed 128³ dense grid
- [ ] Implement 3 core kernels with AOT-friendly signatures
- [ ] Test kernel compilation on host (x86/Vulkan)

**Day 2: AOT Export Script**
- [ ] Complete `aot_export.py` with module export logic
- [ ] Generate metadata.json with kernel signatures
- [ ] Export test module and verify with `ti inspect`

**Deliverable**: Working AOT module export on host

### Phase 2: C++ Runtime (Days 3-4)

**Day 3: Core Runtime Implementation**
- [ ] Implement `taichislam_create/destroy` with C-API
- [ ] Implement `taichislam_clear/insert/extract`
- [ ] Device memory management (allocation/binding)
- [ ] Error handling and logging

**Day 4: API Completion & Testing**
- [ ] Complete remaining API functions
- [ ] Write test program (`test_basic.cpp`)
- [ ] Test on host with CPU/Vulkan backend
- [ ] Debug and fix issues

**Deliverable**: Compilable C++ library with passing tests

### Phase 3: ARM Deployment (Day 5)

**Day 5: Cross-Compilation & Validation**
- [ ] Set up ARM cross-compilation toolchain
- [ ] Configure CMake for ARM Mali G52
- [ ] Build for target architecture
- [ ] Deploy to ARM device and test
- [ ] Performance benchmarking

**Deliverable**: Working system on ARM Mali G52

---

## 6. Risk Assessment & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| AOT kernel compilation fails | Medium | High | Start with simplest kernel, incremental testing |
| NdArray binding issues | Medium | High | Verify with Taichi examples, use explicit shapes |
| ARM Vulkan driver issues | Low | High | Test with `vulkaninfo`, fallback to CPU for debugging |
| Memory allocation failures | Low | Medium | Pre-allocate fixed pools, check available memory |
| Performance below real-time | Medium | High | Profile with `ti.profiler`, optimize kernel launch config |

---

## 7. Success Criteria

### Functional Requirements
- [ ] AOT module exports successfully for Vulkan backend
- [ ] C++ runtime loads AOT module and creates context
- [ ] Point cloud insertion produces correct occupancy grid
- [ ] Extraction returns occupied voxels above threshold
- [ ] System runs on ARM Mali G52 without errors

### Performance Requirements
- [ ] Point cloud processing: < 10ms for 10k points (100Hz effective)
- [ ] Memory usage: < 20MB GPU memory total
- [ ] End-to-end latency: < 50ms (camera to planning input)

---

## 8. Next Steps

1. **Review and approve** this design document
2. **Set up development environment**:
   - Taichi 1.7+ with Vulkan support
   - CMake 3.14+, GCC/Clang
   - ARM cross-compiler (if testing on host)
3. **Begin Phase 1**: Python AOT Export implementation

---

**Document Status**: Design Complete
**Next Action**: Proceed to implementation
