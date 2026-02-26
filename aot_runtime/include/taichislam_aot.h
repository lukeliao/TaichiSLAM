/**
 * TaichiSLAM AOT Runtime C++ Interface
 *
 * This header provides a high-level C++ interface for running TaichiSLAM
 * mapping algorithms (Octomap/TSDF) using Taichi AOT modules on GPU.
 *
 * Supports:
 *   - ARM Mali GPU via Vulkan/OpenGL ES
 *   - Desktop GPUs via CUDA/Vulkan/OpenGL
 *   - Cross-platform deployment
 *
 * Example usage:
 *   TaichiSLAMRuntime runtime(TI_ARCH_VULKAN);
 *   OctomapMapping mapping = runtime.loadOctomap("./aot_modules/taichislam_octo");
 *   mapping.insertPointCloud(points, colors, num_points, R, T);
 */

#pragma once

#include <taichi/taichi_core.h>
#include <taichi/cpp/taichi.hpp>

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>

namespace taichislam {

// Forward declarations
class MappingBase;
class OctomapMapping;
class TSDFMapping;

/**
 * Configuration for mapping algorithms
 */
struct MappingConfig {
    // Map dimensions
    float map_size_xy = 30.0f;      // Map size in XY plane (meters)
    float map_size_z = 10.0f;       // Map size in Z direction (meters)
    float voxel_scale = 0.1f;       // Voxel size (meters)

    // Octomap specific
    int octo_k = 2;                 // Octree branching factor
    int min_occupy_thres = 2;       // Minimum occupancy threshold

    // TSDF specific
    int num_voxel_per_blk_axis = 8; // Voxels per block axis
    float max_ray_length = 10.0f;   // Maximum ray length
    float min_ray_length = 0.3f;    // Minimum ray length

    // General
    bool enable_texture = false;    // Enable color/texture
    int max_disp_particles = 500000; // Maximum display particles

    // Camera parameters (for depth image input)
    float fx = 320.0f, fy = 320.0f; // Focal lengths
    float cx = 320.0f, cy = 240.0f; // Principal point
    int image_width = 640, image_height = 480;
};

/**
 * Pose representation (Rotation + Translation)
 */
struct Pose {
    float R[9];     // Row-major 3x3 rotation matrix
    float T[3];     // Translation vector

    Pose() {
        // Identity rotation
        R[0] = 1; R[1] = 0; R[2] = 0;
        R[3] = 0; R[4] = 1; R[5] = 0;
        R[6] = 0; R[7] = 0; R[8] = 1;
        // Zero translation
        T[0] = T[1] = T[2] = 0;
    }

    Pose(const float* r, const float* t) {
        memcpy(R, r, sizeof(float) * 9);
        memcpy(T, t, sizeof(float) * 3);
    }
};

/**
 * Point cloud data container
 */
struct PointCloud {
    std::vector<float> points;      // XYZ coordinates (N x 3)
    std::vector<uint8_t> colors;  // RGB colors (N x 3), optional
    size_t num_points = 0;

    PointCloud() = default;

    PointCloud(const float* pts, size_t n, const uint8_t* rgb = nullptr) {
        setPoints(pts, n, rgb);
    }

    void setPoints(const float* pts, size_t n, const uint8_t* rgb = nullptr) {
        num_points = n;
        points.resize(n * 3);
        memcpy(points.data(), pts, sizeof(float) * n * 3);

        if (rgb) {
            colors.resize(n * 3);
            memcpy(colors.data(), rgb, sizeof(uint8_t) * n * 3);
        }
    }
};

/**
 * Voxel grid output
 */
struct VoxelGrid {
    std::vector<float> positions;   // XYZ positions
    std::vector<float> sdf_values;    // SDF/TSDF values (optional)
    std::vector<uint8_t> colors;      // RGB colors (optional)
    size_t num_voxels = 0;
};

/**
 * Runtime error exception
 */
class TaichiSLAMError : public std::runtime_error {
public:
    explicit TaichiSLAMError(const std::string& msg) : std::runtime_error(msg) {}
};

/**
 * TaichiSLAM Runtime
 *
 * Manages Taichi runtime and AOT module loading
 */
class TaichiSLAMRuntime {
public:
    /**
     * Constructor
     *
     * @param arch Target architecture (TI_ARCH_VULKAN, TI_ARCH_GLES, etc.)
     * @param device_index GPU device index (default: 0)
     */
    TaichiSLAMRuntime(TiArch arch = TI_ARCH_VULKAN, uint32_t device_index = 0);

    /**
     * Destructor
     */
    ~TaichiSLAMRuntime();

    /**
     * Load Octomap AOT module
     *
     * @param module_path Path to AOT module directory (containing metadata.json)
     * @param config Mapping configuration
     * @return OctomapMapping instance
     */
    std::shared_ptr<OctomapMapping> loadOctomap(
        const std::string& module_path,
        const MappingConfig& config = MappingConfig());

    /**
     * Load TSDF AOT module
     *
     * @param module_path Path to AOT module directory
     * @param config Mapping configuration
     * @return TSDFMapping instance
     */
    std::shared_ptr<TSDFMapping> loadTSDF(
        const std::string& module_path,
        const MappingConfig& config = MappingConfig());

    /**
     * Get native Taichi runtime (for advanced usage)
     */
    ti::Runtime* getNativeRuntime() { return runtime_.get(); }

    /**
     * Check if runtime is initialized
     */
    bool isInitialized() const { return initialized_; }

    /**
     * Get last error message
     */
    std::string getLastError() const { return last_error_; }

private:
    TiArch arch_;
    uint32_t device_index_;
    std::unique_ptr<ti::Runtime> runtime_;
    bool initialized_ = false;
    std::string last_error_;

    void checkError(TiError error, const std::string& context);
};


/**
 * Octomap Mapping Interface
 */
class OctomapMapping {
public:
    virtual ~OctomapMapping() = default;

    /**
     * Insert point cloud into map
     *
     * @param points Point cloud XYZ coordinates (N x 3 float array)
     * @param colors Point cloud RGB colors (N x 3 uint8 array, optional)
     * @param num_points Number of points
     * @param pose Sensor pose (rotation + translation)
     */
    virtual void insertPointCloud(
        const float* points,
        const uint8_t* colors,
        size_t num_points,
        const Pose& pose) = 0;

    /**
     * Insert depth image into map
     *
     * @param depth_image Depth image (H x W uint16 array in mm)
     * @param color_image Color image (H x W x 3 uint8 array, optional)
     * @param width Image width
     * @param height Image height
     * @param pose Sensor pose
     * @param fx, fy, cx, cy Camera intrinsics
     */
    virtual void insertDepthImage(
        const uint16_t* depth_image,
        const uint8_t* color_image,
        int width,
        int height,
        const Pose& pose,
        float fx,
        float fy,
        float cx,
        float cy) = 0;

    /**
     * Extract occupied voxels
     *
     * @param level Octree level to extract
     * @return Voxel grid with positions and colors
     */
    virtual VoxelGrid extractVoxels(int level = 0) = 0;

    /**
     * Reset/clear the map
     */
    virtual void reset() = 0;

    /**
     * Get map statistics
     */
    virtual void getStatistics(size_t* num_voxels, float* map_size_mb) = 0;
};


/**
 * TSDF Mapping Interface
 */
class TSDFMapping {
public:
    virtual ~TSDFMapping() = default;

    /**
     * Insert point cloud into map
     */
    virtual void insertPointCloud(
        const float* points,
        const uint8_t* colors,
        size_t num_points,
        const Pose& pose) = 0;

    /**
     * Insert depth image into map
     */
    virtual void insertDepthImage(
        const uint16_t* depth_image,
        const uint8_t* color_image,
        int width,
        int height,
        const Pose& pose,
        float fx,
        float fy,
        float cx,
        float cy) = 0;

    /**
     * Extract surface voxels (where TSDF ≈ 0)
     */
    virtual VoxelGrid extractSurface() = 0;

    /**
     * Extract TSDF slice at specific Z height
     */
    virtual VoxelGrid extractSlice(float z, float dz = 0.5f) = 0;

    /**
     * Reset/clear the map
     */
    virtual void reset() = 0;

    /**
     * Get map statistics
     */
    virtual void getStatistics(size_t* num_voxels, float* map_size_mb) = 0;
};


} // namespace taichislam
