/**
 * TaichiSLAM AOT Runtime Implementation
 *
 * This implementation uses Taichi C API (not C++ wrapper) for compatibility
 * with Taichi 1.7.4+
 */

#include "taichislam_aot.h"
#include <taichi/taichi_core.h>

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace taichislam {

// Helper macros for error checking
#define CHECK_TI_ERROR(err, msg) \
    if (err != TI_ERROR_SUCCESS) { \
        last_error_ = std::string(msg) + ": " + std::to_string(err); \
        std::cerr << "[TaichiSLAM Error] " << last_error_ << std::endl; \
        return; \
    }

#define CHECK_TI_ERROR_RET(err, msg, ret) \
    if (err != TI_ERROR_SUCCESS) { \
        last_error_ = std::string(msg) + ": " + std::to_string(err); \
        std::cerr << "[TaichiSLAM Error] " << last_error_ << std::endl; \
        return ret; \
    }

// Internal implementation class
class TaichiSLAMRuntime::Impl {
public:
    TiArch arch_ = TI_ARCH_VULKAN;
    uint32_t device_index_ = 0;
    TiRuntime runtime_ = nullptr;
    bool initialized_ = false;
    std::string last_error_;

    Impl(TiArch arch, uint32_t device_index)
        : arch_(arch), device_index_(device_index) {
    }

    ~Impl() {
        if (runtime_) {
            ti_destroy_runtime(runtime_);
            runtime_ = nullptr;
        }
    }

    bool initialize() {
        if (initialized_) {
            return true;
        }

        // Create Taichi runtime
        TiError err = ti_create_runtime(arch_, device_index_, &runtime_);
        if (err != TI_ERROR_SUCCESS || !runtime_) {
            last_error_ = "Failed to create Taichi runtime";
            std::cerr << "[TaichiSLAM Error] " << last_error_ << std::endl;
            return false;
        }

        initialized_ = true;
        std::cout << "[TaichiSLAM] Runtime initialized successfully (arch="
                  << arch_ << ")" << std::endl;
        return true;
    }
};

// OctomapMapping implementation
class OctomapMappingImpl : public OctomapMapping {
public:
    std::shared_ptr<TaichiSLAMRuntime::Impl> runtime_;
    TiAotModule module_ = nullptr;
    MappingConfig config_;

    // Kernels
    TiKernel clear_grid_kernel_ = nullptr;
    TiKernel insert_points_kernel_ = nullptr;
    TiKernel extract_occupancy_kernel_ = nullptr;

    // Fields
    TiMemory occupancy_memory_ = nullptr;
    TiMemory point_buffer_memory_ = nullptr;
    TiMemory num_points_memory_ = nullptr;
    TiMemory output_buffer_memory_ = nullptr;
    TiMemory num_output_memory_ = nullptr;

    OctomapMappingImpl(std::shared_ptr<TaichiSLAMRuntime::Impl> runtime,
                       TiAotModule module,
                       const MappingConfig& config)
        : runtime_(runtime), module_(module), config_(config) {

        // Get kernels
        clear_grid_kernel_ = ti_get_aot_module_kernel(module_, "clear_grid");
        insert_points_kernel_ = ti_get_aot_module_kernel(module_, "insert_points_kernel");
        extract_occupancy_kernel_ = ti_get_aot_module_kernel(module_, "extract_occupancy_kernel");

        // Get fields
        occupancy_memory_ = ti_get_aot_module_field_memory(module_, "occupancy");
        point_buffer_memory_ = ti_get_aot_module_field_memory(module_, "point_buffer");
        num_points_memory_ = ti_get_aot_module_field_memory(module_, "num_points");
        output_buffer_memory_ = ti_get_aot_module_field_memory(module_, "output_buffer");
        num_output_memory_ = ti_get_aot_module_field_memory(module_, "num_output");
    }

    ~OctomapMappingImpl() {
        // Module is owned by runtime, don't destroy here
    }

    void insertPointCloud(const float* points, const uint8_t* colors,
                         size_t num_points, const Pose& pose) override {
        if (!runtime_->runtime_ || num_points == 0) {
            return;
        }

        // TODO: Transform points by pose
        // For now, assume points are in world coordinates

        // Copy points to device
        // Note: This is a simplified version. In production, you'd use
        // ti_launch_compute_graph or proper memory management

        std::cout << "[OctomapMapping] Inserting " << num_points << " points" << std::endl;

        // Launch clear_grid kernel first
        ti_launch_kernel(runtime_->runtime_, clear_grid_kernel_, 0, nullptr, nullptr);

        // TODO: Launch insert_points_kernel with proper arguments
        // This requires setting up ndarray arguments which is more complex

        ti_flush(runtime_->runtime_);
    }

    void insertDepthImage(const uint16_t* depth_image, const uint8_t* color_image,
                         int width, int height, const Pose& pose,
                         float fx, float fy, float cx, float cy) override {
        // TODO: Convert depth image to point cloud and insert
        std::cout << "[OctomapMapping] insertDepthImage not yet implemented" << std::endl;
    }

    VoxelGrid extractVoxels(int level) override {
        VoxelGrid grid;

        if (!runtime_->runtime_) {
            return grid;
        }

        // TODO: Launch extract_occupancy_kernel and read results
        // For now, return empty grid

        return grid;
    }

    void reset() override {
        if (runtime_->runtime_ && clear_grid_kernel_) {
            ti_launch_kernel(runtime_->runtime_, clear_grid_kernel_, 0, nullptr, nullptr);
            ti_flush(runtime_->runtime_);
        }
    }

    void getStatistics(size_t* num_voxels, float* map_size_mb) override {
        // TODO: Calculate actual statistics
        if (num_voxels) *num_voxels = 0;
        if (map_size_mb) *map_size_mb = 0.0f;
    }
};

// Public API Implementation

TaichiSLAMRuntime::TaichiSLAMRuntime(TiArch arch, uint32_t device_index)
    : impl_(std::make_unique<Impl>(arch, device_index)) {

    // Try to initialize immediately
    impl_->initialize();
}

TaichiSLAMRuntime::~TaichiSLAMRuntime() = default;

std::shared_ptr<OctomapMapping> TaichiSLAMRuntime::loadOctomap(
    const std::string& module_path,
    const MappingConfig& config) {

    if (!impl_->initialize()) {
        std::cerr << "[TaichiSLAM] Failed to initialize runtime" << std::endl;
        return nullptr;
    }

    // Load AOT module
    TiAotModule module = ti_load_aot_module(impl_->runtime_, module_path.c_str());
    if (!module) {
        std::cerr << "[TaichiSLAM] Failed to load AOT module from: " << module_path << std::endl;
        return nullptr;
    }

    std::cout << "[TaichiSLAM] Successfully loaded Octomap AOT module" << std::endl;

    // Create OctomapMapping implementation
    return std::make_shared<OctomapMappingImpl>(impl_, module, config);
}

std::shared_ptr<TSDFMapping> TaichiSLAMRuntime::loadTSDF(
    const std::string& module_path,
    const MappingConfig& config) {

    // TODO: Implement TSDF loading
    std::cerr << "[TaichiSLAM] TSDF loading not yet implemented" << std::endl;
    return nullptr;
}

bool TaichiSLAMRuntime::isInitialized() const {
    return impl_->initialized_;
}

std::string TaichiSLAMRuntime::getLastError() const {
    return impl_->last_error_;
}

} // namespace taichislam
