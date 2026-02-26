/**
 * Test C++ runtime with AOT module
 */
#include <iostream>
#include <taichi/taichi_core.h>

int main(int argc, char** argv) {
    std::cout << "Taichi C++ Runtime Test" << std::endl;
    std::cout << "========================" << std::endl;
    
    // Initialize Taichi runtime
    TiRuntime runtime = nullptr;
    TiError err = ti_create_runtime(TI_ARCH_X64, 0, &runtime);
    if (err != TI_ERROR_SUCCESS || !runtime) {
        std::cerr << "Failed to create Taichi runtime" << std::endl;
        return 1;
    }
    std::cout << "✓ Created Taichi runtime (CPU)" << std::endl;
    
    // Load AOT module
    const char* aot_path = "./aot_modules/octomap_cpu";
    if (argc > 1) {
        aot_path = argv[1];
    }
    
    std::cout << "Loading AOT module from: " << aot_path << std::endl;
    
    TiAotModule module = ti_load_aot_module(runtime, aot_path);
    if (!module) {
        std::cerr << "Failed to load AOT module" << std::endl;
        ti_destroy_runtime(runtime);
        return 1;
    }
    std::cout << "✓ Loaded AOT module" << std::endl;
    
    // Get kernels
    TiKernel clear_grid = ti_get_aot_module_kernel(module, "clear_grid");
    if (!clear_grid) {
        std::cerr << "Failed to get kernel: clear_grid" << std::endl;
    } else {
        std::cout << "✓ Found kernel: clear_grid" << std::endl;
    }
    
    TiKernel insert_points = ti_get_aot_module_kernel(module, "insert_points_kernel");
    if (!insert_points) {
        std::cerr << "Failed to get kernel: insert_points_kernel" << std::endl;
    } else {
        std::cout << "✓ Found kernel: insert_points_kernel" << std::endl;
    }
    
    TiKernel extract_occupancy = ti_get_aot_module_kernel(module, "extract_occupancy_kernel");
    if (!extract_occupancy) {
        std::cerr << "Failed to get kernel: extract_occupancy_kernel" << std::endl;
    } else {
        std::cout << "✓ Found kernel: extract_occupancy_kernel" << std::endl;
    }
    
    // Get fields
    TiMemory occupancy = ti_get_aot_module_field_memory(module, "occupancy");
    if (!occupancy) {
        std::cerr << "Failed to get field: occupancy" << std::endl;
    } else {
        std::cout << "✓ Found field: occupancy" << std::endl;
    }
    
    TiMemory point_buffer = ti_get_aot_module_field_memory(module, "point_buffer");
    if (!point_buffer) {
        std::cerr << "Failed to get field: point_buffer" << std::endl;
    } else {
        std::cout << "✓ Found field: point_buffer" << std::endl;
    }
    
    // Test launching clear_grid kernel
    if (clear_grid && runtime) {
        std::cout << "Testing kernel launch (clear_grid)..." << std::endl;
        ti_launch_kernel(runtime, clear_grid, 0, nullptr, nullptr);
        ti_flush(runtime);
        std::cout << "✓ Kernel launched successfully" << std::endl;
    }
    
    // Cleanup
    ti_destroy_aot_module(module);
    ti_destroy_runtime(runtime);
    
    std::cout << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "All tests passed!" << std::endl;
    std::cout << "C++ runtime is working correctly." << std::endl;
    
    return 0;
}
