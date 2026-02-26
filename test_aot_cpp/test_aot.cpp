/**
 * Simple test to verify AOT module loading in C++
 */

#include <taichi/taichi_core.h>
#include <iostream>
#include <string>
#include <cstring>

int main(int argc, char** argv) {
    std::cout << "Taichi AOT Module Loading Test" << std::endl;
    std::cout << "==============================" << std::endl;
    
    // AOT module path
    std::string aot_path = "./aot_modules/octomap_cpu";
    if (argc > 1) {
        aot_path = argv[1];
    }
    
    std::cout << "AOT module path: " << aot_path << std::endl;
    
    // Check if directory exists
    // Note: We're just testing the C-API loading here
    
    // Initialize Taichi runtime (CPU for testing)
    TiRuntime runtime = ti_create_runtime(TI_ARCH_X64);  // Use X64 for CPU
    if (!runtime) {
        std::cerr << "Failed to create Taichi runtime!" << std::endl;
        return 1;
    }
    std::cout << "✓ Created Taichi runtime (CPU)" << std::endl;
    
    // Load AOT module
    TiAotModule module = ti_load_aot_module(runtime, aot_path.c_str());
    if (!module) {
        std::cerr << "Failed to load AOT module from: " << aot_path << std::endl;
        ti_destroy_runtime(runtime);
        return 1;
    }
    std::cout << "✓ Loaded AOT module" << std::endl;
    
    // Get kernels
    TiKernel clear_grid = ti_get_aot_module_kernel(module, "clear_grid");
    if (!clear_grid) {
        std::cerr << "Failed to get kernel: clear_grid" << std::endl;
        ti_destroy_aot_module(module);
        ti_destroy_runtime(runtime);
        return 1;
    }
    std::cout << "✓ Got kernel: clear_grid" << std::endl;
    
    // Get fields
    TiMemory occupancy = ti_get_aot_module_field_memory(module, "occupancy");
    if (!occupancy) {
        std::cerr << "Failed to get field: occupancy" << std::endl;
        ti_destroy_aot_module(module);
        ti_destroy_runtime(runtime);
        return 1;
    }
    std::cout << "✓ Got field: occupancy" << std::endl;
    
    // Clean up
    ti_destroy_aot_module(module);
    ti_destroy_runtime(runtime);
    
    std::cout << std::endl;
    std::cout << "==============================" << std::endl;
    std::cout << "All tests passed!" << std::endl;
    std::cout << "AOT module is ready for use." << std::endl;
    
    return 0;
}
