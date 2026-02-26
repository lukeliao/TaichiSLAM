/**
 * TaichiSLAM AOT Example: ARM Mali GPU Deployment
 *
 * This example demonstrates how to deploy TaichiSLAM on ARM SoC with Mali GPU,
 * such as Rockchip RK3588, Amlogic S922X, etc.
 *
 * Target Platforms:
 *   - Rockchip RK3588 (Mali-G610)
 *   - Amlogic S922X (Mali-G52)
 *   - Allwinner H618 (Mali-G31)
 *   - Other ARM SoC with Mali GPU
 *
 * Backend Selection:
 *   - Vulkan: Recommended for Mali-G series (G31, G52, G57, G71, etc.)
 *   - OpenGL ES: Fallback for older Mali GPUs
 *
 * Build Instructions:
 *   1. Cross-compile on host (recommended):
 *      mkdir build && cd build
 *      cmake .. -DCMAKE_TOOLCHAIN_FILE=../cmake/arm64-toolchain.cmake \
 *               -DTAICHI_C_API_PATH=/path/to/taichi/arm64 \
 *               -DTARGET_ARCH=ARM_MALI
 *      make -j$(nproc)
 *
 *   2. Build on target device:
 *      mkdir build && cd build
 *      cmake .. -DTAICHI_C_API_PATH=/path/to/taichi/c_api
 *      make -j$(nproc)
 *
 * Running on Target:
 *   export LD_LIBRARY_PATH=/path/to/taichi/lib:$LD_LIBRARY_PATH
 *   ./example_arm_mali ./aot_modules/taichislam_octo_mali
 */

#include <taichislam_aot.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <signal.h>

using namespace taichislam;

// Global flag for graceful shutdown
volatile bool g_running = true;

void signalHandler(int sig) {
    std::cout << "\nReceived signal " << sig << ", shutting down..." << std::endl;
    g_running = false;
}

// Read point cloud from file (simple XYZ format)
bool loadPointCloudFromFile(
    const std::string& filename,
    std::vector<float>& points,
    std::vector<uint8_t>& colors,
    size_t max_points = 100000) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    points.clear();
    colors.clear();

    float x, y, z;
    size_t count = 0;

    while (file >> x >> y >> z && count < max_points) {
        points.push_back(x);
        points.push_back(y);
        points.push_back(z);

        // Default color (white)
        colors.push_back(255);
        colors.push_back(255);
        colors.push_back(255);

        count++;
    }

    std::cout << "Loaded " << count << " points from " << filename << std::endl;
    return true;
}

// Simulate sensor data (for testing)
void generateSimulatedSensorData(
    std::vector<float>& points,
    std::vector<uint8_t>& colors,
    size_t num_points,
    float time) {

    // Generate a simple rotating point cloud (simulating a LiDAR)
    points.resize(num_points * 3);
    colors.resize(num_points * 3);

    float radius = 5.0f;
    float angular_velocity = 1.0f;  // rad/s

    for (size_t i = 0; i < num_points; ++i) {
        float angle = 2.0f * M_PI * i / num_points + angular_velocity * time;
        float r = radius * (0.8f + 0.2f * sinf(3.0f * angle));

        points[i*3 + 0] = r * cosf(angle);
        points[i*3 + 1] = r * sinf(angle);
        points[i*3 + 2] = 0.5f * sinf(2.0f * angle);  // Some height variation

        // Color based on height
        uint8_t intensity = static_cast<uint8_t>(128 + 127 * sinf(2.0f * angle));
        colors[i*3 + 0] = intensity;
        colors[i*3 + 1] = 255 - intensity;
        colors[i*3 + 2] = 128;
    }
}

int main(int argc, char** argv) {
    // Setup signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::cout << "========================================" << std::endl;
    std::cout << "TaichiSLAM AOT: ARM Mali GPU Example" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Parse arguments
    std::string aot_module_path = "./aot_modules/taichislam_octo";
    std::string pointcloud_file = "";
    bool use_simulation = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS] [AOT_MODULE_PATH]" << std::endl;
            std::cout << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -h, --help            Show this help message" << std::endl;
            std::cout << "  -f, --file FILE       Load point cloud from file" << std::endl;
            std::cout << "  --no-simulation       Disable simulation mode" << std::endl;
            std::cout << std::endl;
            std::cout << "Arguments:" << std::endl;
            std::cout << "  AOT_MODULE_PATH       Path to AOT module directory" << std::endl;
            std::cout << std::endl;
            return 0;
        } else if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
            pointcloud_file = argv[++i];
            use_simulation = false;
        } else if (arg == "--no-simulation") {
            use_simulation = false;
        } else if (arg[0] != '-') {
            aot_module_path = arg;
        }
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  AOT Module: " << aot_module_path << std::endl;
    std::cout << "  Mode: " << (use_simulation ? "Simulation" : "File") << std::endl;
    if (!pointcloud_file.empty()) {
        std::cout << "  Input file: " << pointcloud_file << std::endl;
    }
    std::cout << std::endl;

    try {
        //=========================================================================
        // Step 1: Initialize Runtime
        //=========================================================================
        std::cout << "[1/5] Initializing TaichiSLAM Runtime..." << std::endl;

        // Auto-detect best available backend
        TiArch arch = TI_ARCH_VULKAN;  // Default to Vulkan

        if (ti::is_arch_available(TI_ARCH_VULKAN)) {
            arch = TI_ARCH_VULKAN;
            std::cout << "      Using Vulkan backend" << std::endl;
        } else if (ti::is_arch_available(TI_ARCH_GLES)) {
            arch = TI_ARCH_GLES;
            std::cout << "      Using OpenGL ES backend" << std::endl;
        } else if (ti::is_arch_available(TI_ARCH_CUDA)) {
            arch = TI_ARCH_CUDA;
            std::cout << "      Using CUDA backend" << std::endl;
        } else {
            throw TaichiSLAMError("No suitable GPU backend available");
        }

        TaichiSLAMRuntime runtime(arch, 0);
        std::cout << "      Runtime initialized!" << std::endl;
        std::cout << std::endl;

        //=========================================================================
        // Step 2: Load AOT Module
        //=========================================================================
        std::cout << "[2/5] Loading AOT module..." << std::endl;

        MappingConfig config;
        config.map_size_xy = 20.0f;
        config.map_size_z = 5.0f;
        config.voxel_scale = 0.1f;
        config.enable_texture = true;

        auto mapping = runtime.loadOctomap(aot_module_path, config);
        std::cout << "      Module loaded!" << std::endl;
        std::cout << std::endl;

        //=========================================================================
        // Step 3: Prepare Input Data
        //=========================================================================
        std::cout << "[3/5] Preparing input data..." << std::endl;

        std::vector<float> points;
        std::vector<uint8_t> colors;

        if (use_simulation) {
            std::cout << "      Using simulated sensor data" << std::endl;
            generateSimulatedSensorData(points, colors, 10000, 0.0f);
        } else if (!pointcloud_file.empty()) {
            std::cout << "      Loading point cloud from: " << pointcloud_file << std::endl;
            if (!loadPointCloudFromFile(pointcloud_file, points, colors)) {
                std::cerr << "Failed to load point cloud, using simulation" << std::endl;
                generateSimulatedSensorData(points, colors, 10000, 0.0f);
            }
        }

        std::cout << "      Prepared " << points.size() / 3 << " points" << std::endl;
        std::cout << std::endl;

        //=========================================================================
        // Step 4: Process Data
        //=========================================================================
        std::cout << "[4/5] Processing data..." << std::endl;

        int num_frames = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        while (g_running) {
            // Update pose for simulation
            Pose pose;
            float time = num_frames * 0.1f;
            float angle = time;
            pose.T[0] = 2.0f * cosf(angle);
            pose.T[1] = 2.0f * sinf(angle);
            pose.T[2] = 1.5f;

            float c = cosf(angle + M_PI);
            float s = sinf(angle + M_PI);
            pose.R[0] = c;  pose.R[1] = -s;  pose.R[2] = 0;
            pose.R[3] = s;  pose.R[4] =  c;  pose.R[5] = 0;
            pose.R[6] = 0;  pose.R[7] =  0;  pose.R[8] = 1;

            // Update simulation data
            if (use_simulation) {
                generateSimulatedSensorData(points, colors, 10000, time);
            }

            // Insert into map
            mapping->insertPointCloud(
                points.data(),
                colors.data(),
                points.size() / 3,
                pose);

            num_frames++;

            // Print progress
            if (num_frames % 10 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    current_time - start_time).count();
                float fps = num_frames * 1000.0f / elapsed;

                std::cout << "  Frame " << num_frames << " | "
                          << "FPS: " << fps << " | "
                          << "Time: " << elapsed << "ms" << std::endl;
            }

            // Break after some frames if not running interactively
            if (num_frames >= 100 && !isatty(STDIN_FILENO)) {
                break;
            }

            // Small delay to not overwhelm the system
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();

        std::cout << "  Processed " << num_frames << " frames in "
                  << total_elapsed << " ms" << std::endl;
        std::cout << "  Average FPS: " << (num_frames * 1000.0f / total_elapsed)
                  << std::endl;
        std::cout << std::endl;

        //=========================================================================
        // Step 5: Extract Results
        //=========================================================================
        std::cout << "[5/5] Extracting results..." << std::endl;

        VoxelGrid voxels = mapping->extractVoxels(0);
        std::cout << "  Extracted " << voxels.num_voxels << " occupied voxels"
                  << std::endl;

        size_t total_voxels;
        float map_size_mb;
        mapping->getStatistics(&total_voxels, &map_size_mb);
        std::cout << "  Map statistics:" << std::endl;
        std::cout << "    Total voxels: " << total_voxels << std::endl;
        std::cout << "    Memory usage: " << map_size_mb << " MB" << std::endl;

        // Save voxels to file (optional)
        if (!voxels.positions.empty()) {
            std::string output_file = "output_voxels.bin";
            std::ofstream ofs(output_file, std::ios::binary);
            if (ofs) {
                // Write header
                uint64_t num_voxels = voxels.num_voxels;
                ofs.write(reinterpret_cast<const char*>(&num_voxels), sizeof(num_voxels));

                // Write positions
                ofs.write(reinterpret_cast<const char*>(voxels.positions.data()),
                         voxels.positions.size() * sizeof(float));

                // Write colors if available
                if (!voxels.colors.empty()) {
                    ofs.write(reinterpret_cast<const char*>(voxels.colors.data()),
                             voxels.colors.size() * sizeof(uint8_t));
                }

                ofs.close();
                std::cout << "  Saved voxels to: " << output_file << std::endl;
            }
        }

        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "ARM Mali GPU Example Completed!" << std::endl;
        std::cout << "========================================" << std::endl;

    } catch (const TaichiSLAMError& e) {
        std::cerr << "[ERROR] TaichiSLAM Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
