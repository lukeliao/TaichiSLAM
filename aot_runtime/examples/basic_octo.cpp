/**
 * TaichiSLAM AOT Example: Basic Octomap Usage
 *
 * This example demonstrates how to use the TaichiSLAM AOT runtime
 * to perform Octomap-based 3D mapping on GPU.
 *
 * Build and run:
 *   mkdir build && cd build
 *   cmake .. -DTAICHI_C_API_PATH=/path/to/taichi/c_api
 *   make -j$(nproc)
 *   ./examples/example_basic_octo
 */

#include <taichislam_aot.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using namespace taichislam;

// Generate synthetic point cloud (simulating a simple room)
void generateRoomPointCloud(
    std::vector<float>& points,
    std::vector<uint8_t>& colors,
    size_t num_points,
    float room_size = 5.0f) {

    points.resize(num_points * 3);
    colors.resize(num_points * 3);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_dist(-room_size/2, room_size/2);
    std::uniform_real_distribution<float> height_dist(0.0f, 3.0f);
    std::uniform_int_distribution<int> color_dist(0, 255);

    // Generate points forming walls, floor, and some objects
    for (size_t i = 0; i < num_points; ++i) {
        // Random point in room with some structure
        int surface = i % 6;  // 6 surfaces: floor, ceiling, 4 walls

        switch (surface) {
            case 0:  // Floor
                points[i*3 + 0] = pos_dist(gen);
                points[i*3 + 1] = pos_dist(gen);
                points[i*3 + 2] = 0.0f;
                break;
            case 1:  // Ceiling
                points[i*3 + 0] = pos_dist(gen);
                points[i*3 + 1] = pos_dist(gen);
                points[i*3 + 2] = 3.0f;
                break;
            default:  // Walls
                int wall = surface - 2;
                float wall_pos = (wall % 2 == 0) ? -room_size/2 : room_size/2;
                if (wall < 2) {
                    // X-walls
                    points[i*3 + 0] = wall_pos;
                    points[i*3 + 1] = pos_dist(gen);
                } else {
                    // Y-walls
                    points[i*3 + 0] = pos_dist(gen);
                    points[i*3 + 1] = wall_pos;
                }
                points[i*3 + 2] = height_dist(gen);
                break;
        }

        // Random color
        colors[i*3 + 0] = static_cast<uint8_t>(color_dist(gen));
        colors[i*3 + 1] = static_cast<uint8_t>(color_dist(gen));
        colors[i*3 + 2] = static_cast<uint8_t>(color_dist(gen));
    }
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "TaichiSLAM AOT Example: Basic Octomap" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Parse arguments
    std::string aot_module_path = "./aot_modules/taichislam_octo";
    if (argc > 1) {
        aot_module_path = argv[1];
    }

    try {
        //=========================================================================
        // Step 1: Initialize TaichiSLAM Runtime
        //=========================================================================
        std::cout << "[1/5] Initializing TaichiSLAM Runtime..." << std::endl;

        // Use Vulkan backend (supports ARM Mali GPU via Vulkan)
        // For OpenGL ES on ARM Mali, use TI_ARCH_GLES
        TiArch arch = TI_ARCH_VULKAN;

        TaichiSLAMRuntime runtime(arch, 0);  // Device 0

        std::cout << "      Runtime initialized successfully!" << std::endl;
        std::cout << std::endl;

        //=========================================================================
        // Step 2: Load AOT Module
        //=========================================================================
        std::cout << "[2/5] Loading AOT Module..." << std::endl;
        std::cout << "      Path: " << aot_module_path << std::endl;

        MappingConfig config;
        config.map_size_xy = 20.0f;
        config.map_size_z = 5.0f;
        config.voxel_scale = 0.1f;
        config.enable_texture = true;
        config.max_disp_particles = 100000;

        // Load Octomap AOT module
        std::shared_ptr<OctomapMapping> mapping = runtime.loadOctomap(
            aot_module_path, config);

        std::cout << "      AOT Module loaded successfully!" << std::endl;
        std::cout << "      Map size: " << config.map_size_xy << "m x "
                  << config.map_size_xy << "m x " << config.map_size_z << "m" << std::endl;
        std::cout << "      Voxel scale: " << config.voxel_scale << "m" << std::endl;
        std::cout << std::endl;

        //=========================================================================
        // Step 3: Generate Synthetic Point Cloud
        //=========================================================================
        std::cout << "[3/5] Generating synthetic point cloud..." << std::endl;

        size_t num_points = 10000;
        std::vector<float> points;
        std::vector<uint8_t> colors;

        generateRoomPointCloud(points, colors, num_points, 5.0f);

        std::cout << "      Generated " << num_points << " points" << std::endl;
        std::cout << std::endl;

        //=========================================================================
        // Step 4: Insert Point Cloud into Map
        //=========================================================================
        std::cout << "[4/5] Inserting point cloud into map..." << std::endl;

        // Create sensor pose (at origin, looking forward)
        Pose pose;
        pose.T[0] = 0.0f;  // X
        pose.T[1] = 0.0f;  // Y
        pose.T[2] = 1.0f;  // Z (1 meter above ground)

        // Run multiple iterations simulating a moving sensor
        int num_frames = 5;
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int frame = 0; frame < num_frames; ++frame) {
            // Move sensor in a circle
            float angle = 2.0f * M_PI * frame / num_frames;
            pose.T[0] = 2.0f * cosf(angle);
            pose.T[1] = 2.0f * sinf(angle);

            // Rotate to face center
            float c = cosf(angle + M_PI);
            float s = sinf(angle + M_PI);
            pose.R[0] = c;  pose.R[1] = -s;  pose.R[2] = 0;
            pose.R[3] = s;  pose.R[4] =  c;  pose.R[5] = 0;
            pose.R[6] = 0;  pose.R[7] =  0;  pose.R[8] = 1;

            // Insert point cloud
            mapping->insertPointCloud(
                points.data(),
                colors.data(),
                num_points,
                pose);

            std::cout << "      Frame " << (frame + 1) << "/" << num_frames
                      << " completed" << std::endl;
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        std::cout << "      Inserted " << num_frames << " frames in "
                  << duration.count() << " ms" << std::endl;
        std::cout << "      Average: " << (duration.count() / num_frames) << " ms/frame" << std::endl;
        std::cout << std::endl;

        //=========================================================================
        // Step 5: Extract Voxels
        //=========================================================================
        std::cout << "[5/5] Extracting occupied voxels..." << std::endl;

        VoxelGrid voxels = mapping->extractVoxels(0);

        std::cout << "      Extracted " << voxels.num_voxels << " voxels" << std::endl;

        // Get statistics
        size_t num_voxels;
        float map_size_mb;
        mapping->getStatistics(&num_voxels, &map_size_mb);

        std::cout << "      Map statistics:" << std::endl;
        std::cout << "        Total voxels: " << num_voxels << std::endl;
        std::cout << "        Memory usage: " << map_size_mb << " MB" << std::endl;

        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Example completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;

    } catch (const TaichiSLAMError& e) {
        std::cerr << "[ERROR] TaichiSLAM Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Standard Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[ERROR] Unknown exception occurred" << std::endl;
        return 1;
    }
}


// Print usage information
void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options] [AOT_MODULE_PATH]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help          Show this help message" << std::endl;
    std::cout << "  -v, --verbose       Enable verbose output" << std::endl;
    std::cout << std::endl;
    std::cout << "Arguments:" << std::endl;
    std::cout << "  AOT_MODULE_PATH     Path to AOT module directory (default: ./aot_modules/taichislam_octo)" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << std::endl;
    std::cout << "  " << program_name << " /path/to/aot_module" << std::endl;
}
