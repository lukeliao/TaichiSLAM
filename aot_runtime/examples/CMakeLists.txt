# TaichiSLAM AOT Examples

# Example 1: Basic Octomap Usage
add_executable(example_basic_octo basic_octo.cpp)
target_link_libraries(example_basic_octo PRIVATE taichislam_aot)
set_target_properties(example_basic_octo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)

# Example 2: Basic TSDF Usage (placeholder)
# add_executable(example_basic_tsdf basic_tsdf.cpp)
# target_link_libraries(example_basic_tsdf PRIVATE taichislam_aot)
# set_target_properties(example_basic_tsdf PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
# )

# Example 3: ARM Mali GPU Example
add_executable(example_arm_mali arm_mali_example.cpp)
target_link_libraries(example_arm_mali PRIVATE taichislam_aot)
set_target_properties(example_arm_mali PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples
)

# Copy AOT modules to examples directory
add_custom_target(copy_aot_modules ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/../aot_modules
        ${CMAKE_BINARY_DIR}/examples/aot_modules
    COMMENT "Copying AOT modules to examples directory"
)
