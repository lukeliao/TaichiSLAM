cmake_minimum_required(VERSION 3.15)
project(TaichiSLAMAOT VERSION 0.1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find Python and Taichi
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Get Taichi installation path
if(NOT TAICHI_PACKAGE_DIR)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import taichi; print(taichi.__path__[0])"
        OUTPUT_VARIABLE TAICHI_PACKAGE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()

# Default path if not found
if(NOT TAICHI_PACKAGE_DIR OR NOT EXISTS ${TAICHI_PACKAGE_DIR})
    set(TAICHI_PACKAGE_DIR "/home/liao/.local/lib/python3.12/site-packages/taichi")
endif()

if(NOT EXISTS ${TAICHI_PACKAGE_DIR})
    message(FATAL_ERROR "Taichi not found at ${TAICHI_PACKAGE_DIR}. Please install Taichi: pip install taichi")
endif()

message(STATUS "Taichi package directory: ${TAICHI_PACKAGE_DIR}")

# Taichi include directory
set(TAICHI_INCLUDE_DIR "${TAICHI_PACKAGE_DIR}/include")
if(NOT EXISTS ${TAICHI_INCLUDE_DIR})
    # Try alternative locations
    set(TAICHI_INCLUDE_DIR "${TAICHI_PACKAGE_DIR}/_lib/c_api/include")
endif()

if(NOT EXISTS ${TAICHI_INCLUDE_DIR})
    message(WARNING "Taichi include directory not found")
    # Try to find it
    find_path(TAICHI_INCLUDE_DIR taichi/taichi_core.h
        PATHS ${TAICHI_PACKAGE_DIR}
        DOC "Taichi include directory"
    )
endif()

message(STATUS "Taichi include directory: ${TAICHI_INCLUDE_DIR}")

# Taichi library directory
set(TAICHI_LIB_DIR "${TAICHI_PACKAGE_DIR}/_lib")
if(NOT EXISTS ${TAICHI_LIB_DIR})
    # Try to find taichi_core library
    find_library(TAICHI_CORE_LIB taichi_core
        PATHS ${TAICHI_PACKAGE_DIR}
        PATH_SUFFIXES _lib lib
        DOC "Taichi core library"
    )
    if(TAICHI_CORE_LIB)
        get_filename_component(TAICHI_LIB_DIR ${TAICHI_CORE_LIB} DIRECTORY)
    endif()
else()
    set(TAICHI_CORE_LIB "${TAICHI_LIB_DIR}/libtaichi_core.so")
endif()

message(STATUS "Taichi library directory: ${TAICHI_LIB_DIR}")
message(STATUS "Taichi core library: ${TAICHI_CORE_LIB}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TAICHI_INCLUDE_DIR}
)

# Library directories
link_directories(
    ${TAICHI_LIB_DIR}
)

# Source files
set(TAICHISLAM_SOURCES
    src/taichislam_aot.cpp
)

# Create library
add_library(taichislam_aot ${TAICHISLAM_SOURCES})

target_link_libraries(taichislam_aot
    ${TAICHI_CORE_LIB}
)

# Set RPATH for finding shared libraries
set_target_properties(taichislam_aot PROPERTIES
    INSTALL_RPATH ${TAICHI_LIB_DIR}
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Examples
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Print summary
message(STATUS "")
message(STATUS "TaichiSLAM AOT Runtime Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Taichi: ${TAICHI_PACKAGE_DIR}")
message(STATUS "  Examples: ${BUILD_EXAMPLES}")
message(STATUS "")
