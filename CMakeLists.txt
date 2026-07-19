cmake_minimum_required(VERSION 3.14)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 1)
project(hnsw VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})

include(r_build.txt)

# hnsw is header-only and dependency-free: the distance kernels live in
# include/hnsw/simd_distance.h and use only <immintrin.h> / <intrin.h> (system
# headers). No Eigen, no FetchContent.
add_library(
    hnsw
    INTERFACE
)

target_include_directories(
    hnsw
    INTERFACE include
)

# The runtime-dispatch function pointers in simd_distance.h are C++17 `inline`
# variables, so every consumer of hnsw must compile as C++17 or newer.
target_compile_features(
    hnsw
    INTERFACE cxx_std_17
)

enable_testing()
add_subdirectory(ut)
add_subdirectory(bench)
