cmake_minimum_required(VERSION 3.14)
project(pyhnsw)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../../include)

# Create the Python module (hnsw is header-only and dependency-free now)
pybind11_add_module(pyhnsw src/pyhnsw.cpp)

# Set C++ standard
target_compile_features(pyhnsw PRIVATE cxx_std_17)

# Include directories
target_include_directories(pyhnsw PRIVATE
    ${CMAKE_SOURCE_DIR}/../../include
)

# Compiler optimizations.
#
# IMPORTANT: build at a portable BASELINE arch -- do NOT use -march=native or
# /arch:AVX2 here. This binary is distributed as a wheel and must run on any
# x86-64 CPU. The AVX2 kernels in simd_distance.h are compiled via a per-function
# target attribute (GCC/Clang) or always-available intrinsics (MSVC), and are
# selected at RUNTIME only when the executing CPU supports them. A global
# -march=native would bake the build machine's ISA into every code path and
# crash (illegal instruction) on older CPUs.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pyhnsw PRIVATE -O3)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(pyhnsw PRIVATE /O2)
endif()

# Install target
install(TARGETS pyhnsw DESTINATION .)