cmake_minimum_required(VERSION 3.14)
project(pyhnsw)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/../../include)

# Find Eigen
find_package(Eigen3 3.3 QUIET)
if(NOT Eigen3_FOUND)
    # If not found, try to use the one from the parent project
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
    )
    FetchContent_MakeAvailable(eigen)
    set(EIGEN3_INCLUDE_DIR ${eigen_SOURCE_DIR})
else()
    set(EIGEN3_INCLUDE_DIR ${EIGEN3_INCLUDE_DIRS})
endif()

# Create the Python module
pybind11_add_module(pyhnsw src/pyhnsw.cpp)

# Set C++ standard
target_compile_features(pyhnsw PRIVATE cxx_std_17)

# Include directories
target_include_directories(pyhnsw PRIVATE 
    ${CMAKE_SOURCE_DIR}/../../include
    ${EIGEN3_INCLUDE_DIR}
)

# Compiler optimizations
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(pyhnsw PRIVATE -O3 -march=native)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(pyhnsw PRIVATE /O2)
endif()

# Install target
install(TARGETS pyhnsw DESTINATION .)