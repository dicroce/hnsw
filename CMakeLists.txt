cmake_minimum_required(VERSION 3.14)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 1)
project(hnsw VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})

include(r_build.txt)
include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    # Eigen is header-only and we only consume its headers. Point SOURCE_SUBDIR
    # at a non-existent directory so FetchContent downloads the source but does
    # NOT add_subdirectory() Eigen's own CMake project. That project otherwise
    # registers ~900 ctest tests (which aren't built, so `ctest` fails) and can
    # fail to configure under some toolchains.
    SOURCE_SUBDIR do-not-build
)

FetchContent_MakeAvailable(eigen)

add_library(
    hnsw
    INTERFACE
)

target_include_directories(
    hnsw
    INTERFACE include
    ${eigen_SOURCE_DIR}
)

enable_testing()
add_subdirectory(ut)
