cmake_minimum_required(VERSION 3.14)
set(PROJECT_VERSION_MAJOR 0)
set(PROJECT_VERSION_MINOR 0)
set(PROJECT_VERSION_PATCH 1)
project(hnsw VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH})

include(r_build.txt)

add_library(
    hnsw
    INTERFACE
)

target_include_directories(
    hnsw
    INTERFACE include
)

add_subdirectory(ut)
