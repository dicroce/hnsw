
add_executable(
    ut
    include/framework.h
    source/framework.cpp
    include/test_hnsw.h
    source/test_hnsw.cpp
)

target_include_directories(
    ut PUBLIC
    include
)

target_link_libraries(
    ut PUBLIC
    hnsw
)
