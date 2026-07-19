# Standalone SIMD micro-benchmark / learning tool. Links the header-only hnsw
# interface target, which supplies the include path and the C++17 requirement.
#
# NOTE: no -march=native / /arch:AVX2 here on purpose. The point of the benchmark
# is to show the AVX2 kernel being picked at RUNTIME from a baseline build.
add_executable(bench_simd bench_simd.cpp)
target_link_libraries(bench_simd PRIVATE hnsw)
