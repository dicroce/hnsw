// ============================================================================
//  bench_simd.cpp -- correctness + speed check for hnsw/simd_distance.h
// ----------------------------------------------------------------------------
//  A standalone micro-benchmark and learning tool for the SIMD distance
//  kernels. It (1) proves every kernel matches the scalar reference and
//  (2) times scalar vs SSE2 vs AVX2 across a range of dimensions.
//
//  Built as part of the CMake project (target `bench_simd`):
//
//      cmake -B build && cmake --build build --target bench_simd
//      ./build/bench/bench_simd        (or build\bench\Release\bench_simd.exe)
//
//  Or compile it directly. IMPORTANT: build at a *baseline* arch (no
//  -march=native / no /arch:AVX2) -- runtime dispatch is the whole point, and
//  we want to prove the AVX2 kernel is selected and correct WITHOUT the whole
//  program being compiled for AVX2.
//
//      MSVC (Developer Command Prompt):
//          cl /O2 /EHsc /std:c++17 /I ..\include bench_simd.cpp
//      clang / gcc:
//          g++ -O3 -std=c++17 -I ../include bench_simd.cpp -o bench_simd
// ============================================================================

#include "hnsw/simd_distance.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>

using namespace dicroce;

// Compare a dispatched/SIMD result against the scalar ground truth. We allow a
// small RELATIVE error because SIMD sums in a different order (see the comment
// in Part 1 of the header). ~1e-4 is comfortably loose for fp32 reductions.
static bool close(float got, float ref)
{
    const float denom = std::max(1.0f, std::fabs(ref));
    return std::fabs(got - ref) / denom < 1e-4f;
}

// Time a batch of calls and divide -> nanoseconds per single kernel call. We
// run many calls to swamp timer/clock overhead.
template <typename Fn>
static double time_ns_per_call(Fn&& fn, int calls)
{
    // warm up (let the CPU ramp, populate caches)
    volatile float sink = 0.0f;
    for (int i = 0; i < calls / 10 + 1; ++i) sink += fn();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < calls; ++i) sink += fn();
    auto t1 = std::chrono::high_resolution_clock::now();
    (void)sink;

    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / calls;
}

int main()
{
    std::printf("selected SIMD backend: %s\n\n", simd::active_backend());

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const size_t dims[] = {96, 128, 256, 512, 768, 1000};

    std::printf("%-6s | %-28s | %-28s | %-9s\n",
                "dim", "L2  scalar / sse2 / avx2 (ns)", "dot scalar / sse2 / avx2 (ns)", "L2 speedup");
    std::printf("-------+------------------------------+------------------------------+----------\n");

    bool all_ok = true;

    for (size_t n : dims)
    {
        std::vector<float> a(n), b(n);
        for (size_t i = 0; i < n; ++i) { a[i] = dist(rng); b[i] = dist(rng); }
        const float* pa = a.data();
        const float* pb = b.data();

        // ---- correctness: every path must match the scalar reference ----
        const float l2_ref = simd::l2_squared_scalar(pa, pb, n);
        const float dt_ref = simd::dot_scalar(pa, pb, n);

#if defined(HNSW_SIMD_X86)
        bool ok = close(simd::l2_squared_sse2(pa, pb, n), l2_ref)
               && close(simd::dot_sse2      (pa, pb, n), dt_ref)
               && close(simd::l2_squared_avx2(pa, pb, n), l2_ref)
               && close(simd::dot_avx2      (pa, pb, n), dt_ref)
               && close(simd::l2_squared    (pa, pb, n), l2_ref)   // public dispatched API
               && close(simd::dot           (pa, pb, n), dt_ref);
#else
        bool ok = close(simd::l2_squared(pa, pb, n), l2_ref)
               && close(simd::dot       (pa, pb, n), dt_ref);
#endif
        if (!ok) { all_ok = false; std::printf("dim %zu: CORRECTNESS FAILURE\n", n); }

        // ---- timing ----
        const int CALLS = 2000000;
        double l2_sc = time_ns_per_call([&]{ return simd::l2_squared_scalar(pa, pb, n); }, CALLS);
        double dt_sc = time_ns_per_call([&]{ return simd::dot_scalar       (pa, pb, n); }, CALLS);
#if defined(HNSW_SIMD_X86)
        double l2_se = time_ns_per_call([&]{ return simd::l2_squared_sse2(pa, pb, n); }, CALLS);
        double dt_se = time_ns_per_call([&]{ return simd::dot_sse2       (pa, pb, n); }, CALLS);
        double l2_av = time_ns_per_call([&]{ return simd::l2_squared_avx2(pa, pb, n); }, CALLS);
        double dt_av = time_ns_per_call([&]{ return simd::dot_avx2       (pa, pb, n); }, CALLS);
#else
        double l2_se = 0, dt_se = 0, l2_av = 0, dt_av = 0;
#endif

        char l2col[64], dtcol[64];
        std::snprintf(l2col, sizeof l2col, "%6.1f / %6.1f / %6.1f", l2_sc, l2_se, l2_av);
        std::snprintf(dtcol, sizeof dtcol, "%6.1f / %6.1f / %6.1f", dt_sc, dt_se, dt_av);
        std::printf("%-6zu | %-28s | %-28s | %6.2fx\n",
                    n, l2col, dtcol, l2_av > 0 ? l2_sc / l2_av : 0.0);
    }

    std::printf("\n%s\n", all_ok ? "ALL CORRECTNESS CHECKS PASSED" : "*** CORRECTNESS FAILURES ***");
    return all_ok ? 0 : 1;
}
