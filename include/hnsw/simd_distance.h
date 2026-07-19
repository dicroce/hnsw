#ifndef __hnsw_simd_distance_h
#define __hnsw_simd_distance_h

// ============================================================================
//  SIMD distance kernels for HNSW  (fp32, with a scalar path for everything else)
// ============================================================================
//
//  WHAT IS SIMD?
//  -------------
//  SIMD = "Single Instruction, Multiple Data". A normal `float` add works on
//  one number at a time. A SIMD add works on a whole *vector* of numbers in one
//  instruction -- 4, 8, or 16 floats at once -- using special wide registers.
//  Each of the values packed into a register lives in its own "lane".
//
//      register width      floats per register      instruction set
//      --------------      -------------------      ---------------
//        128 bits                  4                 SSE / SSE2
//        256 bits                  8                 AVX / AVX2
//        512 bits                 16                 AVX-512   (not used here)
//
//  We write to the CPU through "intrinsics": C functions that map ~1:1 onto a
//  single machine instruction. `__m256` is the C type for a 256-bit register
//  holding 8 floats. `_mm256_add_ps(a, b)` adds two such registers lane-by-lane
//  ("ps" = "packed single-precision", i.e. packed floats).
//
//  WHAT ARE WE ACCELERATING?
//  -------------------------
//  HNSW spends ~80-95% of its time computing distances between vectors. There
//  are exactly two kernels that matter:
//
//      L2 squared:   sum over i of (a[i] - b[i])^2
//      dot product:  sum over i of  a[i] * b[i]
//
//  Both are the same shape: walk two arrays, do a little arithmetic per element,
//  and ACCUMULATE into a running sum (a "reduction"). SIMD does the per-element
//  arithmetic 8 lanes at a time; at the very end we add the 8 partial sums
//  together into one scalar (a "horizontal reduction").
//
//  RUNTIME DISPATCH (why this file is structured the way it is)
//  -----------------------------------------------------------
//  This library ships as a prebuilt Python wheel. We cannot assume the CPU that
//  RUNS the wheel is the CPU that BUILT it. So we compile *all* the kernels
//  (scalar, SSE2, AVX2) into the binary, then at load time ask the CPU what it
//  supports and pick the best one. See Part 5 (detection) and Part 6 (dispatch).
//
//  Reading order for learning: Part 1 (scalar, the ground truth) -> Part 2/3
//  (SSE2, the gentle 4-wide intro) -> Part 4 (AVX2, 8-wide + FMA) -> Part 5/6
//  (how we choose one at runtime) -> Part 7 (the public entry points).
// ============================================================================

#include <cstddef>   // size_t
#include <cstdint>
#include <cmath>     // std::sqrt

// ---- Platform detection ----------------------------------------------------
// We only have hand-written SIMD for x86 / x86-64. On any other CPU (e.g. ARM)
// the public API below falls back to the scalar kernels, so the library still
// builds and runs correctly everywhere -- just without vectorization there yet.
// (An ARM NEON path would slot in exactly where the SSE2/AVX2 ones live.)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #define HNSW_SIMD_X86 1
    // <immintrin.h> declares every x86 SIMD intrinsic (SSE..AVX-512). On
    // GCC/Clang you may *declare* them without -mavx2; you just can't *use* the
    // AVX2 ones outside a function marked with the avx2 target attribute (below).
    #include <immintrin.h>
    #if defined(_MSC_VER)
        // MSVC keeps the CPUID query intrinsics (__cpuid/__cpuidex) in <intrin.h>,
        // separate from the SIMD math intrinsics above. (On GCC/Clang we use
        // __builtin_cpu_supports instead, so this header is MSVC-only.)
        #include <intrin.h>
    #endif
#endif

// On GCC/Clang, this attribute compiles a single function with AVX2+FMA enabled
// even though the rest of the translation unit is built for a baseline CPU.
// MSVC has no such attribute -- it always lets you emit AVX2 intrinsics -- so we
// define the macro to nothing there.
#if defined(HNSW_SIMD_X86) && (defined(__GNUC__) || defined(__clang__))
    #define HNSW_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
    #define HNSW_TARGET_AVX2
#endif

namespace dicroce {
namespace simd {

// ============================================================================
//  PART 1 -- Scalar reference kernels
// ----------------------------------------------------------------------------
//  These are the "ground truth". They are simple, obviously-correct, and work
//  on every CPU and every scalar type (float AND double -- note the template).
//  The SIMD kernels below must produce (nearly) the same answer as these; the
//  benchmark asserts exactly that. When in doubt, trust these.
//
//  "Nearly" the same, not bit-identical: floating-point addition is not
//  associative, and SIMD sums the elements in a different ORDER (lane by lane,
//  then combined) than this left-to-right loop. Different order -> different
//  rounding -> tiny differences. That's expected and harmless for nearest-
//  neighbor search.
// ============================================================================

template <typename T>
inline T l2_squared_scalar(const T* a, const T* b, size_t n)
{
    T sum = T(0);
    for (size_t i = 0; i < n; ++i)
    {
        const T d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

template <typename T>
inline T dot_scalar(const T* a, const T* b, size_t n)
{
    T sum = T(0);
    for (size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}

#if defined(HNSW_SIMD_X86)

// ============================================================================
//  PART 2 -- Horizontal reduction helpers
// ----------------------------------------------------------------------------
//  After the main loop, a SIMD accumulator holds several PARTIAL sums (one per
//  lane) that we still need to add together into a single float. Adding across
//  the lanes of one register is called a "horizontal" add (as opposed to the
//  "vertical", lane-parallel adds we do in the loop). Horizontal ops are
//  relatively slow, so we do them ONCE at the very end -- never inside the loop.
// ============================================================================

// Sum the 4 lanes of a 128-bit register into one float.
// This is a well-known SSE idiom -- it's fine to take the shuffle magic on faith
// at first. The comments trace exactly what lands in lane 0 (the lane we read).
//
// Notation: v = [v0, v1, v2, v3] with v0 in the low lane.
static inline float hsum128(__m128 v)
{
    // Swap the two floats within each pair: [v1, v0, v3, v2].
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
    // Lane-wise add -> [v0+v1, v0+v1, v2+v3, v2+v3].
    __m128 sums = _mm_add_ps(v, shuf);
    // Move the high 64 bits (lanes 2,3 = v2+v3) down into the low lane.
    shuf = _mm_movehl_ps(shuf, sums);
    // Add just the low lane: (v0+v1) + (v2+v3) = total. add_ss = "add scalar".
    sums = _mm_add_ss(sums, shuf);
    // Extract lane 0 as a plain float.
    return _mm_cvtss_f32(sums);
}

// ============================================================================
//  PART 3 -- SSE2 kernels (128-bit, 4 floats per step)
// ----------------------------------------------------------------------------
//  SSE2 is guaranteed present on EVERY x86-64 CPU, so this is our safe fallback
//  when AVX2 is missing. It's also the gentlest introduction: only 4 lanes.
//
//  The shape of every SIMD reduction is the same three parts:
//    (1) main loop: process a full register width per iteration,
//    (2) horizontal reduce the accumulator to one scalar,
//    (3) scalar "tail" loop: handle the leftover elements when n isn't a
//        multiple of the width (e.g. the last 3 floats of a length-131 vector).
// ============================================================================

static inline float l2_squared_sse2(const float* a, const float* b, size_t n)
{
    __m128 acc = _mm_setzero_ps();          // 4 running partial sums, all 0
    size_t i = 0;

    // (1) main loop -- 4 floats at a time
    for (; i + 4 <= n; i += 4)
    {
        __m128 va = _mm_loadu_ps(a + i);    // load 4 floats from a (u = unaligned)
        __m128 vb = _mm_loadu_ps(b + i);    // load 4 floats from b
        __m128 d  = _mm_sub_ps(va, vb);     // d   = a - b        (per lane)
        acc = _mm_add_ps(acc, _mm_mul_ps(d, d)); // acc += d * d  (per lane)
    }

    // (2) collapse the 4 partial sums into one
    float result = hsum128(acc);

    // (3) tail: the last (n % 4) elements
    for (; i < n; ++i)
    {
        const float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

static inline float dot_sse2(const float* a, const float* b, size_t n)
{
    __m128 acc = _mm_setzero_ps();
    size_t i = 0;

    for (; i + 4 <= n; i += 4)
    {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb)); // acc += a * b
    }

    float result = hsum128(acc);
    for (; i < n; ++i)
        result += a[i] * b[i];
    return result;
}

// ============================================================================
//  PART 4 -- AVX2 + FMA kernels (256-bit, 8 floats per step)
// ----------------------------------------------------------------------------
//  Two new ideas here beyond "just wider registers":
//
//  FMA ("fused multiply-add"): `_mm256_fmadd_ps(x, y, z)` computes x*y + z in a
//  SINGLE instruction with a SINGLE rounding step. It's both faster (one op
//  instead of two) and slightly more accurate (no intermediate rounding) than a
//  separate multiply then add. Note: AVX gives you the 256-bit float ops; FMA is
//  a *separate* CPU feature that happens to arrive alongside AVX2 on real chips.
//
//  MULTIPLE ACCUMULATORS (instruction-level parallelism): an FMA has latency --
//  the result of `acc += d*d` isn't ready for a few cycles. If the next
//  iteration reads the SAME acc, the CPU must stall waiting for it: a
//  "dependency chain". By keeping TWO independent accumulators and interleaving
//  them, the CPU can work on both in parallel and hide that latency. We combine
//  the two at the end. (More accumulators help more, up to a point; two is a
//  good, readable default.)
// ============================================================================

// FMA helper: use the real fused instruction when FMA is compiled in, otherwise
// fall back to a plain multiply + add so the code still works. On GCC/Clang the
// avx2 target attribute enables __FMA__ for this function; on MSVC the fmadd
// intrinsic is always available.
HNSW_TARGET_AVX2 static inline __m256 hnsw_fmadd(__m256 x, __m256 y, __m256 z)
{
#if defined(__FMA__) || defined(_MSC_VER)
    return _mm256_fmadd_ps(x, y, z);              // x*y + z, one rounding
#else
    return _mm256_add_ps(_mm256_mul_ps(x, y), z); // x*y then + z, two roundings
#endif
}

// Sum the 8 lanes of a 256-bit register into one float, by folding the upper
// 128 bits onto the lower 128 bits and reusing the SSE reduction.
HNSW_TARGET_AVX2 static inline float hsum256(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);     // lower 4 floats (free -- just a view)
    __m128 hi = _mm256_extractf128_ps(v, 1);   // upper 4 floats
    return hsum128(_mm_add_ps(lo, hi));        // add the halves, then reduce 4->1
}

HNSW_TARGET_AVX2 static inline float l2_squared_avx2(const float* a, const float* b, size_t n)
{
    __m256 acc0 = _mm256_setzero_ps();   // two independent accumulators
    __m256 acc1 = _mm256_setzero_ps();   // (see "multiple accumulators" above)
    size_t i = 0;

    // main loop -- 16 floats per iteration (two 8-wide lanes, interleaved)
    for (; i + 16 <= n; i += 16)
    {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 d0 = _mm256_sub_ps(a0, b0);
        acc0 = hnsw_fmadd(d0, d0, acc0);          // acc0 += d0 * d0

        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        __m256 d1 = _mm256_sub_ps(a1, b1);
        acc1 = hnsw_fmadd(d1, d1, acc1);          // acc1 += d1 * d1
    }

    // handle a leftover group of 8 (when n % 16 is >= 8)
    for (; i + 8 <= n; i += 8)
    {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 d0 = _mm256_sub_ps(a0, b0);
        acc0 = hnsw_fmadd(d0, d0, acc0);
    }

    // combine the two accumulators, then reduce 8 lanes -> 1 scalar
    float result = hsum256(_mm256_add_ps(acc0, acc1));

    // scalar tail: the last (n % 8) elements
    for (; i < n; ++i)
    {
        const float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

HNSW_TARGET_AVX2 static inline float dot_avx2(const float* a, const float* b, size_t n)
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 16 <= n; i += 16)
    {
        acc0 = hnsw_fmadd(_mm256_loadu_ps(a + i),     _mm256_loadu_ps(b + i),     acc0);
        acc1 = hnsw_fmadd(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8)
        acc0 = hnsw_fmadd(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);

    float result = hsum256(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i)
        result += a[i] * b[i];
    return result;
}

// ============================================================================
//  PART 5 -- CPU feature detection
// ----------------------------------------------------------------------------
//  At runtime we must ask THIS machine whether it can run the AVX2 kernels.
//  It's not enough that the CPU has AVX2 -- the OPERATING SYSTEM must also
//  promise to save/restore the wide (YMM) registers across context switches.
//  If it doesn't, using AVX would silently corrupt other programs' state. That's
//  why the MSVC path checks OSXSAVE + XGETBV, not just the AVX2 CPU bit.
//
//  GCC/Clang wrap all of that in __builtin_cpu_supports(), so their path is one
//  line. We show the full MSVC version so you can see what's really going on.
// ============================================================================

inline bool cpu_supports_avx2()
{
#if defined(__GNUC__) || defined(__clang__)
    // The builtin handles CPUID *and* the OS-support (XGETBV) check for us.
    // (On some GCC versions __builtin_cpu_init() must run first; it is safe to
    // call, and the compiler also runs it implicitly before the first query.)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");

#elif defined(_MSC_VER)
    int regs[4];

    // CPUID leaf 0 in EAX tells us the highest leaf number available.
    __cpuid(regs, 0);
    const int max_leaf = regs[0];
    if (max_leaf < 7) return false;              // AVX2 lives in leaf 7

    // Leaf 1: ECX bit 27 = OSXSAVE (OS enabled XGETBV), bit 28 = AVX,
    //         ECX bit 12 = FMA.
    __cpuidex(regs, 1, 0);
    const bool osxsave = (regs[2] & (1 << 27)) != 0;
    const bool avx     = (regs[2] & (1 << 28)) != 0;
    const bool fma     = (regs[2] & (1 << 12)) != 0;
    if (!osxsave || !avx || !fma) return false;

    // Ask the OS (via XGETBV of control register XCR0) whether it actually
    // saves the SSE (bit 1) and AVX/YMM (bit 2) register state. Both must be set.
    const unsigned long long xcr0 = _xgetbv(0 /* _XCR_XFEATURE_ENABLED_MASK */);
    if ((xcr0 & 0x6) != 0x6) return false;

    // Leaf 7, sub-leaf 0: EBX bit 5 = AVX2.
    __cpuidex(regs, 7, 0);
    const bool avx2 = (regs[1] & (1 << 5)) != 0;
    return avx2;

#else
    return false;   // unknown compiler -> play it safe
#endif
}

// ============================================================================
//  PART 6 -- Runtime dispatch
// ----------------------------------------------------------------------------
//  We resolve, ONCE, a function pointer to the best available kernel, and route
//  every distance call through it. Why a function pointer and not an `if` on
//  every call? Because the choice never changes after startup, so:
//    * the pointer is set one time (at program load), and
//    * the indirect call target is constant, so the CPU's branch predictor
//      nails it every time -- the overhead is negligible next to the SIMD loop.
//
//  `inline` variables (C++17) let us DEFINE these globals in a header without
//  violating the one-definition rule: the linker merges the duplicates. They are
//  initialized during static init (before main), so cpu_supports_avx2() runs
//  exactly once, at load time.
// ============================================================================

using dist_fn = float (*)(const float*, const float*, size_t);

inline dist_fn resolve_l2_f32()
{
    if (cpu_supports_avx2()) return &l2_squared_avx2;
    return &l2_squared_sse2;          // SSE2 is always present on x86-64
}
inline dist_fn resolve_dot_f32()
{
    if (cpu_supports_avx2()) return &dot_avx2;
    return &dot_sse2;
}

inline dist_fn g_l2_f32  = resolve_l2_f32();
inline dist_fn g_dot_f32 = resolve_dot_f32();

// A human-readable name for whichever path we chose -- handy for logging/tests.
inline const char* active_backend()
{
    return cpu_supports_avx2() ? "avx2+fma" : "sse2";
}

#else  // ---- not x86: no hand-written SIMD, report the scalar backend --------

inline const char* active_backend() { return "scalar"; }

#endif  // HNSW_SIMD_X86

// ============================================================================
//  PART 7 -- Public API
// ----------------------------------------------------------------------------
//  This is all the rest of the library calls. Overloading gives us:
//    * float  -> dispatched SIMD on x86, scalar elsewhere,
//    * double -> scalar (fp32 is what we vectorized; double stays correct and
//                portable via the templated reference kernels).
//  A non-template overload is preferred over the template during overload
//  resolution, so `l2_squared(const float*, ...)` wins for float arguments.
// ============================================================================

inline float l2_squared(const float* a, const float* b, size_t n)
{
#if defined(HNSW_SIMD_X86)
    return g_l2_f32(a, b, n);
#else
    return l2_squared_scalar(a, b, n);
#endif
}

inline float dot(const float* a, const float* b, size_t n)
{
#if defined(HNSW_SIMD_X86)
    return g_dot_f32(a, b, n);
#else
    return dot_scalar(a, b, n);
#endif
}

// double (and any other scalar type) -> portable reference kernels.
inline double l2_squared(const double* a, const double* b, size_t n) { return l2_squared_scalar(a, b, n); }
inline double dot       (const double* a, const double* b, size_t n) { return dot_scalar(a, b, n); }

// Euclidean norm (length) of a vector: sqrt(dot(v, v)). Used off the hot path
// for cosine normalization, so it just reuses the dispatched dot product.
template <typename T>
inline T norm(const T* v, size_t n) { return std::sqrt(dot(v, v, n)); }

} // namespace simd
} // namespace dicroce

#endif // __hnsw_simd_distance_h
