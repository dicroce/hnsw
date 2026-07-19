# HNSW - Hierarchical Navigable Small Worlds
## Nearest neighbor search for vector embeddings

+ Single header, modern C++.
+ Approximately 500 lines of code.
+ Dependency-free.
+ Fast: hand-written SIMD distance kernels (AVX2/FMA, SSE2, scalar) selected at
  runtime to match the CPU it runs on. See `include/hnsw/simd_distance.h`.

## What is HNSW?
```
// Level 2           *                    *        *                                        *
// Level 1        *  *              *     *        *        *          **                 * *
// Level 0  *     ** *     * * *    *    **   *  * *  * ** **          **     ** *    *   * *   * * *
```
HNSW is a graph structure that consists of levels that are more sparsely populated at the top and more
densely populated at the bottom. Nodes within a layer have connections to other nodes that are near them
on the same level. When a node is inserted a random level is picked and the node is inserted there. It
is also inserted into all levels beneath that level down to 0.

When searches arrive they start at the top and search that level (following connections) until they find
the closest node in the top level. The search then descends and keeps searching nearby nodes. As the
search progresses the code keeps track of the K nearest nodes it has seen. Eventually it either finds
the value OR it finds the closest value on level 0 and the K nearest nodes seen are returned.

### Example

Vectors are passed as `std::vector<scalar>` (e.g. `std::vector<float>`), which
is aliased as `dicroce::hnsw_types<float>::vector_type`.

```cpp
#include "hnsw/hnsw.h"
#include <random>
#include <cstdio>

int main()
{
    using vector_type = dicroce::hnsw_types<float>::vector_type;

    // Create an index for 128-dimensional vectors
    const size_t dim = 128;
    dicroce::hnsw<float> index(dim);

    std::mt19937 gen(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto random_vec = [&]() {
        vector_type v(dim);
        for (size_t i = 0; i < dim; ++i) v[i] = nd(gen);
        return v;
    };

    // Build the index
    const size_t num_vectors = 10000;
    for (size_t i = 0; i < num_vectors; ++i)
        index.add_item(random_vec());

    // Search for the k nearest neighbors of a query vector
    vector_type query = random_vec();
    const size_t k = 10;
    auto results = index.search(query, k);

    for (const auto& result : results)
        printf("label: %lld, distance: %f\n",
               (long long)result.first, result.second);
}
```

### External IDs (labels)

`search` returns `(label, distance)` pairs. By default a vector's label is its
insertion order (0, 1, 2, …), but you can attach your own `int64` label so hits
map straight back to your data (e.g. a frame id, a database row, a nanots
`(timestamp)` key):

```cpp
index.add_item(embedding, /*label=*/frame_id);   // your id
...
auto hits = index.search(query, 10);
for (auto& [label, dist] : hits)
    lookup_frame(label);   // label is the frame_id you supplied
```

Labels need not be unique — that's your choice — and they are persisted by
`save()`/`load()`.

### Persistence

The whole graph (vectors *and* connections) is serialized to a flat binary
file, so loading reconstructs a ready-to-search index **without rebuilding** —
construction is the expensive part of HNSW, so you only pay it once.

```cpp
// Save a built index.
index.save("index.bin");

// Load it back later (returns a ready-to-search index; no rebuild).
auto index = dicroce::hnsw<float>::load("index.bin");
auto results = index.search(query, 10);
```

The on-disk format is self-describing (magic + version + byte-order marker) and
records the `sizeof(scalar)`, so loading a file written for `hnsw<double>` into
an `hnsw<float>` — or a file written on a different-endian machine — fails
loudly instead of corrupting silently.

### Compilation
mkdir build && pushd build && cmake .. && make && popd
