# HNSW - Hierarchical Navigable Small Worlds
## Nearest neighbor search for vectorized embeddings

+ Single header, modern C++
+ Approximately 500 lines of code
+ Fast (and getting faster)

### Example

```
#include "hnsw/hsnw.h"

int main(int argc, char* argv[])
{
    // Create an index for 128-dimensional vectors
    dicroce::hnsw<float> index(128);
        
    const size_t num_vectors = 10000;
    std::vector<std::vector<float>> vectors(num_vectors) = generate_random_vv(num_vectors, 128);
    
    // Build the index    
    for (size_t i = 0; i < vectors.size(); ++i)
        index.add_item(vectors[i]);
    
    // Search for nearest neighbors
    std::vector<float> query(128) = generate_random_v(128);
    
    const size_t k = 10;
    auto results = index.search(query, k);
    
    for (const auto& result : results)
        printf("ID: %lu, Distance: %f\n", result.first, result.second);
}
```

### Compilation
mkdir build && pushd build && cmake .. && make && popd
