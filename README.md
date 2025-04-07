# HNSW - Hierarchical Navigable Small Worlds
## Nearest neighbor search for vector embeddings

+ Single header, modern C++
+ Approximately 500 lines of code
+ Fast (and getting faster)

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
