# pyhnsw - Python Bindings for HNSW

Fast approximate nearest neighbor search using Hierarchical Navigable Small World (HNSW) graphs.

## Features

- **Fast**: SIMD-accelerated distance calculations using Eigen
- **Simple**: Clean Python API similar to popular ANN libraries
- **Flexible**: Supports both L2 and cosine distance metrics
- **Efficient**: Batch operations for adding and searching multiple vectors
- **Type Support**: Both single (float32) and double (float64) precision

## Installation

### From Source

First, ensure you have the required dependencies:

```bash
pip install numpy pybind11
```

Then build and install the package:

```bash
cd bindings/pyhnsw
pip install .
```

### Build with CMake (Alternative)

```bash
cd bindings/pyhnsw
mkdir build && cd build
cmake ..
make
```

## Quick Start

```python
import numpy as np
import pyhnsw

# Create an index for 128-dimensional vectors
index = pyhnsw.HNSW(dim=128, M=16, ef_construction=200, ef_search=100, metric="l2")

# Generate some random data
data = np.random.randn(10000, 128).astype(np.float32)

# Add vectors to the index
index.add_items(data)

# Search for nearest neighbors
query = np.random.randn(128).astype(np.float32)
indices, distances = index.search(query, k=10)

print(f"Found {len(indices)} nearest neighbors")
print(f"Indices: {indices}")
print(f"Distances: {distances}")
```

## API Reference

### HNSW Class

```python
pyhnsw.HNSW(dim, M=16, ef_construction=200, ef_search=100, metric="l2")
```

Creates a new HNSW index.

**Parameters:**
- `dim` (int): Dimensionality of the vectors
- `M` (int): Maximum number of connections per element (default: 16)
- `ef_construction` (int): Size of the dynamic candidate list during construction (default: 200)
- `ef_search` (int): Size of the dynamic candidate list during search (default: 100)
- `metric` (str): Distance metric to use ('l2' or 'cosine', default: 'l2')

### Methods

#### add_item(vec)
Add a single vector to the index.

**Parameters:**
- `vec` (numpy.ndarray): 1D array of shape (dim,) containing the vector

#### add_items(data)
Add multiple vectors to the index.

**Parameters:**
- `data` (numpy.ndarray): 2D array of shape (n_items, dim) containing the vectors

#### search(query, k=10)
Search for k nearest neighbors of a query vector.

**Parameters:**
- `query` (numpy.ndarray): 1D array of shape (dim,) containing the query vector
- `k` (int): Number of nearest neighbors to return (default: 10)

**Returns:**
- `indices` (numpy.ndarray): 1D array of shape (k,) containing the indices of nearest neighbors
- `distances` (numpy.ndarray): 1D array of shape (k,) containing the distances to nearest neighbors

#### batch_search(queries, k=10)
Search for k nearest neighbors of multiple query vectors.

**Parameters:**
- `queries` (numpy.ndarray): 2D array of shape (n_queries, dim) containing the query vectors
- `k` (int): Number of nearest neighbors to return (default: 10)

**Returns:**
- `indices` (numpy.ndarray): 2D array of shape (n_queries, k) containing the indices
- `distances` (numpy.ndarray): 2D array of shape (n_queries, k) containing the distances

#### size()
Get the number of items in the index.

#### dim()
Get the dimensionality of vectors in the index.

## Examples

### Using Cosine Similarity

```python
import numpy as np
import pyhnsw

# Create index with cosine metric
index = pyhnsw.HNSW(dim=512, metric="cosine")

# For cosine similarity, it's recommended to normalize vectors
data = np.random.randn(1000, 512).astype(np.float32)
data = data / np.linalg.norm(data, axis=1, keepdims=True)

index.add_items(data)

# Search with normalized query
query = np.random.randn(512).astype(np.float32)
query = query / np.linalg.norm(query)

indices, distances = index.search(query, k=5)
```

### Batch Processing

```python
import numpy as np
import pyhnsw

# Create index
index = pyhnsw.HNSW(dim=256)

# Add data in batches
batch_size = 1000
for i in range(0, 10000, batch_size):
    batch = np.random.randn(batch_size, 256).astype(np.float32)
    index.add_items(batch)

# Batch search
queries = np.random.randn(100, 256).astype(np.float32)
indices, distances = index.batch_search(queries, k=20)

# Process results
for i, (idx, dist) in enumerate(zip(indices, distances)):
    print(f"Query {i}: Found {len(idx)} neighbors")
```

### Double Precision

```python
import numpy as np
import pyhnsw

# Use HNSWDouble for float64 precision
index = pyhnsw.HNSWDouble(dim=128)

# Use float64 arrays
data = np.random.randn(1000, 128).astype(np.float64)
index.add_items(data)

query = np.random.randn(128).astype(np.float64)
indices, distances = index.search(query, k=10)
```

## Performance Tips

1. **Parameter Tuning**:
   - Increase `M` for better recall at the cost of memory and construction time
   - Increase `ef_construction` for better index quality at the cost of construction time
   - Increase `ef_search` for better search recall at the cost of search time

2. **Batch Operations**:
   - Use `add_items()` instead of multiple `add_item()` calls
   - Use `batch_search()` for multiple queries

3. **Data Preprocessing**:
   - Normalize vectors when using cosine similarity
   - Use float32 for better performance unless you need double precision

## Testing

Run the test suite:

```bash
python test_pyhnsw.py
```

## Dependencies

- Python >= 3.7
- NumPy >= 1.19.0
- pybind11 >= 2.6.0
- Eigen3 (automatically fetched during build)
- C++17 compatible compiler

## License

See the main project LICENSE file.