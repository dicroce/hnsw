#!/usr/bin/env python3
"""
Test script for pyhnsw Python bindings.
"""

import numpy as np
import time
import sys

# Import the module (assumes it's built and installed)
try:
    import pyhnsw
except ImportError:
    print("Error: pyhnsw module not found. Please build and install it first.")
    print("Run: pip install .")
    sys.exit(1)


def test_basic_operations():
    """Test basic HNSW operations."""
    print("Testing basic operations...")
    
    # Create index
    dim = 128
    index = pyhnsw.HNSW(dim=dim, M=16, ef_construction=200, ef_search=100, metric="l2")
    
    # Test empty index
    assert index.size() == 0
    assert index.dim() == dim
    assert len(index) == 0
    print(f"Created index: {index}")
    
    # Add single item
    vec = np.random.randn(dim).astype(np.float32)
    index.add_item(vec)
    assert index.size() == 1
    
    # Search in index with one item
    indices, distances = index.search(vec, k=1)
    assert indices[0] == 0
    assert distances[0] < 1e-6  # Should be very close to 0
    
    print("✓ Basic operations test passed")


def test_batch_operations():
    """Test batch add and search operations."""
    print("\nTesting batch operations...")
    
    dim = 64
    n_items = 1000
    n_queries = 10
    k = 5
    
    # Create index
    index = pyhnsw.HNSW(dim=dim, M=16, ef_construction=200, metric="l2")
    
    # Generate random data
    data = np.random.randn(n_items, dim).astype(np.float32)
    
    # Batch add
    start = time.time()
    index.add_items(data)
    add_time = time.time() - start
    
    assert index.size() == n_items
    print(f"Added {n_items} items in {add_time:.3f} seconds ({n_items/add_time:.0f} items/sec)")
    
    # Batch search
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    start = time.time()
    indices, distances = index.batch_search(queries, k=k)
    search_time = time.time() - start
    
    assert indices.shape == (n_queries, k)
    assert distances.shape == (n_queries, k)
    
    print(f"Searched {n_queries} queries in {search_time:.3f} seconds ({n_queries/search_time:.0f} queries/sec)")
    print("✓ Batch operations test passed")


def test_cosine_similarity():
    """Test cosine similarity metric."""
    print("\nTesting cosine similarity...")
    
    dim = 32
    index = pyhnsw.HNSW(dim=dim, metric="cosine")
    
    # Add normalized vectors
    n_items = 100
    data = np.random.randn(n_items, dim).astype(np.float32)
    # Normalize vectors for cosine similarity
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    
    for vec in data:
        index.add_item(vec)
    
    # Search with a normalized query
    query = np.random.randn(dim).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    indices, distances = index.search(query, k=10)
    
    # Verify distances are in valid range for cosine distance
    assert all(0 <= d <= 2 for d in distances), "Cosine distances should be in [0, 2]"
    
    print("✓ Cosine similarity test passed")


def test_recall():
    """Test recall quality of the index."""
    print("\nTesting recall quality...")
    
    dim = 128
    n_items = 5000
    n_queries = 100
    k = 10
    
    # Generate random data
    np.random.seed(42)  # For reproducibility
    data = np.random.randn(n_items, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    # Build HNSW index
    index = pyhnsw.HNSW(dim=dim, M=16, ef_construction=200, ef_search=100)
    index.add_items(data)
    
    # Get approximate nearest neighbors
    approx_indices, approx_distances = index.batch_search(queries, k=k)
    
    # Calculate exact nearest neighbors (brute force)
    print("Calculating exact nearest neighbors...")
    exact_indices = np.zeros((n_queries, k), dtype=int)
    
    for i, query in enumerate(queries):
        distances = np.sum((data - query) ** 2, axis=1)
        exact_indices[i] = np.argpartition(distances, k)[:k]
        exact_indices[i] = exact_indices[i][np.argsort(distances[exact_indices[i]])]
    
    # Calculate recall
    recall_sum = 0
    for i in range(n_queries):
        approx_set = set(approx_indices[i])
        exact_set = set(exact_indices[i])
        recall_sum += len(approx_set & exact_set) / k
    
    recall = recall_sum / n_queries
    print(f"Recall@{k}: {recall:.3f}")
    
    assert recall > 0.8, f"Recall {recall:.3f} is too low (expected > 0.8)"
    print("✓ Recall test passed")


def test_performance():
    """Test performance with larger dataset."""
    print("\nTesting performance with larger dataset...")
    
    dim = 256
    n_items = 10000
    n_queries = 1000
    k = 20
    
    # Create index with optimized parameters
    index = pyhnsw.HNSW(dim=dim, M=32, ef_construction=400, ef_search=200)
    
    # Generate data
    print(f"Generating {n_items} random vectors of dimension {dim}...")
    data = np.random.randn(n_items, dim).astype(np.float32)
    
    # Measure insertion time
    print("Building index...")
    start = time.time()
    
    # Add in batches for progress reporting
    batch_size = 1000
    for i in range(0, n_items, batch_size):
        batch = data[i:i+batch_size]
        index.add_items(batch)
        if (i + batch_size) % 5000 == 0:
            print(f"  Added {i + batch_size} / {n_items} items")
    
    build_time = time.time() - start
    print(f"Index built in {build_time:.2f} seconds ({n_items/build_time:.0f} items/sec)")
    
    # Measure search time
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    print(f"Searching {n_queries} queries...")
    start = time.time()
    indices, distances = index.batch_search(queries, k=k)
    search_time = time.time() - start
    
    queries_per_sec = n_queries / search_time
    print(f"Search completed in {search_time:.3f} seconds ({queries_per_sec:.0f} queries/sec)")
    print(f"Average query time: {search_time/n_queries*1000:.2f} ms")
    
    print("✓ Performance test passed")


def test_double_precision():
    """Test double precision version."""
    print("\nTesting double precision...")
    
    dim = 64
    index = pyhnsw.HNSWDouble(dim=dim)
    
    # Add items with double precision
    data = np.random.randn(100, dim).astype(np.float64)
    index.add_items(data)
    
    # Search
    query = np.random.randn(dim).astype(np.float64)
    indices, distances = index.search(query, k=5)
    
    assert len(indices) == 5
    assert len(distances) == 5
    
    print("✓ Double precision test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("pyhnsw Test Suite")
    print("=" * 60)
    
    try:
        test_basic_operations()
        test_batch_operations()
        test_cosine_similarity()
        test_recall()
        test_performance()
        test_double_precision()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)