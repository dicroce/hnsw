"""Basic functionality tests for pyhnsw."""

import pytest
import numpy as np
import pyhnsw


class TestBasicOperations:
    """Test basic HNSW operations."""
    
    def test_index_creation(self):
        """Test creating an HNSW index."""
        index = pyhnsw.HNSW(dim=128)
        assert index.dim() == 128
        assert index.size() == 0
        assert len(index) == 0
        
    def test_index_creation_with_params(self):
        """Test creating an index with custom parameters."""
        index = pyhnsw.HNSW(
            dim=64,
            M=32,
            ef_construction=400,
            ef_search=200,
            metric="l2"
        )
        assert index.dim() == 64
        assert index.size() == 0
        
    def test_add_single_item(self):
        """Test adding a single item to the index."""
        dim = 32
        index = pyhnsw.HNSW(dim=dim)
        
        vec = np.random.randn(dim).astype(np.float32)
        index.add_item(vec)
        
        assert index.size() == 1
        assert len(index) == 1
        
    def test_add_multiple_items_individually(self):
        """Test adding multiple items one by one."""
        dim = 16
        n_items = 100
        index = pyhnsw.HNSW(dim=dim)
        
        for i in range(n_items):
            vec = np.random.randn(dim).astype(np.float32)
            index.add_item(vec)
            assert index.size() == i + 1
            
    def test_add_items_batch(self):
        """Test batch adding of items."""
        dim = 64
        n_items = 500
        index = pyhnsw.HNSW(dim=dim)
        
        data = np.random.randn(n_items, dim).astype(np.float32)
        index.add_items(data)
        
        assert index.size() == n_items
        
    def test_search_single_item(self):
        """Test searching with a single item in the index."""
        dim = 32
        index = pyhnsw.HNSW(dim=dim)
        
        vec = np.random.randn(dim).astype(np.float32)
        index.add_item(vec)
        
        # Search for the same vector
        indices, distances = index.search(vec, k=1)
        
        assert len(indices) == 1
        assert len(distances) == 1
        assert indices[0] == 0
        assert distances[0] < 1e-6  # Should be very close to 0
        
    def test_search_multiple_neighbors(self):
        """Test searching for multiple nearest neighbors."""
        dim = 32
        n_items = 100
        k = 10
        
        index = pyhnsw.HNSW(dim=dim)
        data = np.random.randn(n_items, dim).astype(np.float32)
        index.add_items(data)
        
        query = np.random.randn(dim).astype(np.float32)
        indices, distances = index.search(query, k=k)
        
        assert len(indices) == k
        assert len(distances) == k
        # Distances should be sorted
        assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))
        
    def test_batch_search(self):
        """Test batch search functionality."""
        dim = 32
        n_items = 200
        n_queries = 20
        k = 5
        
        index = pyhnsw.HNSW(dim=dim)
        data = np.random.randn(n_items, dim).astype(np.float32)
        index.add_items(data)
        
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        indices, distances = index.batch_search(queries, k=k)
        
        assert indices.shape == (n_queries, k)
        assert distances.shape == (n_queries, k)
        
        # Check that distances are sorted for each query
        for i in range(n_queries):
            assert all(distances[i, j] <= distances[i, j+1] 
                      for j in range(k-1))
            
    def test_repr(self):
        """Test string representation of the index."""
        index = pyhnsw.HNSW(dim=128)
        data = np.random.randn(50, 128).astype(np.float32)
        index.add_items(data)
        
        repr_str = repr(index)
        assert "HNSW" in repr_str
        assert "dim=128" in repr_str
        assert "size=50" in repr_str


class TestDataTypes:
    """Test different data types and precision."""
    
    def test_float32_precision(self):
        """Test with float32 precision."""
        dim = 32
        index = pyhnsw.HNSW(dim=dim)
        
        data = np.random.randn(100, dim).astype(np.float32)
        index.add_items(data)
        
        query = np.random.randn(dim).astype(np.float32)
        indices, distances = index.search(query, k=5)
        
        assert indices.dtype == np.dtype('uint64') or indices.dtype == np.dtype('int64')
        assert distances.dtype == np.float32
        
    def test_float64_precision(self):
        """Test with float64 precision using HNSWDouble."""
        dim = 32
        index = pyhnsw.HNSWDouble(dim=dim)
        
        data = np.random.randn(100, dim).astype(np.float64)
        index.add_items(data)
        
        query = np.random.randn(dim).astype(np.float64)
        indices, distances = index.search(query, k=5)
        
        assert indices.dtype == np.dtype('uint64') or indices.dtype == np.dtype('int64')
        assert distances.dtype == np.float64
        
    def test_mixed_precision_warning(self):
        """Test that mixing precisions works correctly."""
        dim = 16
        index = pyhnsw.HNSW(dim=dim)  # float32 index
        
        # Add float64 data (should be converted internally)
        data = np.random.randn(50, dim).astype(np.float64)
        index.add_items(data)
        
        # Search with float64 query
        query = np.random.randn(dim).astype(np.float64)
        indices, distances = index.search(query, k=3)
        
        assert len(indices) == 3
        assert distances.dtype == np.float32  # Results should be float32