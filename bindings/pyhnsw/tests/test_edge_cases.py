"""Edge case and error handling tests for pyhnsw."""

import pytest
import numpy as np
import pyhnsw


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_index_search(self):
        """Test searching in an empty index."""
        index = pyhnsw.HNSW(dim=32)
        query = np.random.randn(32).astype(np.float32)
        
        indices, distances = index.search(query, k=10)
        assert len(indices) == 0
        assert len(distances) == 0
        
    def test_single_item_search_large_k(self):
        """Test searching for more neighbors than items in index."""
        dim = 16
        index = pyhnsw.HNSW(dim=dim)
        
        vec = np.random.randn(dim).astype(np.float32)
        index.add_item(vec)
        
        # Request 10 neighbors but only 1 item exists
        indices, distances = index.search(vec, k=10)
        assert len(indices) == 1
        assert indices[0] == 0
        
    def test_search_k_equals_index_size(self):
        """Test searching when k equals the index size."""
        dim = 16
        n_items = 50
        
        index = pyhnsw.HNSW(dim=dim)
        data = np.random.randn(n_items, dim).astype(np.float32)
        index.add_items(data)
        
        query = np.random.randn(dim).astype(np.float32)
        indices, distances = index.search(query, k=n_items)
        
        assert len(indices) == n_items
        assert len(set(indices)) == n_items  # All unique
        
    def test_search_k_larger_than_index(self):
        """Test searching when k is larger than index size."""
        dim = 16
        n_items = 20
        
        index = pyhnsw.HNSW(dim=dim)
        data = np.random.randn(n_items, dim).astype(np.float32)
        index.add_items(data)
        
        query = np.random.randn(dim).astype(np.float32)
        indices, distances = index.search(query, k=100)
        
        assert len(indices) == n_items
        assert len(set(indices)) == n_items  # All unique
        
    def test_zero_k_search(self):
        """Test searching with k=0."""
        dim = 16
        index = pyhnsw.HNSW(dim=dim)
        data = np.random.randn(10, dim).astype(np.float32)
        index.add_items(data)
        
        query = np.random.randn(dim).astype(np.float32)
        indices, distances = index.search(query, k=0)
        
        assert len(indices) == 0
        assert len(distances) == 0
        
    def test_very_small_vectors(self):
        """Test with very small magnitude vectors."""
        dim = 32
        index = pyhnsw.HNSW(dim=dim, metric="l2")
        
        # Add vectors with very small values
        small_data = np.random.randn(100, dim).astype(np.float32) * 1e-10
        index.add_items(small_data)
        
        query = np.random.randn(dim).astype(np.float32) * 1e-10
        indices, distances = index.search(query, k=5)
        
        assert len(indices) == 5
        # Distances should be very small
        assert all(d < 1e-15 for d in distances)
        
    def test_very_large_vectors(self):
        """Test with very large magnitude vectors."""
        dim = 32
        index = pyhnsw.HNSW(dim=dim, metric="l2")
        
        # Add vectors with large values
        large_data = np.random.randn(100, dim).astype(np.float32) * 1e6
        index.add_items(large_data)
        
        query = np.random.randn(dim).astype(np.float32) * 1e6
        indices, distances = index.search(query, k=5)
        
        assert len(indices) == 5
        
    def test_duplicate_vectors(self):
        """Test adding duplicate vectors."""
        dim = 16
        index = pyhnsw.HNSW(dim=dim)
        
        # Add the same vector multiple times
        vec = np.random.randn(dim).astype(np.float32)
        for _ in range(10):
            index.add_item(vec)
        
        assert index.size() == 10
        
        # Search should find all duplicates
        indices, distances = index.search(vec, k=10)
        assert len(indices) == 10
        # All distances should be near zero
        assert all(d < 1e-6 for d in distances)
        
    def test_all_zeros_vector(self):
        """Test with zero vectors."""
        dim = 32
        index = pyhnsw.HNSW(dim=dim)
        
        # Add zero vector
        zero_vec = np.zeros(dim, dtype=np.float32)
        index.add_item(zero_vec)
        
        # Add some random vectors
        data = np.random.randn(50, dim).astype(np.float32)
        index.add_items(data)
        
        # Search with zero vector
        indices, distances = index.search(zero_vec, k=5)
        assert indices[0] == 0  # Should find itself first
        assert distances[0] < 1e-6
        
    def test_high_dimensional_vectors(self):
        """Test with very high dimensional vectors."""
        dim = 2048
        n_items = 100
        
        index = pyhnsw.HNSW(dim=dim, M=32, ef_construction=200)
        data = np.random.randn(n_items, dim).astype(np.float32)
        
        # Normalize to prevent numerical issues
        data = data / np.linalg.norm(data, axis=1, keepdims=True)
        index.add_items(data)
        
        query = np.random.randn(dim).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        indices, distances = index.search(query, k=10)
        assert len(indices) == 10
        
    def test_batch_search_empty_index(self):
        """Test batch search on empty index."""
        index = pyhnsw.HNSW(dim=32)
        queries = np.random.randn(10, 32).astype(np.float32)
        
        indices, distances = index.batch_search(queries, k=5)
        
        # Should return empty results for each query
        assert indices.shape == (10, 0) or all(len(row) == 0 for row in indices)
        
    def test_batch_search_partial_results(self):
        """Test batch search when index has fewer items than k."""
        dim = 16
        n_items = 3
        n_queries = 5
        k = 10
        
        index = pyhnsw.HNSW(dim=dim)
        data = np.random.randn(n_items, dim).astype(np.float32)
        index.add_items(data)
        
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        indices, distances = index.batch_search(queries, k=k)
        
        # Each query should return only n_items results
        for i in range(n_queries):
            # Filter out invalid indices (represented as max uint64 value)
            valid_indices = indices[i][indices[i] < n_items]
            assert len(valid_indices) <= n_items


class TestErrorHandling:
    """Test error handling and input validation."""
    
    def test_dimension_mismatch_add(self):
        """Test adding vector with wrong dimension."""
        index = pyhnsw.HNSW(dim=32)
        
        # Try to add vector with wrong dimension
        wrong_vec = np.random.randn(64).astype(np.float32)
        with pytest.raises(RuntimeError):
            index.add_item(wrong_vec)
            
    def test_dimension_mismatch_search(self):
        """Test searching with wrong dimension query."""
        dim = 32
        index = pyhnsw.HNSW(dim=dim)
        
        # Add some data
        data = np.random.randn(10, dim).astype(np.float32)
        index.add_items(data)
        
        # Try to search with wrong dimension
        wrong_query = np.random.randn(64).astype(np.float32)
        with pytest.raises(RuntimeError):
            index.search(wrong_query, k=5)
            
    def test_invalid_input_shape_add(self):
        """Test adding data with invalid shape."""
        index = pyhnsw.HNSW(dim=32)
        
        # Try to add 3D array
        wrong_data = np.random.randn(10, 5, 32).astype(np.float32)
        with pytest.raises(RuntimeError):
            index.add_items(wrong_data)
            
    def test_invalid_input_shape_search(self):
        """Test searching with invalid query shape."""
        index = pyhnsw.HNSW(dim=32)
        data = np.random.randn(10, 32).astype(np.float32)
        index.add_items(data)
        
        # Try to search with 3D array
        wrong_query = np.random.randn(5, 4, 32).astype(np.float32)
        with pytest.raises(RuntimeError):
            index.batch_search(wrong_query, k=5)
            
    def test_negative_k(self):
        """Test searching with negative k."""
        index = pyhnsw.HNSW(dim=32)
        data = np.random.randn(10, 32).astype(np.float32)
        index.add_items(data)
        
        query = np.random.randn(32).astype(np.float32)
        # Most implementations would either error or return empty results
        try:
            indices, distances = index.search(query, k=-5)
            assert len(indices) == 0  # Should return empty if it doesn't error
        except (ValueError, RuntimeError):
            pass  # Expected behavior
            
    def test_nan_values(self):
        """Test handling of NaN values."""
        dim = 16
        index = pyhnsw.HNSW(dim=dim)
        
        # Create vector with NaN
        vec_with_nan = np.random.randn(dim).astype(np.float32)
        vec_with_nan[5] = np.nan
        
        # Adding NaN should either error or handle gracefully
        try:
            index.add_item(vec_with_nan)
            # If it doesn't error, search should handle it
            query = np.random.randn(dim).astype(np.float32)
            indices, distances = index.search(query, k=1)
        except (ValueError, RuntimeError):
            pass  # Expected behavior
            
    def test_inf_values(self):
        """Test handling of infinite values."""
        dim = 16
        index = pyhnsw.HNSW(dim=dim)
        
        # Create vector with inf
        vec_with_inf = np.random.randn(dim).astype(np.float32)
        vec_with_inf[3] = np.inf
        
        # Adding inf should either error or handle gracefully
        try:
            index.add_item(vec_with_inf)
            # If it doesn't error, search should handle it
            query = np.random.randn(dim).astype(np.float32)
            indices, distances = index.search(query, k=1)
        except (ValueError, RuntimeError):
            pass  # Expected behavior


class TestMemoryAndStress:
    """Test memory handling and stress conditions."""
    
    def test_large_batch_add(self):
        """Test adding a very large batch at once."""
        dim = 64
        n_items = 10000
        
        index = pyhnsw.HNSW(dim=dim, M=16, ef_construction=100)
        large_data = np.random.randn(n_items, dim).astype(np.float32)
        
        index.add_items(large_data)
        assert index.size() == n_items
        
    def test_many_small_batches(self):
        """Test adding many small batches."""
        dim = 32
        batch_size = 10
        n_batches = 100
        
        index = pyhnsw.HNSW(dim=dim)
        
        for i in range(n_batches):
            batch = np.random.randn(batch_size, dim).astype(np.float32)
            index.add_items(batch)
            assert index.size() == (i + 1) * batch_size
            
    def test_alternating_add_search(self):
        """Test alternating between adding and searching."""
        dim = 32
        index = pyhnsw.HNSW(dim=dim)
        
        for i in range(10):
            # Add some items
            batch = np.random.randn(100, dim).astype(np.float32)
            index.add_items(batch)
            
            # Search
            query = np.random.randn(dim).astype(np.float32)
            indices, distances = index.search(query, k=min(10, index.size()))
            
            assert len(indices) == min(10, index.size())