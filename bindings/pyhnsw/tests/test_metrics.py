"""Tests for different distance metrics in pyhnsw."""

import pytest
import numpy as np
import pyhnsw


class TestL2Distance:
    """Test L2 (Euclidean) distance metric."""
    
    def test_l2_exact_match(self):
        """Test that exact match returns distance close to 0."""
        dim = 32
        index = pyhnsw.HNSW(dim=dim, metric="l2")
        
        vec = np.random.randn(dim).astype(np.float32)
        index.add_item(vec)
        
        indices, distances = index.search(vec, k=1)
        assert distances[0] < 1e-6
        
    def test_l2_distance_calculation(self):
        """Test L2 distance calculation correctness."""
        dim = 16
        index = pyhnsw.HNSW(dim=dim, metric="l2")
        
        # Add known vectors
        vec1 = np.zeros(dim, dtype=np.float32)
        vec2 = np.ones(dim, dtype=np.float32)
        
        index.add_item(vec1)
        index.add_item(vec2)
        
        # Search for vec1
        indices, distances = index.search(vec1, k=2)
        
        # First should be vec1 itself (distance ~0)
        assert indices[0] == 0
        assert distances[0] < 1e-6
        
        # Second should be vec2 with squared L2 distance = dim
        assert indices[1] == 1
        expected_distance = dim  # squared L2 distance
        assert abs(distances[1] - expected_distance) < 0.1
        
    def test_l2_orthogonal_vectors(self):
        """Test L2 distance with orthogonal vectors."""
        dim = 3
        index = pyhnsw.HNSW(dim=dim, metric="l2")
        
        # Add orthogonal unit vectors
        vec1 = np.array([1, 0, 0], dtype=np.float32)
        vec2 = np.array([0, 1, 0], dtype=np.float32)
        vec3 = np.array([0, 0, 1], dtype=np.float32)
        
        index.add_item(vec1)
        index.add_item(vec2)
        index.add_item(vec3)
        
        # Query with a vector equidistant from all
        query = np.array([1, 1, 1], dtype=np.float32) / np.sqrt(3)
        indices, distances = index.search(query, k=3)
        
        # All distances should be similar
        assert np.std(distances) < 0.1
        
    def test_l2_triangle_inequality(self):
        """Test that L2 distance satisfies triangle inequality."""
        dim = 32
        index = pyhnsw.HNSW(dim=dim, metric="l2")
        
        # Add three random vectors
        vecs = np.random.randn(3, dim).astype(np.float32)
        for vec in vecs:
            index.add_item(vec)
        
        # Calculate distances between all pairs
        distances = {}
        for i in range(3):
            indices, dists = index.search(vecs[i], k=3)
            for j, idx in enumerate(indices):
                if idx != i:
                    distances[(i, idx)] = np.sqrt(dists[j])  # Convert to actual L2
        
        # Check triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        if (0, 2) in distances and (0, 1) in distances and (1, 2) in distances:
            assert distances[(0, 2)] <= distances[(0, 1)] + distances[(1, 2)] + 1e-5


class TestCosineDistance:
    """Test cosine distance metric."""
    
    def test_cosine_exact_match(self):
        """Test that exact match returns distance close to 0."""
        dim = 32
        index = pyhnsw.HNSW(dim=dim, metric="cosine")
        
        vec = np.random.randn(dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # Normalize
        index.add_item(vec)
        
        indices, distances = index.search(vec, k=1)
        assert distances[0] < 1e-6
        
    def test_cosine_parallel_vectors(self):
        """Test cosine distance with parallel vectors."""
        dim = 16
        index = pyhnsw.HNSW(dim=dim, metric="cosine")
        
        # Add parallel vectors (same direction, different magnitudes)
        vec1 = np.ones(dim, dtype=np.float32)
        vec2 = 2 * np.ones(dim, dtype=np.float32)
        vec3 = 0.5 * np.ones(dim, dtype=np.float32)
        
        index.add_item(vec1)
        index.add_item(vec2)
        index.add_item(vec3)
        
        # All should have cosine distance ~0 from each other
        indices, distances = index.search(vec1, k=3)
        
        # All distances should be very small (parallel vectors)
        assert all(d < 1e-5 for d in distances)
        
    def test_cosine_orthogonal_vectors(self):
        """Test cosine distance with orthogonal vectors."""
        dim = 3
        index = pyhnsw.HNSW(dim=dim, metric="cosine")
        
        # Add orthogonal vectors
        vec1 = np.array([1, 0, 0], dtype=np.float32)
        vec2 = np.array([0, 1, 0], dtype=np.float32)
        vec3 = np.array([0, 0, 1], dtype=np.float32)
        
        index.add_item(vec1)
        index.add_item(vec2)
        index.add_item(vec3)
        
        # Search with vec1
        indices, distances = index.search(vec1, k=3)
        
        # First should be vec1 itself (distance ~0)
        assert distances[0] < 1e-6
        
        # Others should have cosine distance ~1 (orthogonal)
        assert all(abs(d - 1.0) < 0.1 for d in distances[1:])
        
    def test_cosine_opposite_vectors(self):
        """Test cosine distance with opposite vectors."""
        dim = 32
        index = pyhnsw.HNSW(dim=dim, metric="cosine")
        
        # Add a vector and its opposite
        vec1 = np.random.randn(dim).astype(np.float32)
        vec2 = -vec1
        
        index.add_item(vec1)
        index.add_item(vec2)
        
        # Search with vec1
        indices, distances = index.search(vec1, k=2)
        
        # First should be vec1 (distance ~0)
        assert distances[0] < 1e-6
        
        # Second should be vec2 with cosine distance ~2 (opposite direction)
        assert abs(distances[1] - 2.0) < 0.1
        
    def test_cosine_normalized_vs_unnormalized(self):
        """Test that cosine distance works with both normalized and unnormalized vectors."""
        dim = 64
        index = pyhnsw.HNSW(dim=dim, metric="cosine")
        
        # Add mix of normalized and unnormalized vectors
        n_items = 100
        data = np.random.randn(n_items, dim).astype(np.float32)
        
        # Normalize half of them
        for i in range(0, n_items, 2):
            data[i] = data[i] / np.linalg.norm(data[i])
        
        index.add_items(data)
        
        # Search with normalized query
        query = np.random.randn(dim).astype(np.float32)
        query = query / np.linalg.norm(query)
        
        indices, distances = index.search(query, k=10)
        
        # All distances should be in valid range [0, 2]
        assert all(0 <= d <= 2.01 for d in distances)
        
    def test_cosine_distance_range(self):
        """Test that cosine distances are in the correct range."""
        dim = 32
        n_items = 500
        index = pyhnsw.HNSW(dim=dim, metric="cosine")
        
        # Add random vectors
        data = np.random.randn(n_items, dim).astype(np.float32)
        index.add_items(data)
        
        # Search with multiple queries
        n_queries = 50
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        indices, distances = index.batch_search(queries, k=20)
        
        # All cosine distances should be in [0, 2]
        assert np.all(distances >= -0.01)  # Allow small numerical error
        assert np.all(distances <= 2.01)


class TestMetricComparison:
    """Compare behavior of different metrics."""
    
    def test_metric_consistency(self):
        """Test that the same data gives consistent results with each metric."""
        dim = 64
        n_items = 200
        
        # Use the same data for both metrics
        np.random.seed(42)
        data = np.random.randn(n_items, dim).astype(np.float32)
        query = np.random.randn(dim).astype(np.float32)
        
        # Build L2 index
        index_l2 = pyhnsw.HNSW(dim=dim, metric="l2")
        index_l2.add_items(data)
        indices_l2, distances_l2 = index_l2.search(query, k=10)
        
        # Build cosine index
        index_cosine = pyhnsw.HNSW(dim=dim, metric="cosine")
        index_cosine.add_items(data)
        indices_cosine, distances_cosine = index_cosine.search(query, k=10)
        
        # The top results might differ, but both should return valid results
        assert len(indices_l2) == len(indices_cosine) == 10
        assert all(0 <= idx < n_items for idx in indices_l2)
        assert all(0 <= idx < n_items for idx in indices_cosine)
        
    def test_invalid_metric(self):
        """Test that invalid metric raises an error."""
        with pytest.raises(Exception):
            index = pyhnsw.HNSW(dim=32, metric="invalid_metric")
            
    def test_metric_with_double_precision(self):
        """Test metrics with double precision."""
        dim = 32
        
        # Test L2 with double
        index_l2 = pyhnsw.HNSWDouble(dim=dim, metric="l2")
        data = np.random.randn(100, dim).astype(np.float64)
        index_l2.add_items(data)
        query = np.random.randn(dim).astype(np.float64)
        indices, distances = index_l2.search(query, k=5)
        assert distances.dtype == np.float64
        
        # Test cosine with double
        index_cosine = pyhnsw.HNSWDouble(dim=dim, metric="cosine")
        index_cosine.add_items(data)
        indices, distances = index_cosine.search(query, k=5)
        assert distances.dtype == np.float64
        assert all(0 <= d <= 2.01 for d in distances)