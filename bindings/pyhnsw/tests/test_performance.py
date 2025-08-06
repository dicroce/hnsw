"""Performance and recall tests for pyhnsw."""

import pytest
import numpy as np
import time
import pyhnsw


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.mark.parametrize("dim,n_items", [
        (64, 1000),
        (128, 5000),
        (256, 10000),
    ])
    def test_insertion_speed(self, dim, n_items):
        """Test insertion speed for different configurations."""
        index = pyhnsw.HNSW(dim=dim, M=16, ef_construction=200)
        data = np.random.randn(n_items, dim).astype(np.float32)
        
        start = time.perf_counter()
        index.add_items(data)
        elapsed = time.perf_counter() - start
        
        items_per_sec = n_items / elapsed
        print(f"\nDim={dim}, Items={n_items}: {items_per_sec:.0f} items/sec")
        
        assert index.size() == n_items
        # Basic performance check - should insert at least 100 items/sec
        assert items_per_sec > 100
        
    @pytest.mark.parametrize("n_queries,k", [
        (100, 10),
        (500, 20),
        (1000, 50),
    ])
    def test_search_speed(self, n_queries, k):
        """Test search speed for different query configurations."""
        dim = 128
        n_items = 10000
        
        # Build index
        index = pyhnsw.HNSW(dim=dim, M=16, ef_search=100)
        data = np.random.randn(n_items, dim).astype(np.float32)
        index.add_items(data)
        
        # Prepare queries
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        
        # Measure search time
        start = time.perf_counter()
        indices, distances = index.batch_search(queries, k=k)
        elapsed = time.perf_counter() - start
        
        queries_per_sec = n_queries / elapsed
        print(f"\nQueries={n_queries}, k={k}: {queries_per_sec:.0f} queries/sec")
        
        assert indices.shape == (n_queries, k)
        # Basic performance check - should handle at least 100 queries/sec
        assert queries_per_sec > 100
        
        
    def test_batch_vs_individual_search(self):
        """Compare batch vs individual search performance."""
        dim = 64
        n_items = 5000
        n_queries = 100
        k = 10
        
        # Build index
        index = pyhnsw.HNSW(dim=dim)
        data = np.random.randn(n_items, dim).astype(np.float32)
        index.add_items(data)
        
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        
        # Individual search
        start = time.perf_counter()
        individual_results = []
        for query in queries:
            indices, distances = index.search(query, k=k)
            individual_results.append((indices, distances))
        individual_time = time.perf_counter() - start
        
        # Batch search
        start = time.perf_counter()
        batch_indices, batch_distances = index.batch_search(queries, k=k)
        batch_time = time.perf_counter() - start
        
        print(f"\nIndividual: {individual_time:.3f}s, Batch: {batch_time:.3f}s")
        print(f"Speedup: {individual_time/batch_time:.2f}x")
        
        # Batch should be at least as fast as individual
        assert batch_time <= individual_time * 1.1  # Allow 10% margin


class TestRecall:
    """Test recall quality of the approximate search."""
    
    def calculate_recall(self, exact_indices, approx_indices):
        """Calculate recall@k."""
        n_queries = exact_indices.shape[0]
        k = exact_indices.shape[1]
        
        recall_sum = 0
        for i in range(n_queries):
            exact_set = set(exact_indices[i])
            approx_set = set(approx_indices[i])
            recall_sum += len(exact_set & approx_set) / k
            
        return recall_sum / n_queries
    
    def get_exact_neighbors(self, data, queries, k):
        """Get exact nearest neighbors using brute force."""
        n_queries = queries.shape[0]
        exact_indices = np.zeros((n_queries, k), dtype=int)
        
        for i, query in enumerate(queries):
            distances = np.sum((data - query) ** 2, axis=1)
            exact_indices[i] = np.argpartition(distances, k)[:k]
            exact_indices[i] = exact_indices[i][np.argsort(distances[exact_indices[i]])]
            
        return exact_indices
    
    @pytest.mark.parametrize("M,ef_construction,expected_recall", [
        (8, 100, 0.7),    # Lower quality parameters
        (16, 200, 0.85),  # Default parameters
        (32, 400, 0.95),  # Higher quality parameters
    ])
    def test_recall_with_parameters(self, M, ef_construction, expected_recall):
        """Test recall with different index parameters."""
        dim = 64
        n_items = 2000
        n_queries = 100
        k = 10
        
        # Use fixed seed for reproducibility
        np.random.seed(42)
        data = np.random.randn(n_items, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        
        # Build HNSW index
        index = pyhnsw.HNSW(dim=dim, M=M, ef_construction=ef_construction)
        index.add_items(data)
        
        # Get approximate neighbors
        approx_indices, _ = index.batch_search(queries, k=k)
        
        # Get exact neighbors
        exact_indices = self.get_exact_neighbors(data, queries, k)
        
        # Calculate recall
        recall = self.calculate_recall(exact_indices, approx_indices)
        print(f"\nM={M}, ef_construction={ef_construction}: Recall@{k}={recall:.3f}")
        
        assert recall >= expected_recall
        
    def test_recall_vs_ef_search(self):
        """Test how ef_search affects recall."""
        dim = 64
        n_items = 2000
        n_queries = 50
        k = 10
        
        np.random.seed(42)
        data = np.random.randn(n_items, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        
        # Get exact neighbors
        exact_indices = self.get_exact_neighbors(data, queries, k)
        
        ef_search_values = [50, 100, 200, 400]
        recalls = []
        
        for ef_search in ef_search_values:
            index = pyhnsw.HNSW(dim=dim, M=16, ef_construction=200, ef_search=ef_search)
            index.add_items(data)
            
            approx_indices, _ = index.batch_search(queries, k=k)
            recall = self.calculate_recall(exact_indices, approx_indices)
            recalls.append(recall)
            print(f"\nef_search={ef_search}: Recall@{k}={recall:.3f}")
        
        # Recall should increase with ef_search
        assert all(recalls[i] <= recalls[i+1] * 1.05 for i in range(len(recalls)-1))
        
    def test_recall_high_dimension(self):
        """Test recall in high-dimensional space."""
        dim = 512
        n_items = 1000
        n_queries = 50
        k = 10
        
        np.random.seed(42)
        data = np.random.randn(n_items, dim).astype(np.float32)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        
        # Build index with higher parameters for high dimensions
        index = pyhnsw.HNSW(dim=dim, M=32, ef_construction=400, ef_search=200)
        index.add_items(data)
        
        # Get approximate neighbors
        approx_indices, _ = index.batch_search(queries, k=k)
        
        # Get exact neighbors
        exact_indices = self.get_exact_neighbors(data, queries, k)
        
        # Calculate recall
        recall = self.calculate_recall(exact_indices, approx_indices)
        print(f"\nHigh dimension (dim={dim}): Recall@{k}={recall:.3f}")
        
        # In high dimensions, we expect lower recall but still reasonable
        assert recall >= 0.7


@pytest.mark.benchmark
class TestBenchmark:
    """Benchmark tests for performance comparison."""
    
    def test_benchmark_insertion(self, benchmark):
        """Benchmark insertion performance."""
        dim = 128
        n_items = 1000
        data = np.random.randn(n_items, dim).astype(np.float32)
        
        def insert_data():
            index = pyhnsw.HNSW(dim=dim)
            index.add_items(data)
            return index
        
        result = benchmark(insert_data)
        assert result.size() == n_items
        
    def test_benchmark_search(self, benchmark):
        """Benchmark search performance."""
        dim = 128
        n_items = 10000
        n_queries = 100
        k = 10
        
        # Setup
        index = pyhnsw.HNSW(dim=dim)
        data = np.random.randn(n_items, dim).astype(np.float32)
        index.add_items(data)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        
        # Benchmark
        indices, distances = benchmark(index.batch_search, queries, k)
        assert indices.shape == (n_queries, k)