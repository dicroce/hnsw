"""Pytest configuration and fixtures for pyhnsw tests."""

import pytest
import numpy as np
import pyhnsw
import sys
import os

# Add parent directory to path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


@pytest.fixture
def small_index():
    """Create a small HNSW index for testing."""
    dim = 32
    index = pyhnsw.HNSW(dim=dim)
    data = np.random.randn(100, dim).astype(np.float32)
    index.add_items(data)
    return index, data


@pytest.fixture
def medium_index():
    """Create a medium-sized HNSW index for testing."""
    dim = 128
    index = pyhnsw.HNSW(dim=dim, M=16, ef_construction=200)
    data = np.random.randn(1000, dim).astype(np.float32)
    index.add_items(data)
    return index, data


@pytest.fixture
def large_index():
    """Create a large HNSW index for testing."""
    dim = 256
    index = pyhnsw.HNSW(dim=dim, M=32, ef_construction=400, ef_search=200)
    data = np.random.randn(5000, dim).astype(np.float32)
    index.add_items(data)
    return index, data


@pytest.fixture
def cosine_index():
    """Create an index with cosine metric."""
    dim = 64
    index = pyhnsw.HNSW(dim=dim, metric="cosine")
    # Use normalized vectors for cosine similarity
    data = np.random.randn(500, dim).astype(np.float32)
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    index.add_items(data)
    return index, data


@pytest.fixture
def double_index():
    """Create a double precision index."""
    dim = 64
    index = pyhnsw.HNSWDouble(dim=dim)
    data = np.random.randn(200, dim).astype(np.float64)
    index.add_items(data)
    return index, data


@pytest.fixture
def empty_index():
    """Create an empty HNSW index."""
    return pyhnsw.HNSW(dim=128)


@pytest.fixture(params=[16, 32, 64, 128, 256])
def dimension(request):
    """Parametrized fixture for different dimensions."""
    return request.param


@pytest.fixture(params=["l2", "cosine"])
def metric(request):
    """Parametrized fixture for different metrics."""
    return request.param


@pytest.fixture(params=[8, 16, 32])
def M_param(request):
    """Parametrized fixture for M parameter."""
    return request.param


@pytest.fixture
def random_data():
    """Generate random test data."""
    def _generate(n_items, dim, dtype=np.float32, normalize=False):
        data = np.random.randn(n_items, dim).astype(dtype)
        if normalize:
            data = data / np.linalg.norm(data, axis=1, keepdims=True)
        return data
    return _generate


@pytest.fixture
def calculate_recall():
    """Calculate recall between exact and approximate results."""
    def _recall(exact_indices, approx_indices):
        n_queries = exact_indices.shape[0]
        k = exact_indices.shape[1]
        
        recall_sum = 0
        for i in range(n_queries):
            exact_set = set(exact_indices[i])
            approx_set = set(approx_indices[i])
            recall_sum += len(exact_set & approx_set) / k
            
        return recall_sum / n_queries
    return _recall


@pytest.fixture
def exact_search():
    """Perform exact nearest neighbor search."""
    def _search(data, queries, k, metric="l2"):
        n_queries = queries.shape[0]
        exact_indices = np.zeros((n_queries, k), dtype=int)
        
        for i, query in enumerate(queries):
            if metric == "l2":
                distances = np.sum((data - query) ** 2, axis=1)
            elif metric == "cosine":
                # Cosine distance = 1 - cosine_similarity
                norms_data = np.linalg.norm(data, axis=1)
                norm_query = np.linalg.norm(query)
                cosine_sim = np.dot(data, query) / (norms_data * norm_query + 1e-10)
                distances = 1 - cosine_sim
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            exact_indices[i] = np.argpartition(distances, min(k, len(distances)-1))[:k]
            exact_indices[i] = exact_indices[i][np.argsort(distances[exact_indices[i]])]
            
        return exact_indices
    return _search


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "stress: mark test as stress test"
    )


# Test collection modifiers
def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    # Add skip markers for slow tests unless explicitly requested
    if not config.getoption("--runslow", default=False):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
                
    # Add skip markers for stress tests unless explicitly requested
    if not config.getoption("--runstress", default=False):
        skip_stress = pytest.mark.skip(reason="need --runstress option to run")
        for item in items:
            if "stress" in item.keywords:
                item.add_marker(skip_stress)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runstress", action="store_true", default=False, help="run stress tests"
    )
    parser.addoption(
        "--runbenchmark", action="store_true", default=False, help="run benchmark tests"
    )