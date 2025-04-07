
#include "test_hnsw.h"
#include "hnsw/hnsw.h"
#include "Eigen/Dense"
#include <random>
#include <vector>
#include <chrono>

using namespace std;
using namespace dicroce;

REGISTER_TEST_FIXTURE(test_hnsw);

void test_hnsw::setup()
{
}

void test_hnsw::teardown()
{
}

void test_hnsw::test_basic()
{
    // Create an index for 128-dimensional vectors
    hnsw<float> index(128);

    // Generate some random vectors
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);
    
    const size_t num_vectors = 10000;
    std::vector<hnsw_types<float>::vector_type> vectors(num_vectors);
    
    for (auto& vec : vectors)
    {
        vec.resize(128);
        for (auto& val : vec)
            val = dist(gen);
    }
    
    // Build the index
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < vectors.size(); ++i)
    {
        index.add_item(vectors[i]);
        if (i % 1000 == 0 && i > 0)
            printf("Added %lu vectors\n", i);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    printf("Index built in %f seconds\n", elapsed.count());

    // Search for nearest neighbors
    hnsw_types<float>::vector_type query(128);
    for (auto& val : query)
        val = dist(gen);
    
    start = std::chrono::high_resolution_clock::now();
    
    const size_t k = 10;
    auto results = index.search(query, k);
    
    end = std::chrono::high_resolution_clock::now();
    elapsed = end - start;
    
    printf("Search completed in %f seconds\n", elapsed.count());
    printf("Found %lu results:\n", results.size());
    
    for (const auto& result : results)
        printf("ID: %lu, Distance: %f\n", result.first, result.second);
}