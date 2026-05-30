
#include "test_hnsw.h"
#include "hnsw/hnsw.h"
#include "Eigen/Dense"
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace dicroce;

REGISTER_TEST_FIXTURE(test_hnsw);

namespace
{
    using vector_type = hnsw_types<float>::vector_type;

    vector_type random_vector(std::mt19937& gen, size_t dim)
    {
        std::normal_distribution<float> nd(0.0f, 1.0f);
        vector_type v(dim);
        for (size_t i = 0; i < dim; ++i)
            v[i] = nd(gen);
        return v;
    }

    // Brute-force exact L2 nearest neighbors (indices into data).
    std::vector<size_t> exact_neighbors(const std::vector<vector_type>& data,
                                        const vector_type& query, size_t k)
    {
        std::vector<std::pair<float, size_t>> all(data.size());
        for (size_t i = 0; i < data.size(); ++i)
            all[i] = { (query - data[i]).squaredNorm(), i };
        k = std::min(k, all.size());
        std::partial_sort(all.begin(), all.begin() + k, all.end());
        std::vector<size_t> ids;
        for (size_t i = 0; i < k; ++i)
            ids.push_back(all[i].second);
        return ids;
    }
}

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

void test_hnsw::test_recall()
{
    const size_t dim = 64, n = 2000, n_queries = 100, k = 10;

    std::mt19937 gen(42);
    std::vector<vector_type> data(n), queries(n_queries);
    for (auto& v : data)    v = random_vector(gen, dim);
    for (auto& q : queries) q = random_vector(gen, dim);

    hnsw<float> index(dim);
    for (auto& v : data)
        index.add_item(v);

    RTF_ASSERT_EQUAL(index.size(), n);

    double recall_sum = 0.0;
    for (const auto& query : queries)
    {
        auto exact = exact_neighbors(data, query, k);
        auto approx = index.search(query, k);

        size_t hits = 0;
        for (const auto& r : approx)
            if (std::find(exact.begin(), exact.end(), r.first) != exact.end())
                ++hits;
        recall_sum += static_cast<double>(hits) / k;
    }

    double recall = recall_sum / n_queries;
    printf("Recall@%zu = %.3f\n", k, recall);

    // Default parameters (M=16, efConstruction=200, efSearch=100) should easily
    // clear this on 2k random 64-d vectors; measured ~0.99.
    RTF_ASSERT(recall >= 0.90);
}

void test_hnsw::test_cosine()
{
    const size_t dim = 32, n = 500;

    hnsw_config cfg;
    cfg.METRIC = hnsw_config::distance_metric::COSINE;

    std::mt19937 gen(7);
    std::vector<vector_type> data(n);
    for (auto& v : data)
        v = random_vector(gen, dim);

    hnsw<float> index(dim, cfg);
    for (auto& v : data)
        index.add_item(v);

    vector_type query = random_vector(gen, dim);

    // Query scale must not change the ranking, and cosine distances must be
    // valid (in [0, 2]) regardless of whether the query was pre-normalized.
    auto r_raw  = index.search(query, 10);
    auto r_norm = index.search((query / query.norm()).eval(), 10);

    RTF_ASSERT_EQUAL(r_raw.size(), r_norm.size());
    for (size_t i = 0; i < r_raw.size(); ++i)
    {
        RTF_ASSERT_EQUAL(r_raw[i].first, r_norm[i].first);
        RTF_ASSERT(r_raw[i].second >= -1e-4f && r_raw[i].second <= 2.0f + 1e-4f);
    }
}

void test_hnsw::test_edge_cases()
{
    // Search on an empty index returns nothing.
    {
        hnsw<float> index(4);
        vector_type q(4); q.setZero();
        RTF_ASSERT(index.search(q, 5).empty());
    }

    // Requesting more neighbors than exist returns exactly what's available.
    {
        hnsw<float> index(4);
        std::mt19937 gen(1);
        for (int i = 0; i < 3; ++i)
            index.add_item(random_vector(gen, 4));
        vector_type q(4); q.setZero();
        RTF_ASSERT_EQUAL(index.search(q, 10).size(), (size_t)3);
    }

    // An exact-match query returns that item first at (near) zero distance.
    {
        hnsw<float> index(4);
        std::mt19937 gen(9);
        vector_type target(4); target << 1, 2, 3, 4;
        size_t target_id = 0;
        for (int i = 0; i < 100; ++i)
        {
            vector_type v = (i == 50) ? target : random_vector(gen, 4);
            if (i == 50) target_id = i;
            index.add_item(v);
        }
        auto r = index.search(target, 1);
        RTF_ASSERT_EQUAL(r.size(), (size_t)1);
        RTF_ASSERT_EQUAL(r[0].first, target_id);
        RTF_ASSERT(r[0].second < 1e-4f);
    }

    // Dimension mismatch is rejected.
    {
        hnsw<float> index(8);
        vector_type wrong(4); wrong.setZero();
        RTF_ASSERT_THROWS(index.add_item(wrong), std::invalid_argument);
        RTF_ASSERT_THROWS(index.search(wrong, 1), std::invalid_argument);
    }
}