#ifndef __hnsw_h
#define __hnsw_h

#include <vector>
#include <random>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <memory>
#include <chrono>
#include <limits>

#include <cmath>

#include <Eigen/Dense>

namespace dicroce
{

// Level 2           *                    *        *                                        *
// Level 1        *  *              *     *        *        *          **                 * *
// Level 0  *     ** *     * * *    *    **   *  * *  * ** **          **     ** *    *   * *   * * *
//
// In an HNSW graph lower levels are more dense and higher levels are more sparse.
// Searches start at the top level and work their way down to the bottom level.
// When inserting a new node a max level is selected and the node is inserted on that level and each lower level down to level 0.

// Configuration parameters for HNSW
struct hnsw_config
{
    // Maximum number of connections per element per layer
    size_t M = 16;
    // Size of the dynamic candidate list for construction
    size_t CONSTRUCTION_EXPANSION_FACTOR = 200; 
    // Size of the dynamic candidate list for search
    size_t SEARCH_EXPANSION_FACTOR = 100;
    // Normalization factor for level generation
    float NORMALIZATION_FACTOR = 0.5f;
    // Maximum level in the graph
    size_t MAX_LEVEL = 16;
    
    // Distance metric to use (L2 or cosine)
    enum class distance_metric
    {
        L2,
        COSINE
    };
    
    distance_metric METRIC = distance_metric::L2;
};

template<typename scalar>
struct hnsw_types {
    using vector_type = Eigen::VectorX<scalar>;
};

template<typename scalar>
class hnsw;

template<typename scalar>
class hnsw_node final
{
    friend class hnsw<scalar>;

public:
    using vector_type = typename hnsw_types<scalar>::vector_type;

    hnsw_node(const vector_type& vec, size_t id, size_t level) :
        _vec(vec), _id(id), _level(level)
    {
        // Initialize connections for each level
        _connections.resize(level + 1);
    }
    
    const vector_type& get_vector() const
    {
        return _vec;
    }
    
    size_t get_id() const
    {
        return _id;
    }
    
    size_t get_level() const
    {
        return _level;
    }
    
    const std::vector<std::vector<size_t>>& get_connections() const
    {
        return _connections;
    }
    
    void add_connection(size_t level, size_t node_id)
    {
        if (level <= _level)
        {
            // Ensure _connections has enough levels
            if (_connections.size() <= level)
                _connections.resize(level + 1);

            _connections[level].push_back(node_id);
        }
    }
    
    void set_connections(size_t level, const std::vector<size_t>& connections)
    {
        if (level <= _level)
        {
            // Ensure _connections has enough levels
            if (_connections.size() <= level)
                _connections.resize(level + 1);

            _connections[level] = connections;
        }
    }
    
private:
    // The vector data
    vector_type _vec;
    // Unique identifier
    size_t _id;
    // Level of this node in the hierarchy
    size_t _level;
    // Connections at each level
    std::vector<std::vector<size_t>> _connections;
};

// Utility for priority queue
template<typename T>
struct neighbor final
{
    size_t _id;
    T _distance;
    
    neighbor(size_t id, T distance) :
        _id(id), _distance(distance)
    {
    }
    
    bool operator<(const neighbor& other) const
    {
        return _distance < other._distance;
    }
    
    bool operator>(const neighbor& other) const
    {
        return _distance > other._distance;
    }
};

// Main hnsw index class
template<typename scalar>
class hnsw final
{
public:
    using vector_type = typename hnsw_types<scalar>::vector_type;

    hnsw(size_t dim, const hnsw_config& config = hnsw_config()) :
        _dim(dim), _config(config), _random_engine(std::random_device{}())
    {
        _level_generator = std::geometric_distribution<size_t>(_config.NORMALIZATION_FACTOR);
    }
    
    // Add a vector to the index
    void add_item(const vector_type& vec)
    {
        if (vec.size() != _dim)
            throw std::invalid_argument("Vector dimension does not match index dimension");
        
        const size_t idx = _nodes.size();
        
        // Generate random level for this node
        size_t random_level = _level_generator(_random_engine);
        random_level = std::min(random_level, _config.MAX_LEVEL);
        
        if (random_level > _current_max_level)
            _current_max_level = random_level;
        
        // If this is the first node, it's the entry point
        if (_nodes.empty())
        {
            auto node = std::make_unique<hnsw_node<scalar>>(vec, idx, random_level);
            _nodes.push_back(std::move(node));
            _entrypoint = 0;
            return;
        }
        
        // Create new node
        auto next_node = std::make_unique<hnsw_node<scalar>>(vec, idx, random_level);

        for (size_t i = 0; i <= random_level; ++i)
            next_node->_connections[i].reserve(_config.M);

        std::vector<size_t> ep_copy{_entrypoint};
        
        // Search from top level to level 1
        int curr_level = static_cast<int>(_current_max_level);
        while (curr_level > static_cast<int>(random_level))
        {
            std::vector<size_t> next_ep = _search_base_layer(vec, ep_copy, 1, curr_level);
            if (!next_ep.empty())
                ep_copy = next_ep;
            curr_level--;
        }

        // Update the entry point if this node is at a higher level
        if (random_level > _nodes[_entrypoint]->get_level())
            _entrypoint = idx;

        // For levels that this node will exist in
        
        while (curr_level >= 0)
        {
            // Find the nearest neighbors at this level
            std::vector<size_t> neighbors = _search_base_layer(vec, ep_copy, _config.CONSTRUCTION_EXPANSION_FACTOR, curr_level);
            
            // Add bidirectional connections
            next_node->set_connections(curr_level, neighbors);
            
            // Add connections back to this node
            for (size_t neighbor_id : neighbors)
            {
                // Validate neighbor ID
                if (neighbor_id >= _nodes.size())
                    continue;

                _nodes[neighbor_id]->add_connection(curr_level, idx);
            }
            
            // Ensure the graph is small-world by pruning connections
            for (size_t neighbor_id : neighbors)
                _prune_connections(neighbor_id, curr_level);
            
            // Update entry point for the next level down
            ep_copy = neighbors;
            
            // Protection against infinite loop - break if we reach level 0
            if (curr_level == 0)
                break;

            curr_level--;
        }
        
        // Add node to the index
        _nodes.push_back(std::move(next_node));
    }
    
    // Search for k nearest neighbors
    std::vector<std::pair<size_t, scalar>> search(const vector_type& query, size_t k) const
    {
        if (query.size() != _dim)
            throw std::invalid_argument("Query dimension does not match index dimension");
        
        if (_nodes.empty())
            return {};
        
        std::vector<size_t> entrypoints{_entrypoint};
        size_t curr_level = _current_max_level;
        
        // Traverse the graph from top level down to level 1
        while (curr_level > 0)
        {
            entrypoints = _search_base_layer(query, entrypoints, 1, curr_level);
            curr_level--;
        }
        
        // Do the final search at the bottom layer with ef_search
        std::vector<size_t> results = _search_base_layer(query, entrypoints, _config.SEARCH_EXPANSION_FACTOR, 0);
        
        // Calculate distances for the results
        std::vector<neighbor<scalar>> candidates;
        for (size_t id : results)
        {
            scalar dist = _distance(query, _nodes[id]->get_vector());
            candidates.emplace_back(id, dist);
        }
        
        // Sort by distance
        std::sort(candidates.begin(), candidates.end());
        
        // Return top k results
        std::vector<std::pair<size_t, scalar>> final_results;
        for (size_t i = 0; i < std::min(k, candidates.size()); ++i)
            final_results.emplace_back(candidates[i]._id, candidates[i]._distance);
        
        return final_results;
    }

    // Get the number of items in the index
    size_t size() const
    {
        return _nodes.size();
    }
    
    // Get the dimension of vectors in the index
    size_t dim() const
    {
        return _dim;
    }
    
private:
    // Dimension of vectors
    size_t _dim;
    // Configuration parameters
    hnsw_config _config;
    // All nodes in the index
    std::vector<std::unique_ptr<hnsw_node<scalar>>> _nodes;
    // Entry point to the graph
    size_t _entrypoint = 0;
    // Current maximum level in the graph
    size_t _current_max_level = 0;
    // Random number generator
    std::mt19937 _random_engine;
    // For generating node levels
    std::geometric_distribution<size_t> _level_generator;
    
    // Search at a specific level
    std::vector<size_t> _search_base_layer(
        const vector_type& query,
        const std::vector<size_t>& entrypoints,
        size_t ef,
        size_t level
    ) const
    {
        // Protection against empty entrypoints
        if (entrypoints.empty())
            return {};
        
        // Protection against empty index
        if (_nodes.empty())
            return {};
        
        // Ensure ef is at least 1
        ef = std::max(size_t(1), ef);
        
        // Priority queue for the closest neighbors found so far (max heap)
        std::priority_queue<neighbor<scalar>> result;
        
        // Priority queue for candidates to explore (min heap)
        std::priority_queue<neighbor<scalar>, std::vector<neighbor<scalar>>, std::greater<>> candidates;
        
        // Set to track visited nodes
        std::vector<bool> visited(_nodes.size(), false);
        
        // Initialize with entrypoints
        scalar furthest_dist = std::numeric_limits<scalar>::max();
        for (size_t ep : entrypoints)
        {
            // Validate index
            if (ep >= _nodes.size())
                continue;
            
            scalar dist = _distance(query, _nodes[ep]->get_vector());
            candidates.emplace(ep, dist);
            result.emplace(ep, dist);
            visited[ep] = true;
            
            if (result.size() > ef)
                result.pop();
            
            if (!result.empty())
                furthest_dist = result.top()._distance;
        }
        
        // Maximum iterations to prevent infinite loop
        const size_t max_iterations = _nodes.size() * 2;
        size_t iterations = 0;

        size_t considered = 0;
        
        // Main search loop
        while (!candidates.empty() && iterations < max_iterations)
        {
            iterations++;
            
            neighbor<scalar> current = candidates.top();
            candidates.pop();
            
            // Stop if we've found enough closer neighbors
            if (current._distance > furthest_dist && considered > ef)
                break;
            
            // Explore neighbors of current node
            const auto& connections = _nodes[current._id]->get_connections();
            // Skip if this node doesn't have connections at this level
            if (level >= connections.size())
                continue;
            
            const auto& level_connections = connections[level];
            for (size_t neighbor_id : level_connections)
            {
                ++considered;

                // Validate index
                if (neighbor_id >= _nodes.size())
                    continue;

                if (!visited[neighbor_id])
                {
                    visited[neighbor_id] = true;
                    
                    scalar dist = _distance(query, _nodes[neighbor_id]->get_vector());
                    
                    // Add to results if distance is small enough
                    if (dist < furthest_dist || result.size() < ef)
                    {
                        candidates.emplace(neighbor_id, dist);
                        result.emplace(neighbor_id, dist);
                        
                        if (result.size() > ef)
                            result.pop();
                        
                        if (!result.empty())
                            furthest_dist = result.top()._distance;
                    }
                }
            }
        }

        // Convert results to vector of IDs
        std::vector<size_t> result_ids;
        while (!result.empty())
        {
            result_ids.push_back(result.top()._id);
            result.pop();
        }
        
        return result_ids;
    }
    
    // Prune connections to maintain small-world property
    void _prune_connections(size_t node_id, size_t level)
    {
        // Check if node_id is valid
        if (node_id >= _nodes.size())
            return;
        
        auto& node = _nodes[node_id];
        
        // Check if node has connections at this level
        if (level >= node->_connections.size())
            return;
        
        auto& connections = node->_connections[level];
        
        if (connections.size() <= _config.M)
            return;
        
        // Calculate distances from this node to all connections
        std::vector<neighbor<scalar>> neighbors;
        for (size_t i = 0; i < connections.size(); ++i)
        {
            size_t neighbor_id = connections[i];
            
            // Skip invalid indices
            if (neighbor_id >= _nodes.size())
                continue;
            
            scalar dist = _distance(node->get_vector(), _nodes[neighbor_id]->get_vector());
            neighbors.emplace_back(neighbor_id, dist);
        }
        
        // Sort by distance
        std::sort(neighbors.begin(), neighbors.end());
        
        // Keep only M closest connections
        connections.clear();
        for (size_t i = 0; i < std::min(_config.M, neighbors.size()); ++i)
            connections.push_back(neighbors[i]._id);
    }
    
    // Calculate distance between two vectors based on the chosen metric
    scalar _distance(const vector_type& a, const vector_type& b) const
    {
        if (_config.METRIC == hnsw_config::distance_metric::L2)
            return _l2_distance(a, b);
        else return _cosine_distance(a, b);
    }
    
    // L2 squared distance
    scalar _l2_distance(const vector_type& a, const vector_type& b) const
    {
        return (a - b).squaredNorm();
    }
    
    // Cosine distance (1 - cosine similarity)
    scalar _cosine_distance(const vector_type& a, const vector_type& b) const
    {
        scalar dot = a.dot(b);
        scalar norma = a.squaredNorm();
        scalar normb = b.squaredNorm();

        if (norma <= 0 || normb <= 0)
            return 1.0;

        return 1.0 - dot / (std::sqrt(norma) * std::sqrt(normb));
    }
};

}

#endif
