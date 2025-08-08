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
    // Maximum number of connections per element per layer (except layer 0)
    size_t M = 16;
    // Maximum number of connections for layer 0 (usually 2*M)
    size_t M0 = 32;
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
    
    // Add a vector to the index (normalized-once for COSINE, diversified selection)
    void add_item(const vector_type& vec)
    {
        if (vec.size() != _dim)
            throw std::invalid_argument("Vector dimension does not match index dimension");

        const size_t idx = _nodes.size();

        // Pick a random level for this node
        size_t random_level = _level_generator(_random_engine);
        random_level = std::min(random_level, _config.MAX_LEVEL);

        if (random_level > _current_max_level)
            _current_max_level = random_level;

        // Prepare stored vector (normalize once for cosine)
        vector_type stored = vec;
        if (_config.METRIC == hnsw_config::distance_metric::COSINE) {
            scalar n = stored.norm();
            if (n > std::numeric_limits<scalar>::epsilon()) stored /= n;
        }

        // First node: just insert and set entrypoint
        if (_nodes.empty())
        {
            auto node = std::make_unique<hnsw_node<scalar>>(stored, idx, random_level);
            _nodes.push_back(std::move(node));
            _entrypoint = 0;
            return;
        }

        // Create the new node with reserved per-level capacity
        auto next_node = std::make_unique<hnsw_node<scalar>>(stored, idx, random_level);
        for (size_t i = 0; i <= random_level; ++i) {
            size_t capacity = (i == 0) ? _config.M0 : _config.M;
            next_node->_connections[i].reserve(capacity);
        }

        // Add node to index BEFORE wiring links
        hnsw_node<scalar>* node_ptr = next_node.get();
        _nodes.push_back(std::move(next_node));

        // Greedy descent from top to (random_level + 1)
        std::vector<size_t> ep_copy{_entrypoint};
        int curr_level = static_cast<int>(_current_max_level);
        while (curr_level > static_cast<int>(random_level))
        {
            std::vector<size_t> next_ep = _search_base_layer(stored, ep_copy, 1, curr_level);
            if (!next_ep.empty()) ep_copy = next_ep;
            curr_level--;
        }

        // If this node’s level eclipses the entrypoint’s, update it
        if (random_level > _nodes[_entrypoint]->get_level())
            _entrypoint = idx;

        // Insert the node on all its levels down to 0
        while (curr_level >= 0)
        {
            // 1) efConstruction search at this level
            std::vector<size_t> candidates =
                _search_base_layer(stored, ep_copy, _config.CONSTRUCTION_EXPANSION_FACTOR, curr_level);

            // 2) Diversified selection for the NEW node (cap links to M/M0)
            const bool is_level0 = (curr_level == 0);
            const size_t cap = is_level0 ? _config.M0 : _config.M;

            std::vector<size_t> sel_u, pruned_u;
            _select_neighbors_heuristic(
                idx,
                candidates,
                curr_level,
                cap,
                /*extendCandidates=*/is_level0,
                /*keepPrunedConnections=*/is_level0,
                sel_u,
                &pruned_u,
                /*max_extension=*/32  // cap expansion at L0 to keep inserts fast (tune as needed)
            );

            // 3) Set the new node’s connections (bounded by cap)
            node_ptr->set_connections(curr_level, sel_u);

            // 4) Add reverse edges + re-prune each neighbor
            for (size_t v : sel_u)
            {
                if (v >= _nodes.size()) continue;
                auto* nv = _nodes[v].get();

                // Bag = v's current neighbors at this level + u
                std::vector<size_t> bag =
                    (nv->get_connections().size() > static_cast<size_t>(curr_level))
                    ? nv->get_connections()[curr_level]
                    : std::vector<size_t>{};
                bag.push_back(idx);

                // Optional: skip dedup (diversified selection will ignore dups).
                // If you really want to dedup:
                // std::sort(bag.begin(), bag.end());
                // bag.erase(std::unique(bag.begin(), bag.end()), bag.end());

                auto new_v = _reselect_for_node(v, curr_level, bag, cap, is_level0);
                nv->set_connections(curr_level, new_v);
            }

            // (Optional) If you want extra stickiness at L0, you can link some of pruned_u -> u here,
            // but it increases degree; most keep it off or very conservative.

            // 5) Use the selected neighbors as entrypoints for the next lower level
            ep_copy = sel_u;

            if (curr_level == 0) break;
            curr_level--;
        }
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

    // ---------- tiny private helpers (no public API changes) ----------
    inline scalar _distance(size_t a_id, size_t b_id) const {
        return _distance(_nodes[a_id]->get_vector(), _nodes[b_id]->get_vector());
    }
    inline scalar _distance(size_t a_id, const vector_type& b) const {
        return _distance(_nodes[a_id]->get_vector(), b);
    }
    
    // Diversified selection (no new public structs)
    inline void _select_neighbors_heuristic(
        size_t center_id,
        const std::vector<size_t>& initial_candidates,
        int level,
        size_t maxM,
        bool extendCandidates,
        bool keepPrunedConnections,
        std::vector<size_t>& out_selected,
        std::vector<size_t>* out_pruned = nullptr,
        size_t max_extension = 0
    ) {
        out_selected.clear();
        if (out_pruned) out_pruned->clear();

        // Build candidate set C (unique)
        std::vector<size_t> C;
        C.reserve(initial_candidates.size() * (extendCandidates ? 2 : 1));
        std::unordered_set<size_t> seen;
        seen.reserve(C.capacity() * 2);

        auto push_unique = [&](size_t id) {
            if (id != center_id && id < _nodes.size() && seen.insert(id).second)
                C.push_back(id);
        };
        for (size_t id : initial_candidates) push_unique(id);

        if (extendCandidates) {
            size_t added = 0;
            for (size_t cid : initial_candidates) {
                if (cid >= _nodes.size()) continue;
                const auto& conns = _nodes[cid]->get_connections();
                if (level >= conns.size()) continue;
                for (size_t nid : conns[level]) {
                    if (max_extension && added >= max_extension) break;
                    size_t before = seen.size();
                    push_unique(nid);
                    if (seen.size() != before) ++added;
                }
                if (max_extension && added >= max_extension) break;
            }
        }

        if (C.empty()) return;

        // Precompute d(center, *)
        std::unordered_map<size_t, scalar> d_center;
        d_center.reserve(C.size() * 2);
        for (size_t id : C) {
            d_center.emplace(id, _distance(center_id, id));
        }

        // Sort by increasing d(center, *)
        std::sort(C.begin(), C.end(), [&](size_t a, size_t b){
            return d_center[a] < d_center[b];
        });

        // Greedy diversified pick
        out_selected.reserve(std::min(maxM, C.size()));
        if (out_pruned) out_pruned->reserve(C.size());

        for (size_t e_id : C) {
            const scalar de = d_center[e_id];
            bool ok = true;
            for (size_t r_id : out_selected) {
                scalar dr = _distance(e_id, r_id);
                if (dr < de) { ok = false; break; }
            }
            if (ok) {
                out_selected.push_back(e_id);
                if (out_selected.size() == maxM) break;
            } else if (keepPrunedConnections && out_pruned) {
                out_pruned->push_back(e_id);
            }
        }

        // Optional backfill if too few passed the diversification test
        if (out_selected.size() < maxM) {
            for (size_t e_id : C) {
                if (out_selected.size() == maxM) break;
                if (std::find(out_selected.begin(), out_selected.end(), e_id) == out_selected.end())
                    out_selected.push_back(e_id);
            }
        }
    }

    inline std::vector<size_t> _reselect_for_node(
        size_t node_id,
        int level,
        const std::vector<size_t>& bag,
        size_t cap,
        bool is_level0
    ) {
        std::vector<size_t> selected;
        _select_neighbors_heuristic(
            node_id,
            bag,
            level,
            cap,
            /*extendCandidates=*/is_level0,
            /*keepPrunedConnections=*/false,
            selected,
            /*out_pruned=*/nullptr,
            /*max_extension=*/0
        );
        return selected;
    }
    // -----------------------------------------------------------------

    // Search at a specific level (fixed stopping condition)
    std::vector<size_t> _search_base_layer(
        const vector_type& query,
        const std::vector<size_t>& entrypoints,
        size_t ef,
        size_t level
    ) const
    {
        if (entrypoints.empty() || _nodes.empty()) return {};
        ef = std::max<size_t>(1, ef);

        // max-heap for results (W), min-heap for candidates (C)
        std::priority_queue<neighbor<scalar>> W;
        std::priority_queue<neighbor<scalar>, std::vector<neighbor<scalar>>, std::greater<>> C;

        std::vector<bool> visited(_nodes.size(), false);

        // Seed from entrypoints
        for (size_t ep : entrypoints)
        {
            if (ep >= _nodes.size()) continue;
            scalar d = _distance(query, _nodes[ep]->get_vector());
            W.emplace(ep, d);
            C.emplace(ep, d);
            visited[ep] = true;
            if (W.size() > ef) W.pop();
        }

        auto worst = [&]() -> scalar {
            return W.empty() ? std::numeric_limits<scalar>::max() : W.top()._distance;
        };

        // Standard HNSW loop: pop best candidate; stop when it is worse than worst in W
        while (!C.empty())
        {
            auto curr = C.top(); C.pop();
            if (curr._distance > worst()) break; // <-- KEY: proper stopping rule

            const auto& conns = _nodes[curr._id]->get_connections();
            if (level >= conns.size()) continue;

            for (size_t nb : conns[level])
            {
                if (nb >= _nodes.size()) continue;
                if (visited[nb]) continue;
                visited[nb] = true;

                scalar d = _distance(query, _nodes[nb]->get_vector());
                if (W.size() < ef || d < worst())
                {
                    C.emplace(nb, d);
                    W.emplace(nb, d);
                    if (W.size() > ef) W.pop();
                }
            }
        }

        // Dump W to sorted vector (ascending distance)
        std::vector<neighbor<scalar>> tmp;
        while (!W.empty()) { tmp.push_back(W.top()); W.pop(); }
        std::sort(tmp.begin(), tmp.end());
        std::vector<size_t> ids;
        ids.reserve(tmp.size());
        for (auto& n : tmp) ids.push_back(n._id);
        return ids;
    }
    
    // NOTE: Old 70/30 pruning removed in favor of diversified selection during insertion.
    // (Keeping a stub here would be misleading, so it's gone.)

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
    
    // Cosine distance (1 - cosine similarity) with numerical stability
    scalar _cosine_distance(const vector_type& a, const vector_type& b) const
    {
        // Vectors are pre-normalized
        return scalar(1.0) - a.dot(b);
    }
};

}

#endif
