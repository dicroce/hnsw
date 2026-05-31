#ifndef __hnsw_h
#define __hnsw_h

#include <vector>
#include <random>
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <memory>
#include <limits>
#include <unordered_map>
#include <stdexcept>
#include <fstream>
#include <string>
#include <cstdint>

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
    // Maximum number of connections for layer 0 (the dense bottom). 0 = auto,
    // resolved to 2*M at construction (the standard choice).
    size_t M0 = 0;
    // Size of the dynamic candidate list for construction
    size_t CONSTRUCTION_EXPANSION_FACTOR = 200;
    // Size of the dynamic candidate list for search
    size_t SEARCH_EXPANSION_FACTOR = 100;
    // Level-generation normalization factor mL in level = floor(-ln(U) * mL).
    // 0 = auto, resolved to 1/ln(M) at construction (the value from the HNSW
    // paper, which makes P(level >= 1) = 1/M so upper layers stay sparse).
    float NORMALIZATION_FACTOR = 0.0f;
    // Maximum level in the graph (safety cap; with mL=1/ln(M) the natural top
    // level is ~log_M(N), so this never binds for realistic N).
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

    hnsw_node(const vector_type& vec, size_t id, size_t level, int64_t label) :
        _vec(vec), _id(id), _level(level), _label(label)
    {
        // Initialize connections for each level
        _connections.resize(level + 1);
    }

    const vector_type& get_vector() const
    {
        return _vec;
    }

    // Internal index of this node (its position in the index; used for graph wiring).
    size_t get_id() const
    {
        return _id;
    }

    // Caller-supplied external label returned by search().
    int64_t get_label() const
    {
        return _label;
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
    // Internal index (position in the index), used for graph connections.
    size_t _id;
    // Level of this node in the hierarchy
    size_t _level;
    // Caller-supplied external label (what search() returns).
    int64_t _label;
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
        // Resolve "auto" config values to concrete ones (so they also persist
        // correctly through save()/load()).
        if (_config.M0 == 0)
            _config.M0 = 2 * _config.M;
        if (_config.NORMALIZATION_FACTOR <= 0.0f)
            _config.NORMALIZATION_FACTOR = static_cast<float>(
                1.0 / std::log(static_cast<double>(std::max<size_t>(_config.M, 2))));
    }
    
    // Add a vector with an auto-assigned label equal to its insertion order
    // (0, 1, 2, ...), matching the pre-label behavior where search returned
    // insertion indices.
    void add_item(const vector_type& vec)
    {
        add_item(vec, static_cast<int64_t>(_nodes.size()));
    }

    // Add a vector to the index. The label is returned by search() for any hit;
    // it lets callers map results back to their own keys (e.g. a frame id). Labels
    // need not be unique -- that's the caller's choice.
    void add_item(const vector_type& vec, int64_t label)
    {
        if (vec.size() != _dim)
            throw std::invalid_argument("Vector dimension does not match index dimension");

        const size_t idx = _nodes.size();

        // Pick a random level: level = floor(-ln(U) * mL), U ~ uniform(0,1].
        // (1 - U) maps the [0,1) draw into (0,1], avoiding log(0).
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        const double mL = _config.NORMALIZATION_FACTOR;
        size_t random_level = static_cast<size_t>(-std::log(1.0 - unit(_random_engine)) * mL);
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
            auto node = std::make_unique<hnsw_node<scalar>>(stored, idx, random_level, label);
            _nodes.push_back(std::move(node));
            _entrypoint = 0;
            return;
        }

        // Create the new node with reserved per-level capacity
        auto next_node = std::make_unique<hnsw_node<scalar>>(stored, idx, random_level, label);
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

            // 4) Add reverse edges + re-prune each neighbor.
            //    Standard HNSW shrink step: append u to v's neighbor list, and
            //    only re-run the selection heuristic when v is now over capacity.
            //    The cheap append-when-under-cap fast path avoids paying the
            //    O(C^2) heuristic on the common case.
            for (size_t v : sel_u)
            {
                if (v >= _nodes.size()) continue;
                auto* nv = _nodes[v].get();

                // v's current neighbors at this level.
                std::vector<size_t> bag =
                    (nv->get_connections().size() > static_cast<size_t>(curr_level))
                    ? nv->get_connections()[curr_level]
                    : std::vector<size_t>{};

                if (bag.size() < cap)
                {
                    // Under capacity: just add the reverse edge, no heuristic.
                    nv->add_connection(curr_level, idx);
                }
                else
                {
                    // Over capacity: re-select v's best `cap` neighbors from its
                    // existing links plus u. No candidate extension here -- the
                    // shrink operates only on v's own connection set.
                    bag.push_back(idx);
                    auto new_v = _reselect_for_node(v, curr_level, bag, cap);
                    nv->set_connections(curr_level, new_v);
                }
            }

            // (Optional) If you want extra stickiness at L0, you can link some of pruned_u -> u here,
            // but it increases degree; most keep it off or very conservative.

            // 5) Use the selected neighbors as entrypoints for the next lower level
            ep_copy = sel_u;

            if (curr_level == 0) break;
            curr_level--;
        }
    }
    
    // Search for k nearest neighbors. Each result is (label, distance), where
    // label is the caller-supplied id from add_item (insertion order if none was
    // given). Results are sorted by ascending distance.
    std::vector<std::pair<int64_t, scalar>> search(const vector_type& query, size_t k) const
    {
        if (query.size() != _dim)
            throw std::invalid_argument("Query dimension does not match index dimension");
        
        if (_nodes.empty())
            return {};

        // Stored vectors are normalized once for COSINE; normalize the query to
        // match so the returned distances are true cosine distances in [0, 2].
        // (Ranking is unaffected by query scale, but the distance values are.)
        const vector_type* qptr = &query;
        vector_type query_normalized;
        if (_config.METRIC == hnsw_config::distance_metric::COSINE)
        {
            scalar n = query.norm();
            if (n > std::numeric_limits<scalar>::epsilon())
            {
                query_normalized = query / n;
                qptr = &query_normalized;
            }
        }
        const vector_type& q = *qptr;

        std::vector<size_t> entrypoints{_entrypoint};
        size_t curr_level = _current_max_level;

        // Traverse the graph from top level down to level 1
        while (curr_level > 0)
        {
            entrypoints = _search_base_layer(q, entrypoints, 1, curr_level);
            curr_level--;
        }

        // Do the final search at the bottom layer with ef_search
        std::vector<size_t> results = _search_base_layer(q, entrypoints, _config.SEARCH_EXPANSION_FACTOR, 0);

        // Calculate distances for the results
        std::vector<neighbor<scalar>> candidates;
        for (size_t id : results)
        {
            scalar dist = _distance(q, _nodes[id]->get_vector());
            candidates.emplace_back(id, dist);
        }
        
        // Sort by distance
        std::sort(candidates.begin(), candidates.end());
        
        // Return top k results, mapping internal node ids to external labels.
        std::vector<std::pair<int64_t, scalar>> final_results;
        for (size_t i = 0; i < std::min(k, candidates.size()); ++i)
            final_results.emplace_back(_nodes[candidates[i]._id]->get_label(),
                                       candidates[i]._distance);

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

    // --------------------------- persistence ---------------------------
    //
    // Flat, self-describing binary snapshot. The whole graph (vectors AND
    // adjacency) is written, so load() reconstructs a ready-to-search index
    // WITHOUT rebuilding -- construction is the expensive part of HNSW, and
    // the point of persistence is to pay it only once.
    //
    // Layout (sequential, fixed-width integers for 32/64-bit portability):
    //   magic[8]="HNSWIDX\0", u32 version, u32 byte_order_mark, u32 sizeof(scalar), u32 flags
    //   u64 dim
    //   config: u64 M, u64 M0, u64 efC, u64 efS, f32 norm_factor, u64 max_level, u32 metric
    //   u64 entrypoint, u64 current_max_level, u64 node_count
    //   per node (v2): u64 level, i64 label, scalar[dim] vector,
    //                  then for L in [0,level]: u64 cnt, u64 ids[cnt]
    //   (v1 had no per-node label; load() reads v1 and defaults label = position.)
    //
    // The "ids" in a node's connection lists are internal node positions (graph
    // wiring), distinct from the external i64 label that search() returns.
    // The byte_order_mark lets load() reject a file written on a machine with
    // different endianness rather than silently corrupting. The layout is kept
    // sequential and self-describing so a future zero-copy mmap loader is
    // straightforward to add.
    void save(const std::string& path) const
    {
        std::ofstream os(path, std::ios::binary | std::ios::trunc);
        if (!os)
            throw std::runtime_error("hnsw::save: cannot open file for writing: " + path);

        auto put = [&](const auto& v) {
            os.write(reinterpret_cast<const char*>(&v), sizeof(v));
        };
        auto put_u64 = [&](uint64_t v) { put(v); };

        const char magic[8] = {'H','N','S','W','I','D','X','\0'};
        os.write(magic, 8);
        put(static_cast<uint32_t>(FORMAT_VERSION));
        put(static_cast<uint32_t>(BYTE_ORDER_MARK));
        put(static_cast<uint32_t>(sizeof(scalar)));
        put(static_cast<uint32_t>(0)); // flags (reserved)

        put_u64(_dim);
        put_u64(_config.M);
        put_u64(_config.M0);
        put_u64(_config.CONSTRUCTION_EXPANSION_FACTOR);
        put_u64(_config.SEARCH_EXPANSION_FACTOR);
        put(static_cast<float>(_config.NORMALIZATION_FACTOR));
        put_u64(_config.MAX_LEVEL);
        put(static_cast<uint32_t>(_config.METRIC));

        put_u64(_entrypoint);
        put_u64(_current_max_level);
        put_u64(_nodes.size());

        for (const auto& node : _nodes)
        {
            const uint64_t level = node->get_level();
            put_u64(level);
            put(static_cast<int64_t>(node->get_label()));

            const vector_type& vec = node->get_vector();
            os.write(reinterpret_cast<const char*>(vec.data()),
                     static_cast<std::streamsize>(_dim * sizeof(scalar)));

            const auto& conns = node->get_connections();
            for (uint64_t L = 0; L <= level; ++L)
            {
                if (L < conns.size())
                {
                    put_u64(conns[L].size());
                    for (size_t id : conns[L]) put_u64(static_cast<uint64_t>(id));
                }
                else
                {
                    put_u64(0);
                }
            }
        }

        if (!os)
            throw std::runtime_error("hnsw::save: write failed for: " + path);
    }

    // Load a previously saved index. Returns a ready-to-search index; no graph
    // reconstruction is performed.
    static hnsw<scalar> load(const std::string& path)
    {
        std::ifstream is(path, std::ios::binary);
        if (!is)
            throw std::runtime_error("hnsw::load: cannot open file for reading: " + path);

        auto get = [&](auto& v) {
            is.read(reinterpret_cast<char*>(&v), sizeof(v));
            if (!is)
                throw std::runtime_error("hnsw::load: unexpected end of file: " + path);
        };
        auto get_u64 = [&]() { uint64_t v; get(v); return v; };

        char magic[8];
        is.read(magic, 8);
        const char expected[8] = {'H','N','S','W','I','D','X','\0'};
        for (int i = 0; i < 8; ++i)
            if (!is || magic[i] != expected[i])
                throw std::runtime_error("hnsw::load: bad magic (not an hnsw index): " + path);

        uint32_t version = 0, bom = 0, scalar_size = 0, flags = 0;
        get(version); get(bom); get(scalar_size); get(flags);
        if (version == 0 || version > FORMAT_VERSION)
            throw std::runtime_error("hnsw::load: unsupported format version");
        const bool has_labels = (version >= 2); // v1 files predate external labels
        if (bom != BYTE_ORDER_MARK)
            throw std::runtime_error("hnsw::load: endianness mismatch (file written on a different platform)");
        if (scalar_size != sizeof(scalar))
            throw std::runtime_error("hnsw::load: scalar type size mismatch (float vs double?)");

        const uint64_t dim = get_u64();

        hnsw_config cfg;
        cfg.M = static_cast<size_t>(get_u64());
        cfg.M0 = static_cast<size_t>(get_u64());
        cfg.CONSTRUCTION_EXPANSION_FACTOR = static_cast<size_t>(get_u64());
        cfg.SEARCH_EXPANSION_FACTOR = static_cast<size_t>(get_u64());
        float norm_factor = 0.0f; get(norm_factor); cfg.NORMALIZATION_FACTOR = norm_factor;
        cfg.MAX_LEVEL = static_cast<size_t>(get_u64());
        uint32_t metric = 0; get(metric);
        cfg.METRIC = static_cast<hnsw_config::distance_metric>(metric);

        hnsw<scalar> idx(static_cast<size_t>(dim), cfg);

        idx._entrypoint = static_cast<size_t>(get_u64());
        idx._current_max_level = static_cast<size_t>(get_u64());
        const uint64_t node_count = get_u64();

        idx._nodes.reserve(static_cast<size_t>(node_count));
        for (uint64_t i = 0; i < node_count; ++i)
        {
            const uint64_t level = get_u64();

            // v1 had no label; default it to the node's position to match the
            // old "search returns insertion index" behavior.
            int64_t label = static_cast<int64_t>(i);
            if (has_labels) get(label);

            vector_type vec(static_cast<Eigen::Index>(dim));
            is.read(reinterpret_cast<char*>(vec.data()),
                    static_cast<std::streamsize>(dim * sizeof(scalar)));
            if (!is)
                throw std::runtime_error("hnsw::load: truncated vector data: " + path);

            auto node = std::make_unique<hnsw_node<scalar>>(
                vec, static_cast<size_t>(i), static_cast<size_t>(level), label);

            for (uint64_t L = 0; L <= level; ++L)
            {
                const uint64_t cnt = get_u64();
                std::vector<size_t> ids;
                ids.reserve(static_cast<size_t>(cnt));
                for (uint64_t c = 0; c < cnt; ++c)
                    ids.push_back(static_cast<size_t>(get_u64()));
                node->set_connections(static_cast<size_t>(L), ids);
            }

            idx._nodes.push_back(std::move(node));
        }

        return idx;
    }
    // -------------------------------------------------------------------

private:
    // On-disk format constants. v2 added the per-node external label; v1 files
    // still load (label defaults to insertion position).
    static constexpr uint32_t FORMAT_VERSION = 2;
    static constexpr uint32_t BYTE_ORDER_MARK = 0x01020304u;

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
        size_t cap
    ) {
        std::vector<size_t> selected;
        _select_neighbors_heuristic(
            node_id,
            bag,
            level,
            cap,
            /*extendCandidates=*/false,   // shrink never extends: candidate set is v's own links
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
