#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "hnsw/hnsw.h"
#include <sstream>

namespace py = pybind11;
using namespace dicroce;

template<typename scalar>
class PyHNSW {
private:
    std::unique_ptr<hnsw<scalar>> index_;
    size_t dim_;
    
public:
    PyHNSW(size_t dim, size_t M = 16, size_t ef_construction = 200, 
           size_t ef_search = 100, const std::string& metric = "l2") 
        : dim_(dim) {
        
        hnsw_config config;
        config.M = M;
        config.CONSTRUCTION_EXPANSION_FACTOR = ef_construction;
        config.SEARCH_EXPANSION_FACTOR = ef_search;
        
        if (metric == "cosine") {
            config.METRIC = hnsw_config::distance_metric::COSINE;
        } else if (metric == "l2") {
            config.METRIC = hnsw_config::distance_metric::L2;
        } else {
            throw std::invalid_argument("Invalid metric. Use 'l2' or 'cosine'");
        }
        
        index_ = std::make_unique<hnsw<scalar>>(dim, config);
    }
    
    void add_item(py::array_t<scalar> vec, py::object item_id = py::none()) {
        if (vec.ndim() != 1) {
            throw std::runtime_error("Input must be a 1D array");
        }
        
        if (static_cast<size_t>(vec.size()) != dim_) {
            throw std::runtime_error("Vector dimension does not match index dimension");
        }
        
        // Convert numpy array to Eigen vector
        Eigen::Map<const typename hnsw_types<scalar>::vector_type> eigen_vec(
            vec.data(), vec.size()
        );
        
        typename hnsw_types<scalar>::vector_type vec_copy = eigen_vec;
        index_->add_item(vec_copy);
    }
    
    void add_items(py::array_t<scalar> data) {
        if (data.ndim() != 2) {
            throw std::runtime_error("Input must be a 2D array");
        }
        
        auto shape = data.shape();
        size_t num_items = shape[0];
        size_t item_dim = shape[1];
        
        if (item_dim != dim_) {
            throw std::runtime_error("Vector dimension does not match index dimension");
        }
        
        // Add each row as a separate item
        for (size_t i = 0; i < num_items; ++i) {
            Eigen::Map<const typename hnsw_types<scalar>::vector_type> eigen_vec(
                data.data() + i * item_dim, item_dim
            );
            
            typename hnsw_types<scalar>::vector_type vec_copy = eigen_vec;
            index_->add_item(vec_copy);
        }
    }
    
    py::tuple search(py::array_t<scalar> query, int k = 10) {
        if (query.ndim() != 1) {
            throw std::runtime_error("Query must be a 1D array");
        }
        
        if (static_cast<size_t>(query.size()) != dim_) {
            throw std::runtime_error("Query dimension does not match index dimension");
        }
        
        if (k < 0) {
            throw std::runtime_error("k must be non-negative");
        }
        
        if (k == 0) {
            // Return empty arrays for k=0
            py::array_t<size_t> indices(0);
            py::array_t<scalar> distances(0);
            return py::make_tuple(indices, distances);
        }
        
        // Convert numpy array to Eigen vector
        Eigen::Map<const typename hnsw_types<scalar>::vector_type> eigen_query(
            query.data(), query.size()
        );
        
        typename hnsw_types<scalar>::vector_type query_copy = eigen_query;
        auto results = index_->search(query_copy, static_cast<size_t>(k));
        
        // Convert results to numpy arrays
        size_t num_results = results.size();
        py::array_t<size_t> indices(num_results);
        py::array_t<scalar> distances(num_results);
        
        auto indices_ptr = static_cast<size_t*>(indices.mutable_data());
        auto distances_ptr = static_cast<scalar*>(distances.mutable_data());
        
        for (size_t i = 0; i < num_results; ++i) {
            indices_ptr[i] = results[i].first;
            distances_ptr[i] = results[i].second;
        }
        
        return py::make_tuple(indices, distances);
    }
    
    py::tuple batch_search(py::array_t<scalar> queries, int k = 10) {
        if (queries.ndim() != 2) {
            throw std::runtime_error("Queries must be a 2D array");
        }
        
        auto shape = queries.shape();
        size_t num_queries = shape[0];
        size_t query_dim = shape[1];
        
        if (query_dim != dim_) {
            throw std::runtime_error("Query dimension does not match index dimension");
        }
        
        if (k < 0) {
            throw std::runtime_error("k must be non-negative");
        }
        
        // Handle k=0 case
        if (k == 0) {
            auto np = py::module_::import("numpy");
            py::array_t<size_t> all_indices = np.attr("empty")(py::make_tuple(num_queries, 0), py::dtype::of<size_t>());
            py::array_t<scalar> all_distances = np.attr("empty")(py::make_tuple(num_queries, 0), py::dtype::of<scalar>());
            return py::make_tuple(all_indices, all_distances);
        }
        
        // Handle empty index case - return empty arrays
        if (index_->size() == 0) {
            auto np = py::module_::import("numpy");
            py::array_t<size_t> all_indices = np.attr("empty")(py::make_tuple(num_queries, 0), py::dtype::of<size_t>());
            py::array_t<scalar> all_distances = np.attr("empty")(py::make_tuple(num_queries, 0), py::dtype::of<scalar>());
            return py::make_tuple(all_indices, all_distances);
        }
        
        // Always use requested k for array dimensions, but limit actual search
        size_t requested_k = static_cast<size_t>(k);
        size_t effective_k = std::min(requested_k, index_->size());
        
        // Prepare output arrays with requested dimensions (for consistency)
        auto np = py::module_::import("numpy");
        py::array_t<size_t> all_indices = np.attr("empty")(py::make_tuple(num_queries, requested_k), py::dtype::of<size_t>());
        py::array_t<scalar> all_distances = np.attr("empty")(py::make_tuple(num_queries, requested_k), py::dtype::of<scalar>());
        
        auto indices_ptr = static_cast<size_t*>(all_indices.mutable_data());
        auto distances_ptr = static_cast<scalar*>(all_distances.mutable_data());
        
        // Process each query
        for (size_t q = 0; q < num_queries; ++q) {
            Eigen::Map<const typename hnsw_types<scalar>::vector_type> eigen_query(
                queries.data() + q * query_dim, query_dim
            );
            
            typename hnsw_types<scalar>::vector_type query_copy = eigen_query;
            auto results = index_->search(query_copy, effective_k);
            
            // Fill output arrays with actual results
            for (size_t i = 0; i < results.size(); ++i) {
                indices_ptr[q * requested_k + i] = results[i].first;
                distances_ptr[q * requested_k + i] = results[i].second;
            }
            
            // Fill remaining slots with invalid values
            for (size_t i = results.size(); i < requested_k; ++i) {
                indices_ptr[q * requested_k + i] = static_cast<size_t>(-1);
                distances_ptr[q * requested_k + i] = std::numeric_limits<scalar>::infinity();
            }
        }
        
        return py::make_tuple(all_indices, all_distances);
    }
    
    size_t size() const {
        return index_->size();
    }
    
    size_t dim() const {
        return index_->dim();
    }
    
    std::string __repr__() const {
        std::stringstream ss;
        ss << "HNSW(dim=" << dim_ << ", size=" << size() << ")";
        return ss.str();
    }
};

PYBIND11_MODULE(pyhnsw, m) {
    m.doc() = "Python bindings for HNSW (Hierarchical Navigable Small World) vector search";
    
    // Float version
    py::class_<PyHNSW<float>>(m, "HNSW")
        .def(py::init<size_t, size_t, size_t, size_t, const std::string&>(),
             py::arg("dim"),
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("ef_search") = 100,
             py::arg("metric") = "l2",
             R"pbdoc(
             Create a new HNSW index.
             
             Parameters
             ----------
             dim : int
                 Dimensionality of the vectors
             M : int, optional
                 Maximum number of connections per element (default: 16)
             ef_construction : int, optional
                 Size of the dynamic candidate list during construction (default: 200)
             ef_search : int, optional
                 Size of the dynamic candidate list during search (default: 100)
             metric : str, optional
                 Distance metric to use ('l2' or 'cosine', default: 'l2')
             )pbdoc")
        .def("add_item", &PyHNSW<float>::add_item,
             py::arg("vec"),
             py::arg("item_id") = py::none(),
             R"pbdoc(
             Add a single vector to the index.
             
             Parameters
             ----------
             vec : numpy.ndarray
                 1D array of shape (dim,) containing the vector
             item_id : optional
                 Currently unused, for API compatibility
             )pbdoc")
        .def("add_items", &PyHNSW<float>::add_items,
             py::arg("data"),
             R"pbdoc(
             Add multiple vectors to the index.
             
             Parameters
             ----------
             data : numpy.ndarray
                 2D array of shape (n_items, dim) containing the vectors
             )pbdoc")
        .def("search", &PyHNSW<float>::search,
             py::arg("query"),
             py::arg("k") = 10,
             R"pbdoc(
             Search for k nearest neighbors of a query vector.
             
             Parameters
             ----------
             query : numpy.ndarray
                 1D array of shape (dim,) containing the query vector
             k : int, optional
                 Number of nearest neighbors to return (default: 10)
             
             Returns
             -------
             indices : numpy.ndarray
                 1D array of shape (k,) containing the indices of nearest neighbors
             distances : numpy.ndarray
                 1D array of shape (k,) containing the distances to nearest neighbors
             )pbdoc")
        .def("batch_search", &PyHNSW<float>::batch_search,
             py::arg("queries"),
             py::arg("k") = 10,
             R"pbdoc(
             Search for k nearest neighbors of multiple query vectors.
             
             Parameters
             ----------
             queries : numpy.ndarray
                 2D array of shape (n_queries, dim) containing the query vectors
             k : int, optional
                 Number of nearest neighbors to return (default: 10)
             
             Returns
             -------
             indices : numpy.ndarray
                 2D array of shape (n_queries, k) containing the indices of nearest neighbors
             distances : numpy.ndarray
                 2D array of shape (n_queries, k) containing the distances to nearest neighbors
             )pbdoc")
        .def("size", &PyHNSW<float>::size,
             "Get the number of items in the index")
        .def("dim", &PyHNSW<float>::dim,
             "Get the dimensionality of vectors in the index")
        .def("__len__", &PyHNSW<float>::size)
        .def("__repr__", &PyHNSW<float>::__repr__);
    
    // Double version
    py::class_<PyHNSW<double>>(m, "HNSWDouble")
        .def(py::init<size_t, size_t, size_t, size_t, const std::string&>(),
             py::arg("dim"),
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("ef_search") = 100,
             py::arg("metric") = "l2")
        .def("add_item", &PyHNSW<double>::add_item,
             py::arg("vec"),
             py::arg("item_id") = py::none())
        .def("add_items", &PyHNSW<double>::add_items,
             py::arg("data"))
        .def("search", &PyHNSW<double>::search,
             py::arg("query"),
             py::arg("k") = 10)
        .def("batch_search", &PyHNSW<double>::batch_search,
             py::arg("queries"),
             py::arg("k") = 10)
        .def("size", &PyHNSW<double>::size)
        .def("dim", &PyHNSW<double>::dim)
        .def("__len__", &PyHNSW<double>::size)
        .def("__repr__", &PyHNSW<double>::__repr__);
}