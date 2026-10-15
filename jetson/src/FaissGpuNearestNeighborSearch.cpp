#include "FaissGpuNearestNeighborSearch.hpp"

#include <faiss/Index.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

struct FaissGpuNearestNeighborSearch::Impl {
    faiss::gpu::StandardGpuResources resources;
    std::unique_ptr<faiss::gpu::GpuIndexFlatL2> index;

    explicit Impl(int dimensions) {
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = 0;
        config.useFloat16 = false;
        index = std::make_unique<faiss::gpu::GpuIndexFlatL2>(
            &resources, dimensions, config
        );
    }
};

namespace {
faiss::idx_t checkedIndex(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<faiss::idx_t>::max())) {
        throw std::overflow_error(std::string(name) + " exceeds the FAISS index limit.");
    }
    return static_cast<faiss::idx_t>(value);
}
}  // namespace

FaissGpuNearestNeighborSearch::FaissGpuNearestNeighborSearch(
    const MemoryBank& memoryBank, std::size_t maximumQueries
) : rows_(memoryBank.rows()),
    dimensions_(memoryBank.dimensions()),
    maximumQueries_(maximumQueries) {
    if (dimensions_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Embedding dimensions exceed the FAISS int32 limit.");
    }
    if (rows_ == 0 || dimensions_ == 0 || maximumQueries_ == 0) {
        throw std::invalid_argument("FAISS GPU NN requires a non-empty bank and query capacity.");
    }
    impl_ = std::make_unique<Impl>(static_cast<int>(dimensions_));
    impl_->index->add(checkedIndex(rows_, "Memory-bank rows"), memoryBank.values().data());
}

FaissGpuNearestNeighborSearch::~FaissGpuNearestNeighborSearch() = default;

SearchResult FaissGpuNearestNeighborSearch::search(
    const float* queries,
    std::size_t queryCount,
    std::size_t dimensions,
    const MemoryBank& memoryBank
) const {
    if (queries == nullptr || queryCount == 0 || queryCount > maximumQueries_) {
        throw std::invalid_argument("Invalid FAISS GPU NN query buffer or query count.");
    }
    if (dimensions != dimensions_ || memoryBank.dimensions() != dimensions_
        || memoryBank.rows() != rows_) {
        throw std::invalid_argument("FAISS GPU NN memory-bank shape changed after initialization.");
    }

    std::vector<float> squaredDistances(queryCount);
    std::vector<faiss::idx_t> labels(queryCount);
    const auto start = std::chrono::steady_clock::now();
    impl_->index->search(
        checkedIndex(queryCount, "Query count"), queries, 1,
        squaredDistances.data(), labels.data()
    );
    const auto end = std::chrono::steady_clock::now();

    SearchResult result;
    result.deviceMilliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    result.distances.reserve(queryCount);
    result.indices.reserve(queryCount);
    for (std::size_t query = 0; query < queryCount; ++query) {
        if (labels[query] < 0) {
            throw std::runtime_error("FAISS GPU NN returned an invalid nearest-neighbor label.");
        }
        result.distances.push_back(std::sqrt(std::max(0.0F, squaredDistances[query])));
        result.indices.push_back(static_cast<std::size_t>(labels[query]));
    }
    return result;
}
