#pragma once

#include "NearestNeighborSearch.hpp"

#include <cstddef>
#include <memory>

class FaissGpuNearestNeighborSearch final : public INearestNeighborSearch {
public:
    FaissGpuNearestNeighborSearch(const MemoryBank& memoryBank, std::size_t maximumQueries,
                                  bool useFloat16 = false);
    ~FaissGpuNearestNeighborSearch() override;

    FaissGpuNearestNeighborSearch(const FaissGpuNearestNeighborSearch&) = delete;
    FaissGpuNearestNeighborSearch& operator=(const FaissGpuNearestNeighborSearch&) = delete;

    SearchResult search(
        const float* queries,
        std::size_t queryCount,
        std::size_t dimensions,
        const MemoryBank& memoryBank
    ) const override;

    std::size_t deviceBankStorageBytes() const noexcept override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::size_t rows_{0};
    std::size_t dimensions_{0};
    std::size_t maximumQueries_{0};
    std::size_t deviceBankStorageBytes_{0};
};
