#pragma once

#include "CudaBuffer.hpp"
#include "NearestNeighborSearch.hpp"

class CudaBruteForceSearch final : public INearestNeighborSearch {
public:
    CudaBruteForceSearch(const MemoryBank& memoryBank, std::size_t maximumQueries);

    SearchResult search(
        const float* queries,
        std::size_t queryCount,
        std::size_t dimensions,
        const MemoryBank& memoryBank
    ) const override;

private:
    std::size_t rows_{0};
    std::size_t dimensions_{0};
    std::size_t maximumQueries_{0};
    mutable CudaBuffer bankBuffer_;
    mutable CudaBuffer queryBuffer_;
    mutable CudaBuffer distanceBuffer_;
    mutable CudaBuffer indexBuffer_;
    mutable CudaStream stream_;
};
