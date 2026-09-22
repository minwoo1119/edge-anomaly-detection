#pragma once

#include "MemoryBank.hpp"

#include <cstddef>
#include <vector>


struct SearchResult {
    std::vector<float> distances;
    std::vector<std::size_t> indices;
};


class INearestNeighborSearch {
public:
    virtual ~INearestNeighborSearch() = default;

    virtual SearchResult search(
        const float* queries,
        std::size_t queryCount,
        std::size_t dimensions,
        const MemoryBank& memoryBank
    ) const = 0;
};


class CpuBruteForceSearch final : public INearestNeighborSearch {
public:
    SearchResult search(
        const float* queries,
        std::size_t queryCount,
        std::size_t dimensions,
        const MemoryBank& memoryBank
    ) const override;
};
