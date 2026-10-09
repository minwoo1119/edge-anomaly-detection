#pragma once

#include "CudaBuffer.hpp"
#include "NearestNeighborSearch.hpp"

class CudaBruteForceSearch final : public INearestNeighborSearch {
public:
    struct DeviceNchwResult {
        SearchResult nearest;
        std::vector<float> maximumDistanceQuery;
    };

    CudaBruteForceSearch(const MemoryBank& memoryBank, std::size_t maximumQueries,
                        bool warpParallel = false, bool tiled = false, bool cacheQuery = false, bool doubleBuffer = false);

    SearchResult search(
        const float* queries,
        std::size_t queryCount,
        std::size_t dimensions,
        const MemoryBank& memoryBank
    ) const override;

    DeviceNchwResult searchDeviceNchw(
        const float* deviceNchw,
        std::size_t channels,
        std::size_t height,
        std::size_t width,
        const MemoryBank& memoryBank
    ) const;

private:
    bool warpParallel_{false};
    bool tiled_{false};
    bool cacheQuery_{false};
    bool doubleBuffer_{false};
    std::size_t rows_{0};
    std::size_t dimensions_{0};
    std::size_t maximumQueries_{0};
    mutable CudaBuffer bankBuffer_;
    mutable CudaBuffer queryBuffer_;
    mutable CudaBuffer distanceBuffer_;
    mutable CudaBuffer indexBuffer_;
    mutable CudaStream stream_;
    mutable CudaEvent searchStart_;
    mutable CudaEvent searchEnd_;
};
