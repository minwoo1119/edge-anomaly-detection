#pragma once

#include "CudaBuffer.hpp"
#include "NearestNeighborSearch.hpp"

#include <cublas_v2.h>

class CublasNearestNeighborSearch final : public INearestNeighborSearch {
public:
    CublasNearestNeighborSearch(const MemoryBank& memoryBank, std::size_t maximumQueries);
    ~CublasNearestNeighborSearch() override;

    CublasNearestNeighborSearch(const CublasNearestNeighborSearch&) = delete;
    CublasNearestNeighborSearch& operator=(const CublasNearestNeighborSearch&) = delete;

    SearchResult search(
        const float* queries,
        std::size_t queryCount,
        std::size_t dimensions,
        const MemoryBank& memoryBank
    ) const override;

    std::size_t deviceBankStorageBytes() const noexcept override {
        return bankBuffer_.sizeBytes();
    }

private:
    std::size_t rows_{0};
    std::size_t dimensions_{0};
    std::size_t maximumQueries_{0};
    mutable cublasHandle_t handle_{nullptr};
    mutable CudaBuffer bankBuffer_;
    mutable CudaBuffer bankNormBuffer_;
    mutable CudaBuffer queryBuffer_;
    mutable CudaBuffer queryNormBuffer_;
    mutable CudaBuffer dotBuffer_;
    mutable CudaBuffer distanceBuffer_;
    mutable CudaBuffer indexBuffer_;
    mutable CudaStream stream_;
    mutable CudaEvent searchStart_;
    mutable CudaEvent searchEnd_;
};
