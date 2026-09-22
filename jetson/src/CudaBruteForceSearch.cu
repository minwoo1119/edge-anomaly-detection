#include "CudaNearestNeighborSearch.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr int threadsPerBlock = 256;

__global__ void nearestNeighborKernel(
    const float* queries,
    const float* bank,
    std::size_t queryCount,
    std::size_t bankRows,
    std::size_t dimensions,
    float* outputDistances,
    unsigned long long* outputIndices
) {
    const std::size_t queryIndex = blockIdx.x;
    if (queryIndex >= queryCount) return;

    float bestDistance = CUDART_INF_F;
    unsigned long long bestIndex = 0;
    const float* query = queries + queryIndex * dimensions;
    for (std::size_t bankIndex = threadIdx.x; bankIndex < bankRows; bankIndex += blockDim.x) {
        const float* row = bank + bankIndex * dimensions;
        float squaredDistance = 0.0F;
        for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
            const float difference = query[dimension] - row[dimension];
            squaredDistance = fmaf(difference, difference, squaredDistance);
        }
        if (squaredDistance < bestDistance) {
            bestDistance = squaredDistance;
            bestIndex = static_cast<unsigned long long>(bankIndex);
        }
    }

    __shared__ float distances[threadsPerBlock];
    __shared__ unsigned long long indices[threadsPerBlock];
    distances[threadIdx.x] = bestDistance;
    indices[threadIdx.x] = bestIndex;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset && distances[threadIdx.x + offset] < distances[threadIdx.x]) {
            distances[threadIdx.x] = distances[threadIdx.x + offset];
            indices[threadIdx.x] = indices[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        outputDistances[queryIndex] = sqrtf(distances[0]);
        outputIndices[queryIndex] = indices[0];
    }
}
}  // namespace

CudaBruteForceSearch::CudaBruteForceSearch(
    const MemoryBank& memoryBank,
    std::size_t maximumQueries
) : rows_(memoryBank.rows()),
    dimensions_(memoryBank.dimensions()),
    maximumQueries_(maximumQueries),
    bankBuffer_(memoryBank.sizeBytes()),
    queryBuffer_(maximumQueries * dimensions_ * sizeof(float)),
    distanceBuffer_(maximumQueries * sizeof(float)),
    indexBuffer_(maximumQueries * sizeof(unsigned long long)) {
    if (maximumQueries_ == 0) {
        throw std::invalid_argument("CUDA NN maximum query count must be positive.");
    }
    checkCuda(
        cudaMemcpyAsync(
            bankBuffer_.data(), memoryBank.values().data(), memoryBank.sizeBytes(),
            cudaMemcpyHostToDevice, stream_.get()
        ),
        "Memory bank H2D copy failed"
    );
    checkCuda(cudaStreamSynchronize(stream_.get()), "Memory bank upload sync failed");
}

SearchResult CudaBruteForceSearch::search(
    const float* queries,
    std::size_t queryCount,
    std::size_t dimensions,
    const MemoryBank& memoryBank
) const {
    if (queries == nullptr || queryCount == 0 || queryCount > maximumQueries_) {
        throw std::invalid_argument("Invalid CUDA NN query buffer or query count.");
    }
    if (dimensions != dimensions_ || memoryBank.dimensions() != dimensions_
        || memoryBank.rows() != rows_) {
        throw std::invalid_argument("CUDA NN memory bank shape changed after initialization.");
    }

    checkCuda(
        cudaMemcpyAsync(
            queryBuffer_.data(), queries, queryCount * dimensions_ * sizeof(float),
            cudaMemcpyHostToDevice, stream_.get()
        ),
        "CUDA NN query H2D copy failed"
    );
    nearestNeighborKernel<<<static_cast<unsigned int>(queryCount), threadsPerBlock, 0, stream_.get()>>>(
        static_cast<const float*>(queryBuffer_.data()),
        static_cast<const float*>(bankBuffer_.data()),
        queryCount,
        rows_,
        dimensions_,
        static_cast<float*>(distanceBuffer_.data()),
        static_cast<unsigned long long*>(indexBuffer_.data())
    );
    checkCuda(cudaGetLastError(), "CUDA NN kernel launch failed");

    SearchResult result;
    result.distances.resize(queryCount);
    std::vector<unsigned long long> indices(queryCount);
    checkCuda(
        cudaMemcpyAsync(
            result.distances.data(), distanceBuffer_.data(), queryCount * sizeof(float),
            cudaMemcpyDeviceToHost, stream_.get()
        ),
        "CUDA NN distance D2H copy failed"
    );
    checkCuda(
        cudaMemcpyAsync(
            indices.data(), indexBuffer_.data(), queryCount * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost, stream_.get()
        ),
        "CUDA NN index D2H copy failed"
    );
    checkCuda(cudaStreamSynchronize(stream_.get()), "CUDA NN stream sync failed");
    result.indices.reserve(queryCount);
    for (const unsigned long long index : indices) {
        result.indices.push_back(static_cast<std::size_t>(index));
    }
    return result;
}
