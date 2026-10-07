#include "CudaNearestNeighborSearch.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
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
        if (threadIdx.x < offset) {
            const float candidateDistance = distances[threadIdx.x + offset];
            const unsigned long long candidateIndex = indices[threadIdx.x + offset];
            const bool candidateIsBetter = candidateDistance < distances[threadIdx.x]
                || (candidateDistance == distances[threadIdx.x]
                    && candidateIndex < indices[threadIdx.x]);
            if (candidateIsBetter) {
                distances[threadIdx.x] = candidateDistance;
                indices[threadIdx.x] = candidateIndex;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        outputDistances[queryIndex] = sqrtf(distances[0]);
        outputIndices[queryIndex] = indices[0];
    }
}

__global__ void nearestNeighborNchwKernel(
    const float* nchw,
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
    for (std::size_t bankIndex = threadIdx.x; bankIndex < bankRows; bankIndex += blockDim.x) {
        const float* row = bank + bankIndex * dimensions;
        float squaredDistance = 0.0F;
        for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
            const float difference = nchw[dimension * queryCount + queryIndex] - row[dimension];
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
        if (threadIdx.x < offset) {
            const float candidateDistance = distances[threadIdx.x + offset];
            const unsigned long long candidateIndex = indices[threadIdx.x + offset];
            const bool candidateIsBetter = candidateDistance < distances[threadIdx.x]
                || (candidateDistance == distances[threadIdx.x]
                    && candidateIndex < indices[threadIdx.x]);
            if (candidateIsBetter) {
                distances[threadIdx.x] = candidateDistance;
                indices[threadIdx.x] = candidateIndex;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        outputDistances[queryIndex] = sqrtf(distances[0]);
        outputIndices[queryIndex] = indices[0];
    }
}

__global__ void gatherNchwPatchKernel(
    const float* nchw,
    std::size_t patchIndex,
    std::size_t patchCount,
    std::size_t dimensions,
    float* output
) {
    const std::size_t dimension = blockIdx.x * blockDim.x + threadIdx.x;
    if (dimension < dimensions) {
        output[dimension] = nchw[dimension * patchCount + patchIndex];
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

    searchStart_.record(stream_.get());
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
    searchEnd_.record(stream_.get());
    searchEnd_.synchronize();
    result.deviceMilliseconds = CudaEvent::elapsedMilliseconds(searchStart_, searchEnd_);
    result.indices.reserve(queryCount);
    for (const unsigned long long index : indices) {
        result.indices.push_back(static_cast<std::size_t>(index));
    }
    return result;
}

CudaBruteForceSearch::DeviceNchwResult CudaBruteForceSearch::searchDeviceNchw(
    const float* deviceNchw,
    std::size_t channels,
    std::size_t height,
    std::size_t width,
    const MemoryBank& memoryBank
) const {
    if (deviceNchw == nullptr || channels != dimensions_ || height == 0 || width == 0) {
        throw std::invalid_argument("Invalid device NCHW embedding shape.");
    }
    if (memoryBank.dimensions() != dimensions_ || memoryBank.rows() != rows_) {
        throw std::invalid_argument("CUDA NN memory bank shape changed after initialization.");
    }
    const std::size_t queryCount = height * width;
    if (queryCount > maximumQueries_) {
        throw std::invalid_argument("Device NCHW query count exceeds the allocated maximum.");
    }

    searchStart_.record(stream_.get());
    nearestNeighborNchwKernel<<<static_cast<unsigned int>(queryCount), threadsPerBlock, 0, stream_.get()>>>(
        deviceNchw,
        static_cast<const float*>(bankBuffer_.data()),
        queryCount,
        rows_,
        dimensions_,
        static_cast<float*>(distanceBuffer_.data()),
        static_cast<unsigned long long*>(indexBuffer_.data())
    );
    checkCuda(cudaGetLastError(), "CUDA device NCHW NN kernel launch failed");

    DeviceNchwResult output;
    output.nearest.distances.resize(queryCount);
    std::vector<unsigned long long> indices(queryCount);
    checkCuda(
        cudaMemcpyAsync(
            output.nearest.distances.data(),
            distanceBuffer_.data(),
            queryCount * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_.get()
        ),
        "CUDA NN distances D2H failed"
    );
    checkCuda(
        cudaMemcpyAsync(
            indices.data(),
            indexBuffer_.data(),
            queryCount * sizeof(unsigned long long),
            cudaMemcpyDeviceToHost,
            stream_.get()
        ),
        "CUDA NN indices D2H failed"
    );
    checkCuda(cudaStreamSynchronize(stream_.get()), "CUDA NN result sync failed");

    const auto maximum = std::max_element(
        output.nearest.distances.begin(), output.nearest.distances.end()
    );
    const std::size_t maximumPatch = static_cast<std::size_t>(
        std::distance(output.nearest.distances.begin(), maximum)
    );
    const unsigned int gatherBlocks = static_cast<unsigned int>(
        (dimensions_ + threadsPerBlock - 1) / threadsPerBlock
    );
    gatherNchwPatchKernel<<<gatherBlocks, threadsPerBlock, 0, stream_.get()>>>(
        deviceNchw, maximumPatch, queryCount, dimensions_,
        static_cast<float*>(queryBuffer_.data())
    );
    checkCuda(cudaGetLastError(), "CUDA maximum patch gather kernel launch failed");
    output.maximumDistanceQuery.resize(dimensions_);
    checkCuda(
        cudaMemcpyAsync(
            output.maximumDistanceQuery.data(),
            queryBuffer_.data(),
            dimensions_ * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream_.get()
        ),
        "CUDA maximum patch D2H failed"
    );
    searchEnd_.record(stream_.get());
    searchEnd_.synchronize();
    output.nearest.deviceMilliseconds = CudaEvent::elapsedMilliseconds(
        searchStart_, searchEnd_
    );
    output.nearest.indices.reserve(queryCount);
    for (const unsigned long long index : indices) {
        output.nearest.indices.push_back(static_cast<std::size_t>(index));
    }
    return output;
}
