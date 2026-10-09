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

// Eight query warps reuse four full bank rows staged by the whole block.
// Invalid tail queries still participate in every block barrier.
template<bool nchwLayout, bool cacheQuery = false>
__global__ void tiledNearestNeighborKernel(
    const float* queries, const float* bank, std::size_t queryCount,
    std::size_t bankRows, std::size_t dimensions,
    float* outputDistances, unsigned long long* outputIndices
) {
    constexpr int bankTile = 4;
    constexpr int queryTile = threadsPerBlock / 32;
    const int lane = threadIdx.x % 32;
    const std::size_t q = blockIdx.x * queryTile + threadIdx.x / 32;
    extern __shared__ float bankCache[];
    float* queryCache = bankCache + bankTile * dimensions;
    if (cacheQuery) {
        for (std::size_t i = threadIdx.x; i < queryTile * dimensions; i += blockDim.x) {
            const std::size_t cachedQ = blockIdx.x * queryTile + i / dimensions;
            const std::size_t d = i % dimensions;
            queryCache[i] = cachedQ < queryCount
                ? (nchwLayout ? queries[d * queryCount + cachedQ]
                              : queries[cachedQ * dimensions + d]) : 0.0F;
        }
        __syncthreads();
    }
    float best = CUDART_INF_F;
    unsigned long long bestIndex = 0;
    for (std::size_t base = 0; base < bankRows; base += bankTile) {
        const int rows = static_cast<int>(min(static_cast<unsigned long long>(bankTile),
                                            static_cast<unsigned long long>(bankRows - base)));
        for (std::size_t i = threadIdx.x; i < rows * dimensions; i += blockDim.x)
            bankCache[i] = bank[base * dimensions + i];
        __syncthreads();
        float squared[bankTile] = {0.0F, 0.0F, 0.0F, 0.0F};
        if (q < queryCount) {
            for (std::size_t d = lane; d < dimensions; d += 32) {
                const float value = cacheQuery
                    ? queryCache[(threadIdx.x / 32) * dimensions + d]
                    : (nchwLayout ? queries[d * queryCount + q]
                                  : queries[q * dimensions + d]);
                #pragma unroll
                for (int r = 0; r < bankTile; ++r) {
                    if (r < rows) {
                        const float difference = value - bankCache[r * dimensions + d];
                        squared[r] = fmaf(difference, difference, squared[r]);
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < bankTile; ++r) {
            for (int offset = 16; offset > 0; offset /= 2)
                squared[r] += __shfl_down_sync(0xffffffffU, squared[r], offset);
            if (q < queryCount && lane == 0 && r < rows
                && (squared[r] < best || (squared[r] == best && base + r < bestIndex))) {
                best = squared[r];
                bestIndex = base + r;
            }
        }
        __syncthreads();
    }
    if (q < queryCount && lane == 0) {
        outputDistances[q] = sqrtf(best);
        outputIndices[q] = bestIndex;
    }
}

// Ampere asynchronous global-to-shared copies. Odd feature dimensions use
// the bounded synchronous fallback so test and generic input shapes stay valid.
__device__ void stageBankTile(float* destination, const float* source,
                              std::size_t rows, std::size_t dimensions) {
#if __CUDA_ARCH__ >= 800
    if (dimensions % 4 == 0) {
        for (std::size_t i = threadIdx.x * 4; i < rows * dimensions; i += blockDim.x * 4) {
            const unsigned int address = static_cast<unsigned int>(__cvta_generic_to_shared(destination + i));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::
                         "r"(address), "l"(source + i) : "memory");
        }
    } else
#endif
    {
        for (std::size_t i = threadIdx.x; i < rows * dimensions; i += blockDim.x)
            destination[i] = source[i];
    }
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ void finishBankTile() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

// Eight query warps reuse four full bank rows staged by the whole block.
// Invalid tail queries still participate in every block barrier.
template<bool nchwLayout>
__global__ void doubleBufferedNearestNeighborKernel(
    const float* queries, const float* bank, std::size_t queryCount,
    std::size_t bankRows, std::size_t dimensions,
    float* outputDistances, unsigned long long* outputIndices
) {
    constexpr int bankTile = 4;
    constexpr int queryTile = threadsPerBlock / 32;
    const int lane = threadIdx.x % 32;
    const std::size_t q = blockIdx.x * queryTile + threadIdx.x / 32;
    extern __shared__ float bankCache[];
    int current = 0;
    stageBankTile(bankCache, bank, min(bankRows, std::size_t(4)), dimensions);
    finishBankTile();
    __syncthreads();
    float best = CUDART_INF_F;
    unsigned long long bestIndex = 0;
    for (std::size_t base = 0; base < bankRows; base += bankTile) {
        const int rows = static_cast<int>(min(static_cast<unsigned long long>(bankTile),
                                            static_cast<unsigned long long>(bankRows - base)));
        const std::size_t nextBase = base + bankTile;
        float* currentBank = bankCache + current * bankTile * dimensions;
        if (nextBase < bankRows) {
            const std::size_t nextRows = min(bankRows - nextBase, std::size_t(4));
            stageBankTile(bankCache + (1 - current) * bankTile * dimensions,
                          bank + nextBase * dimensions, nextRows, dimensions);
        }
        float squared[bankTile] = {0.0F, 0.0F, 0.0F, 0.0F};
        if (q < queryCount) {
            for (std::size_t d = lane; d < dimensions; d += 32) {
                const float value = nchwLayout ? queries[d * queryCount + q]
                                              : queries[q * dimensions + d];
                #pragma unroll
                for (int r = 0; r < bankTile; ++r) {
                    if (r < rows) {
                        const float difference = value - currentBank[r * dimensions + d];
                        squared[r] = fmaf(difference, difference, squared[r]);
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < bankTile; ++r) {
            for (int offset = 16; offset > 0; offset /= 2)
                squared[r] += __shfl_down_sync(0xffffffffU, squared[r], offset);
            if (q < queryCount && lane == 0 && r < rows
                && (squared[r] < best || (squared[r] == best && base + r < bestIndex))) {
                best = squared[r];
                bestIndex = base + r;
            }
        }
        finishBankTile();
        __syncthreads();
        current = 1 - current;
    }
    if (q < queryCount && lane == 0) {
        outputDistances[q] = sqrtf(best);
        outputIndices[q] = bestIndex;
    }
}

// One warp cooperates on a bank row: adjacent lanes read adjacent dimensions.
// Cache the query once per block, including for device-resident NCHW inputs.
template<bool nchwLayout>
__global__ void warpNearestNeighborKernel(
    const float* queries, const float* bank, std::size_t queryCount,
    std::size_t bankRows, std::size_t dimensions,
    float* outputDistances, unsigned long long* outputIndices
) {
    const std::size_t queryIndex = blockIdx.x;
    if (queryIndex >= queryCount) return;
    extern __shared__ float query[];
    for (std::size_t d = threadIdx.x; d < dimensions; d += blockDim.x)
        query[d] = nchwLayout ? queries[d * queryCount + queryIndex]
                             : queries[queryIndex * dimensions + d];
    __syncthreads();
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    constexpr int warpCount = threadsPerBlock / 32;
    float best = CUDART_INF_F;
    unsigned long long bestIndex = 0;
    for (std::size_t row = warp; row < bankRows; row += warpCount) {
        float squared = 0.0F;
        for (std::size_t d = lane; d < dimensions; d += 32) {
            const float difference = query[d] - bank[row * dimensions + d];
            squared = fmaf(difference, difference, squared);
        }
        for (int offset = 16; offset > 0; offset /= 2)
            squared += __shfl_down_sync(0xffffffffU, squared, offset);
        if (lane == 0 && (squared < best || (squared == best && row < bestIndex))) {
            best = squared;
            bestIndex = row;
        }
    }
    __shared__ float warpDistances[warpCount];
    __shared__ unsigned long long warpIndices[warpCount];
    if (lane == 0) {
        warpDistances[warp] = best;
        warpIndices[warp] = bestIndex;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < warpCount; ++w) {
            if (warpDistances[w] < best
                || (warpDistances[w] == best && warpIndices[w] < bestIndex)) {
                best = warpDistances[w];
                bestIndex = warpIndices[w];
            }
        }
        outputDistances[queryIndex] = sqrtf(best);
        outputIndices[queryIndex] = bestIndex;
    }
}

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

// Convert [channels, patches] to [patches, channels] entirely on the GPU.
// Padding avoids shared-memory bank conflicts during transposed reads.
__global__ void nchwToPatchMajorKernel(const float* input, float* output,
                                      std::size_t patches, std::size_t channels) {
    __shared__ float tile[32][33];
    const std::size_t patch = blockIdx.x * 32 + threadIdx.x;
    const std::size_t channel = blockIdx.y * 32 + threadIdx.y;
    for (int offset = 0; offset < 32; offset += 8) {
        if (patch < patches && channel + offset < channels)
            tile[threadIdx.y + offset][threadIdx.x] = input[(channel + offset) * patches + patch];
    }
    __syncthreads();
    const std::size_t outputChannel = blockIdx.y * 32 + threadIdx.x;
    const std::size_t outputPatch = blockIdx.x * 32 + threadIdx.y;
    for (int offset = 0; offset < 32; offset += 8) {
        if (outputChannel < channels && outputPatch + offset < patches)
            output[(outputPatch + offset) * channels + outputChannel] = tile[threadIdx.x][threadIdx.y + offset];
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
    std::size_t maximumQueries,
    bool warpParallel,
    bool tiled,
    bool cacheQuery,
    bool doubleBuffer,
    bool transposeDeviceInput
) : warpParallel_(warpParallel), tiled_(tiled || cacheQuery || doubleBuffer || transposeDeviceInput), cacheQuery_(cacheQuery), doubleBuffer_(doubleBuffer || transposeDeviceInput), transposeDeviceInput_(transposeDeviceInput), rows_(memoryBank.rows()),
    dimensions_(memoryBank.dimensions()),
    maximumQueries_(maximumQueries),
    bankBuffer_(memoryBank.sizeBytes()),
    queryBuffer_(maximumQueries * dimensions_ * sizeof(float)),
    distanceBuffer_(maximumQueries * sizeof(float)),
    indexBuffer_(maximumQueries * sizeof(unsigned long long)) {
    if (maximumQueries_ == 0) {
        throw std::invalid_argument("CUDA NN maximum query count must be positive.");
    }
    if (tiled_) {
        int device = 0;
        int sharedLimit = 0;
        checkCuda(cudaGetDevice(&device), "CUDA NN device query failed");
        checkCuda(cudaDeviceGetAttribute(&sharedLimit, (cacheQuery_ || doubleBuffer_) ? cudaDevAttrMaxSharedMemoryPerBlockOptin : cudaDevAttrMaxSharedMemoryPerBlock, device),
                  "CUDA NN shared-memory limit query failed");
        if (dimensions_ > static_cast<std::size_t>(sharedLimit) / ((cacheQuery_ ? 12 : (doubleBuffer_ ? 8 : 4)) * sizeof(float)))
            throw std::invalid_argument("CUDA tiled data exceeds shared-memory capacity.");
        if (doubleBuffer_) {
            const int bytes = static_cast<int>(8 * dimensions_ * sizeof(float));
            checkCuda(cudaFuncSetAttribute(doubleBufferedNearestNeighborKernel<false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, bytes), "Double buffer shared-memory opt-in failed");
            checkCuda(cudaFuncSetAttribute(doubleBufferedNearestNeighborKernel<true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, bytes), "Double buffer NCHW opt-in failed");
        }
        if (cacheQuery_) {
            const int bytes = static_cast<int>(12 * dimensions_ * sizeof(float));
            checkCuda(cudaFuncSetAttribute(tiledNearestNeighborKernel<false, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, bytes), "Cached kernel shared-memory opt-in failed");
            checkCuda(cudaFuncSetAttribute(tiledNearestNeighborKernel<true, true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, bytes), "Cached NCHW shared-memory opt-in failed");
        }
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
    if (doubleBuffer_) {
        doubleBufferedNearestNeighborKernel<false><<<static_cast<unsigned int>((queryCount + 7) / 8),
            threadsPerBlock, 8 * dimensions_ * sizeof(float), stream_.get()>>>(
            static_cast<const float*>(queryBuffer_.data()), static_cast<const float*>(bankBuffer_.data()),
            queryCount, rows_, dimensions_, static_cast<float*>(distanceBuffer_.data()),
            static_cast<unsigned long long*>(indexBuffer_.data()));
    } else if (cacheQuery_) {
        tiledNearestNeighborKernel<false, true><<<static_cast<unsigned int>((queryCount + 7) / 8),
            threadsPerBlock, 12 * dimensions_ * sizeof(float), stream_.get()>>>(
            static_cast<const float*>(queryBuffer_.data()), static_cast<const float*>(bankBuffer_.data()),
            queryCount, rows_, dimensions_, static_cast<float*>(distanceBuffer_.data()),
            static_cast<unsigned long long*>(indexBuffer_.data()));
    } else if (tiled_) {
        tiledNearestNeighborKernel<false><<<static_cast<unsigned int>((queryCount + 7) / 8),
            threadsPerBlock, 4 * dimensions_ * sizeof(float), stream_.get()>>>(
            static_cast<const float*>(queryBuffer_.data()),
            static_cast<const float*>(bankBuffer_.data()), queryCount, rows_, dimensions_,
            static_cast<float*>(distanceBuffer_.data()),
            static_cast<unsigned long long*>(indexBuffer_.data()));
    } else if (warpParallel_) {
        warpNearestNeighborKernel<false><<<static_cast<unsigned int>(queryCount), threadsPerBlock,
            dimensions_ * sizeof(float), stream_.get()>>>(
            static_cast<const float*>(queryBuffer_.data()),
            static_cast<const float*>(bankBuffer_.data()), queryCount, rows_, dimensions_,
            static_cast<float*>(distanceBuffer_.data()),
            static_cast<unsigned long long*>(indexBuffer_.data()));
    } else {
    nearestNeighborKernel<<<static_cast<unsigned int>(queryCount), threadsPerBlock, 0, stream_.get()>>>(
        static_cast<const float*>(queryBuffer_.data()),
        static_cast<const float*>(bankBuffer_.data()),
        queryCount,
        rows_,
        dimensions_,
        static_cast<float*>(distanceBuffer_.data()),
        static_cast<unsigned long long*>(indexBuffer_.data())
    );
    }
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
    if (transposeDeviceInput_) {
        nchwToPatchMajorKernel<<<dim3(static_cast<unsigned int>((queryCount + 31) / 32),
                                     static_cast<unsigned int>((dimensions_ + 31) / 32)),
                                dim3(32, 8), 0, stream_.get()>>>(
            deviceNchw, static_cast<float*>(queryBuffer_.data()), queryCount, dimensions_);
        checkCuda(cudaGetLastError(), "GPU NCHW transpose launch failed");
        doubleBufferedNearestNeighborKernel<false><<<static_cast<unsigned int>((queryCount + 7) / 8),
            threadsPerBlock, 8 * dimensions_ * sizeof(float), stream_.get()>>>(
            static_cast<const float*>(queryBuffer_.data()), static_cast<const float*>(bankBuffer_.data()),
            queryCount, rows_, dimensions_, static_cast<float*>(distanceBuffer_.data()),
            static_cast<unsigned long long*>(indexBuffer_.data()));
    } else if (doubleBuffer_) {
        doubleBufferedNearestNeighborKernel<true><<<static_cast<unsigned int>((queryCount + 7) / 8),
            threadsPerBlock, 8 * dimensions_ * sizeof(float), stream_.get()>>>(
            deviceNchw, static_cast<const float*>(bankBuffer_.data()),
            queryCount, rows_, dimensions_, static_cast<float*>(distanceBuffer_.data()),
            static_cast<unsigned long long*>(indexBuffer_.data()));
    } else if (cacheQuery_) {
        tiledNearestNeighborKernel<true, true><<<static_cast<unsigned int>((queryCount + 7) / 8),
            threadsPerBlock, 12 * dimensions_ * sizeof(float), stream_.get()>>>(
            deviceNchw, static_cast<const float*>(bankBuffer_.data()),
            queryCount, rows_, dimensions_, static_cast<float*>(distanceBuffer_.data()),
            static_cast<unsigned long long*>(indexBuffer_.data()));
    } else if (tiled_) {
        tiledNearestNeighborKernel<true><<<static_cast<unsigned int>((queryCount + 7) / 8),
            threadsPerBlock, 4 * dimensions_ * sizeof(float), stream_.get()>>>(
            deviceNchw, static_cast<const float*>(bankBuffer_.data()), queryCount, rows_, dimensions_,
            static_cast<float*>(distanceBuffer_.data()),
            static_cast<unsigned long long*>(indexBuffer_.data()));
    } else if (warpParallel_) {
        warpNearestNeighborKernel<true><<<static_cast<unsigned int>(queryCount), threadsPerBlock,
            dimensions_ * sizeof(float), stream_.get()>>>(
            deviceNchw, static_cast<const float*>(bankBuffer_.data()), queryCount, rows_, dimensions_,
            static_cast<float*>(distanceBuffer_.data()),
            static_cast<unsigned long long*>(indexBuffer_.data()));
    } else {
    nearestNeighborNchwKernel<<<static_cast<unsigned int>(queryCount), threadsPerBlock, 0, stream_.get()>>>(
        deviceNchw,
        static_cast<const float*>(bankBuffer_.data()),
        queryCount,
        rows_,
        dimensions_,
        static_cast<float*>(distanceBuffer_.data()),
        static_cast<unsigned long long*>(indexBuffer_.data())
    );
    }
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
