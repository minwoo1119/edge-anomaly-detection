#include "CublasNearestNeighborSearch.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int threadsPerBlock = 256;

void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuBLAS status "
            + std::to_string(static_cast<int>(status))
        );
    }
}

int checkedInt(std::size_t value, const char* name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error(std::string(name) + " exceeds the cuBLAS int32 limit.");
    }
    return static_cast<int>(value);
}

__global__ void rowSquaredNormKernel(
    const float* values, std::size_t rows, std::size_t dimensions, float* norms
) {
    const std::size_t row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0F;
    for (std::size_t d = threadIdx.x; d < dimensions; d += blockDim.x) {
        const float value = values[row * dimensions + d];
        sum = fmaf(value, value, sum);
    }
    __shared__ float partial[threadsPerBlock];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) partial[threadIdx.x] += partial[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) norms[row] = partial[0];
}

// dotProducts is a column-major [bankRows, queryCount] cuBLAS result.
__global__ void reduceL2Kernel(
    const float* dotProducts,
    const float* bankNorms,
    const float* queryNorms,
    std::size_t bankRows,
    std::size_t queryCount,
    float* distances,
    unsigned long long* indices
) {
    const std::size_t query = blockIdx.x;
    if (query >= queryCount) return;
    float best = CUDART_INF_F;
    unsigned long long bestIndex = 0;
    const float* dots = dotProducts + query * bankRows;
    for (std::size_t row = threadIdx.x; row < bankRows; row += blockDim.x) {
        // Round-off can make a mathematically non-negative squared distance
        // slightly negative. Clamp before sqrt while retaining FP32 accumulation.
        const float squared = fmaxf(
            0.0F, queryNorms[query] + bankNorms[row] - 2.0F * dots[row]
        );
        if (squared < best || (squared == best && row < bestIndex)) {
            best = squared;
            bestIndex = static_cast<unsigned long long>(row);
        }
    }
    __shared__ float blockDistances[threadsPerBlock];
    __shared__ unsigned long long blockIndices[threadsPerBlock];
    blockDistances[threadIdx.x] = best;
    blockIndices[threadIdx.x] = bestIndex;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            const float candidate = blockDistances[threadIdx.x + offset];
            const unsigned long long candidateIndex = blockIndices[threadIdx.x + offset];
            if (candidate < blockDistances[threadIdx.x]
                || (candidate == blockDistances[threadIdx.x]
                    && candidateIndex < blockIndices[threadIdx.x])) {
                blockDistances[threadIdx.x] = candidate;
                blockIndices[threadIdx.x] = candidateIndex;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        distances[query] = sqrtf(blockDistances[0]);
        indices[query] = blockIndices[0];
    }
}
}  // namespace

CublasNearestNeighborSearch::CublasNearestNeighborSearch(
    const MemoryBank& memoryBank, std::size_t maximumQueries
) : rows_(memoryBank.rows()),
    dimensions_(memoryBank.dimensions()),
    maximumQueries_(maximumQueries),
    bankBuffer_(memoryBank.sizeBytes()),
    bankNormBuffer_(rows_ * sizeof(float)),
    queryBuffer_(maximumQueries_ * dimensions_ * sizeof(float)),
    queryNormBuffer_(maximumQueries_ * sizeof(float)),
    dotBuffer_(rows_ * maximumQueries_ * sizeof(float)),
    distanceBuffer_(maximumQueries_ * sizeof(float)),
    indexBuffer_(maximumQueries_ * sizeof(unsigned long long)) {
    if (maximumQueries_ == 0) {
        throw std::invalid_argument("cuBLAS NN maximum query count must be positive.");
    }
    checkedInt(rows_, "Memory-bank rows");
    checkedInt(dimensions_, "Embedding dimensions");
    checkedInt(maximumQueries_, "Maximum query count");
    checkCublas(cublasCreate(&handle_), "cublasCreate");
    try {
        checkCublas(cublasSetStream(handle_, stream_.get()), "cublasSetStream");
        checkCublas(cublasSetMathMode(handle_, CUBLAS_PEDANTIC_MATH), "cublasSetMathMode");
        checkCuda(cudaMemcpyAsync(
            bankBuffer_.data(), memoryBank.values().data(), memoryBank.sizeBytes(),
            cudaMemcpyHostToDevice, stream_.get()
        ), "cuBLAS NN memory-bank H2D copy failed");
        rowSquaredNormKernel<<<static_cast<unsigned int>(rows_), threadsPerBlock, 0, stream_.get()>>>(
            static_cast<const float*>(bankBuffer_.data()), rows_, dimensions_,
            static_cast<float*>(bankNormBuffer_.data())
        );
        checkCuda(cudaGetLastError(), "cuBLAS NN bank-norm kernel launch failed");
        checkCuda(cudaStreamSynchronize(stream_.get()), "cuBLAS NN initialization sync failed");
    } catch (...) {
        cublasDestroy(handle_);
        handle_ = nullptr;
        throw;
    }
}

CublasNearestNeighborSearch::~CublasNearestNeighborSearch() {
    if (handle_ != nullptr) cublasDestroy(handle_);
}

SearchResult CublasNearestNeighborSearch::search(
    const float* queries,
    std::size_t queryCount,
    std::size_t dimensions,
    const MemoryBank& memoryBank
) const {
    if (queries == nullptr || queryCount == 0 || queryCount > maximumQueries_) {
        throw std::invalid_argument("Invalid cuBLAS NN query buffer or query count.");
    }
    if (dimensions != dimensions_ || memoryBank.dimensions() != dimensions_
        || memoryBank.rows() != rows_) {
        throw std::invalid_argument("cuBLAS NN memory-bank shape changed after initialization.");
    }

    searchStart_.record(stream_.get());
    checkCuda(cudaMemcpyAsync(
        queryBuffer_.data(), queries, queryCount * dimensions_ * sizeof(float),
        cudaMemcpyHostToDevice, stream_.get()
    ), "cuBLAS NN query H2D copy failed");
    rowSquaredNormKernel<<<static_cast<unsigned int>(queryCount), threadsPerBlock, 0, stream_.get()>>>(
        static_cast<const float*>(queryBuffer_.data()), queryCount, dimensions_,
        static_cast<float*>(queryNormBuffer_.data())
    );
    checkCuda(cudaGetLastError(), "cuBLAS NN query-norm kernel launch failed");

    const float alpha = 1.0F;
    const float beta = 0.0F;
    // Row-major bank [R,D] and queries [Q,D] are interpreted as column-major
    // [D,R] and [D,Q]. op(bank)^T * queries yields [R,Q].
    checkCublas(cublasSgemm(
        handle_, CUBLAS_OP_T, CUBLAS_OP_N,
        checkedInt(rows_, "Memory-bank rows"),
        checkedInt(queryCount, "Query count"),
        checkedInt(dimensions_, "Embedding dimensions"),
        &alpha,
        static_cast<const float*>(bankBuffer_.data()), checkedInt(dimensions_, "Bank lda"),
        static_cast<const float*>(queryBuffer_.data()), checkedInt(dimensions_, "Query ldb"),
        &beta,
        static_cast<float*>(dotBuffer_.data()), checkedInt(rows_, "Output ldc")
    ), "cublasSgemm");
    reduceL2Kernel<<<static_cast<unsigned int>(queryCount), threadsPerBlock, 0, stream_.get()>>>(
        static_cast<const float*>(dotBuffer_.data()),
        static_cast<const float*>(bankNormBuffer_.data()),
        static_cast<const float*>(queryNormBuffer_.data()),
        rows_, queryCount,
        static_cast<float*>(distanceBuffer_.data()),
        static_cast<unsigned long long*>(indexBuffer_.data())
    );
    checkCuda(cudaGetLastError(), "cuBLAS NN L2 reduction kernel launch failed");

    SearchResult result;
    result.distances.resize(queryCount);
    std::vector<unsigned long long> rawIndices(queryCount);
    checkCuda(cudaMemcpyAsync(
        result.distances.data(), distanceBuffer_.data(), queryCount * sizeof(float),
        cudaMemcpyDeviceToHost, stream_.get()
    ), "cuBLAS NN distance D2H copy failed");
    checkCuda(cudaMemcpyAsync(
        rawIndices.data(), indexBuffer_.data(), queryCount * sizeof(unsigned long long),
        cudaMemcpyDeviceToHost, stream_.get()
    ), "cuBLAS NN index D2H copy failed");
    searchEnd_.record(stream_.get());
    searchEnd_.synchronize();
    result.deviceMilliseconds = CudaEvent::elapsedMilliseconds(searchStart_, searchEnd_);
    result.indices.reserve(queryCount);
    for (const auto index : rawIndices) result.indices.push_back(static_cast<std::size_t>(index));
    return result;
}
