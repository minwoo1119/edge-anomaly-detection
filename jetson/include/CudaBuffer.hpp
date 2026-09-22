#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>


inline void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status)
        );
    }
}


class CudaBuffer {
public:
    CudaBuffer() = default;

    explicit CudaBuffer(std::size_t bytes) {
        allocate(bytes);
    }

    ~CudaBuffer() {
        release();
    }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          bytes_(std::exchange(other.bytes_, 0)) {}

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            bytes_ = std::exchange(other.bytes_, 0);
        }
        return *this;
    }

    void allocate(std::size_t bytes) {
        if (bytes == 0) {
            throw std::invalid_argument("CUDA buffer size must be positive.");
        }
        release();
        checkCuda(cudaMalloc(&data_, bytes), "cudaMalloc failed");
        bytes_ = bytes;
    }

    void* data() noexcept {
        return data_;
    }

    const void* data() const noexcept {
        return data_;
    }

    std::size_t sizeBytes() const noexcept {
        return bytes_;
    }

private:
    void release() noexcept {
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
            bytes_ = 0;
        }
    }

    void* data_{nullptr};
    std::size_t bytes_{0};
};


class CudaStream {
public:
    CudaStream() {
        checkCuda(
            cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags failed"
        );
    }

    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    cudaStream_t get() const noexcept {
        return stream_;
    }

private:
    cudaStream_t stream_{nullptr};
};
