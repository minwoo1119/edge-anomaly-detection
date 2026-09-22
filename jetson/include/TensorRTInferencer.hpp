#pragma once

#include "CudaBuffer.hpp"

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct TensorRTTimings {
    double h2dMs{0.0};
    double inferenceMs{0.0};
    double d2hMs{0.0};
};

class TensorRTInferencer {
public:
    explicit TensorRTInferencer(const std::string& enginePath);

    TensorRTInferencer(const TensorRTInferencer&) = delete;
    TensorRTInferencer& operator=(const TensorRTInferencer&) = delete;

    std::vector<float> infer(
        const std::vector<float>& input,
        TensorRTTimings* timings = nullptr
    );

    const std::vector<std::int64_t>& inputShape() const noexcept;
    const std::vector<std::int64_t>& outputShape() const noexcept;
    std::size_t inputElementCount() const noexcept;
    std::size_t outputElementCount() const noexcept;

private:
    class Logger final : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* message) noexcept override;
    };

    static std::vector<char> readEngine(const std::string& path);
    static std::vector<std::int64_t> toShape(const nvinfer1::Dims& dims);
    static std::size_t elementCount(const nvinfer1::Dims& dims);

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    std::string inputName_;
    std::string outputName_;
    std::vector<std::int64_t> inputShape_;
    std::vector<std::int64_t> outputShape_;
    std::size_t inputElements_{0};
    std::size_t outputElements_{0};
    CudaBuffer inputBuffer_;
    CudaBuffer outputBuffer_;
    CudaStream stream_;
    CudaEvent h2dStart_;
    CudaEvent h2dEnd_;
    CudaEvent inferenceEnd_;
    CudaEvent d2hEnd_;
};
