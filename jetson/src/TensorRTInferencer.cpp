#include "TensorRTInferencer.hpp"

#include <NvInferRuntime.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>


void TensorRTInferencer::Logger::log(
    Severity severity,
    const char* message
) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cerr << "[TensorRT] " << message << '\n';
    }
}


std::vector<char> TensorRTInferencer::readEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open TensorRT engine: " + path);
    }

    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + path);
    }

    file.seekg(0, std::ios::beg);
    std::vector<char> bytes(static_cast<std::size_t>(size));
    if (!file.read(bytes.data(), size)) {
        throw std::runtime_error("Failed to read TensorRT engine: " + path);
    }
    return bytes;
}


std::vector<std::int64_t> TensorRTInferencer::toShape(
    const nvinfer1::Dims& dims
) {
    std::vector<std::int64_t> shape;
    shape.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int index = 0; index < dims.nbDims; ++index) {
        shape.push_back(dims.d[index]);
    }
    return shape;
}


std::size_t TensorRTInferencer::elementCount(const nvinfer1::Dims& dims) {
    if (dims.nbDims <= 0) {
        throw std::runtime_error("Tensor shape has no dimensions.");
    }

    std::size_t count = 1;
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) {
            throw std::runtime_error(
                "Only fully specified TensorRT tensor shapes are supported."
            );
        }
        const auto dimension = static_cast<std::size_t>(dims.d[index]);
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::overflow_error("Tensor element count overflow.");
        }
        count *= dimension;
    }
    return count;
}


TensorRTInferencer::TensorRTInferencer(const std::string& enginePath) {
    const std::vector<char> engineBytes = readEngine(enginePath);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }

    engine_.reset(runtime_->deserializeCudaEngine(
        engineBytes.data(),
        engineBytes.size()
    ));
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize TensorRT engine.");
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create TensorRT execution context.");
    }

    for (int index = 0; index < engine_->getNbIOTensors(); ++index) {
        const char* name = engine_->getIOTensorName(index);
        if (name == nullptr) {
            throw std::runtime_error("TensorRT engine contains an unnamed I/O tensor.");
        }

        if (engine_->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error(
                "Only FP32 engine I/O tensors are currently supported: "
                + std::string(name)
            );
        }

        const nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            if (!inputName_.empty()) {
                throw std::runtime_error("Expected exactly one engine input tensor.");
            }
            inputName_ = name;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            if (!outputName_.empty()) {
                throw std::runtime_error("Expected exactly one engine output tensor.");
            }
            outputName_ = name;
        }
    }

    if (inputName_.empty() || outputName_.empty()) {
        throw std::runtime_error("TensorRT engine must have one input and one output.");
    }

    const nvinfer1::Dims inputDims = context_->getTensorShape(inputName_.c_str());
    const nvinfer1::Dims outputDims = context_->getTensorShape(outputName_.c_str());
    inputShape_ = toShape(inputDims);
    outputShape_ = toShape(outputDims);
    inputElements_ = elementCount(inputDims);
    outputElements_ = elementCount(outputDims);

    inputBuffer_.allocate(inputElements_ * sizeof(float));
    outputBuffer_.allocate(outputElements_ * sizeof(float));

    if (!context_->setTensorAddress(inputName_.c_str(), inputBuffer_.data())
        || !context_->setTensorAddress(outputName_.c_str(), outputBuffer_.data())) {
        throw std::runtime_error("Failed to bind TensorRT I/O tensor addresses.");
    }
}


std::vector<float> TensorRTInferencer::infer(
    const std::vector<float>& input
) {
    if (input.size() != inputElements_) {
        throw std::invalid_argument(
            "Input element count does not match TensorRT engine input."
        );
    }

    std::vector<float> output(outputElements_);
    checkCuda(
        cudaMemcpyAsync(
            inputBuffer_.data(),
            input.data(),
            inputBuffer_.sizeBytes(),
            cudaMemcpyHostToDevice,
            stream_.get()
        ),
        "Input cudaMemcpyAsync failed"
    );

    if (!context_->enqueueV3(stream_.get())) {
        throw std::runtime_error("TensorRT enqueueV3 failed.");
    }

    checkCuda(
        cudaMemcpyAsync(
            output.data(),
            outputBuffer_.data(),
            outputBuffer_.sizeBytes(),
            cudaMemcpyDeviceToHost,
            stream_.get()
        ),
        "Output cudaMemcpyAsync failed"
    );
    checkCuda(cudaStreamSynchronize(stream_.get()), "CUDA stream sync failed");
    return output;
}


const std::vector<std::int64_t>& TensorRTInferencer::inputShape() const noexcept {
    return inputShape_;
}


const std::vector<std::int64_t>& TensorRTInferencer::outputShape() const noexcept {
    return outputShape_;
}


std::size_t TensorRTInferencer::inputElementCount() const noexcept {
    return inputElements_;
}


std::size_t TensorRTInferencer::outputElementCount() const noexcept {
    return outputElements_;
}
