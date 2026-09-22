#pragma once

#include <cstddef>
#include <string>

struct RuntimeConfig {
    std::string enginePath;
    std::string memoryBankPath;
    std::string category;
    std::string model;
    std::string precision;
    std::string bankPrecision;
    std::string nnBackend;
    std::string device;
    std::string jetpack;
    std::string cudaVersion;
    std::string tensorrtVersion;
    std::string opencvVersion;
    std::string powerMode;
    std::string runId;
    int inputWidth{0};
    int inputHeight{0};
    float threshold{0.0F};
    double coresetRatio{0.0};
    std::size_t numNeighbors{0};
    double gaussianSigma{0.0};
    int warmup{0};
    int repeats{0};

    static RuntimeConfig load(const std::string& path);
};
