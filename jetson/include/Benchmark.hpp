#pragma once

#include "RuntimeConfig.hpp"

#include <cstddef>
#include <string>
#include <vector>

struct StageTimings {
    double preprocessMs{0.0};
    double h2dMs{0.0};
    double trtMs{0.0};
    double d2hMs{0.0};
    double reshapeMs{0.0};
    double nnMs{0.0};
    double postprocessMs{0.0};
    double totalMs{0.0};
};

struct SummaryStatistics {
    double mean{0.0};
    double median{0.0};
    double standardDeviation{0.0};
    double minimum{0.0};
    double maximum{0.0};
    double p95{0.0};
};

SummaryStatistics summarize(const std::vector<double>& values);
void printBenchmarkSummary(const std::vector<StageTimings>& samples);
void writeBenchmarkCsv(
    const std::string& path,
    const RuntimeConfig& config,
    const std::vector<StageTimings>& samples,
    std::size_t memoryBankBytes,
    std::size_t engineBytes
);
