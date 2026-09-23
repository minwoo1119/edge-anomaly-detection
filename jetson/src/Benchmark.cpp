#include "Benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <ctime>
#include <vector>

#ifndef EDGE_GIT_COMMIT
#define EDGE_GIT_COMMIT "unknown"
#endif

#ifndef EDGE_GIT_DIRTY
#define EDGE_GIT_DIRTY 1
#endif

#ifndef EDGE_BUILD_TYPE
#define EDGE_BUILD_TYPE "unknown"
#endif

namespace {
double percentile(std::vector<double> sorted, double fraction) {
    std::sort(sorted.begin(), sorted.end());
    const double position = fraction * static_cast<double>(sorted.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

std::string timestampUtc() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_r(&time, &utc);
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

std::vector<double> field(
    const std::vector<StageTimings>& samples,
    double StageTimings::*member
) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const auto& sample : samples) values.push_back(sample.*member);
    return values;
}

void printStage(const char* name, const std::vector<double>& values) {
    const SummaryStatistics stats = summarize(values);
    std::cout << std::left << std::setw(14) << name
              << " mean=" << std::setw(10) << stats.mean
              << " median=" << std::setw(10) << stats.median
              << " std=" << std::setw(10) << stats.standardDeviation
              << " min=" << std::setw(10) << stats.minimum
              << " max=" << std::setw(10) << stats.maximum
              << " p95=" << stats.p95 << '\n';
}

std::string csvCell(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (const char character : value) {
        escaped += character == '\"' ? "\"\"" : std::string(1, character);
    }
    escaped += '\"';
    return escaped;
}
}  // namespace

SummaryStatistics summarize(const std::vector<double>& values) {
    if (values.empty()) throw std::invalid_argument("Cannot summarize an empty sample set.");
    SummaryStatistics result;
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double squaredDifferenceSum = 0.0;
    for (const double value : values) {
        const double difference = value - result.mean;
        squaredDifferenceSum += difference * difference;
    }
    result.standardDeviation = std::sqrt(squaredDifferenceSum / values.size());
    const auto bounds = std::minmax_element(values.begin(), values.end());
    result.minimum = *bounds.first;
    result.maximum = *bounds.second;
    result.median = percentile(values, 0.5);
    result.p95 = percentile(values, 0.95);
    return result;
}

void printBenchmarkSummary(const std::vector<StageTimings>& samples) {
    std::cout << "stage latency (ms), samples=" << samples.size() << '\n';
    printStage("preprocess", field(samples, &StageTimings::preprocessMs));
    printStage("h2d", field(samples, &StageTimings::h2dMs));
    printStage("tensorrt", field(samples, &StageTimings::trtMs));
    printStage("d2h", field(samples, &StageTimings::d2hMs));
    printStage("reshape", field(samples, &StageTimings::reshapeMs));
    printStage("nn", field(samples, &StageTimings::nnMs));
    printStage("postprocess", field(samples, &StageTimings::postprocessMs));
    printStage("total", field(samples, &StageTimings::totalMs));
}

const char* benchmarkBuildType() noexcept {
    return EDGE_BUILD_TYPE;
}

const char* benchmarkGitCommit() noexcept {
    return EDGE_GIT_COMMIT;
}

bool benchmarkGitDirty() noexcept {
    return EDGE_GIT_DIRTY != 0;
}

void writeBenchmarkCsv(
    const std::string& path,
    const RuntimeConfig& config,
    const BenchmarkMetadata& metadata,
    const std::vector<StageTimings>& samples,
    std::size_t memoryBankBytes,
    std::size_t engineBytes
) {
    const std::filesystem::path csvPath(path);
    if (csvPath.has_parent_path()) std::filesystem::create_directories(csvPath.parent_path());
    const std::string header =
        "timestamp,git_commit,git_dirty,build_type,device,jetpack,cuda,tensorrt,opencv,"
        "power_mode,category,model,precision,coreset_ratio,bank_precision,nn_backend,run_id,"
        "iteration,decision_enabled,threshold,threshold_space,threshold_source,config_path,"
        "engine_path,memory_bank_path,image_path,config_sha256,engine_sha256,memory_bank_sha256,"
        "image_sha256,preprocess_ms,h2d_ms,trt_ms,d2h_ms,reshape_ms,nn_ms,post_ms,total_ms,fps,"
        "bank_memory_mb,engine_size_mb";
    const bool writeHeader = !std::filesystem::exists(csvPath) || std::filesystem::file_size(csvPath) == 0;
    if (!writeHeader) {
        std::ifstream existing(csvPath);
        std::string existingHeader;
        std::getline(existing, existingHeader);
        if (existingHeader != header) {
            throw std::runtime_error(
                "Benchmark CSV schema mismatch; use a new output file: " + path
            );
        }
    }
    std::ofstream output(csvPath, std::ios::app);
    if (!output) throw std::runtime_error("Failed to open benchmark CSV: " + path);
    if (writeHeader) output << header << '\n';
    const std::string timestamp = timestampUtc();
    const double bankMb = static_cast<double>(memoryBankBytes) / (1024.0 * 1024.0);
    const double engineMb = static_cast<double>(engineBytes) / (1024.0 * 1024.0);
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const StageTimings& sample = samples[index];
        const double fps = sample.totalMs > 0.0 ? 1000.0 / sample.totalMs : 0.0;
        output << timestamp << ',' << EDGE_GIT_COMMIT << ',' << (EDGE_GIT_DIRTY ? "true" : "false")
               << ',' << csvCell(EDGE_BUILD_TYPE) << ',' << csvCell(config.device) << ','
               << csvCell(config.jetpack) << ',' << csvCell(config.cudaVersion) << ','
               << csvCell(config.tensorrtVersion) << ',' << csvCell(config.opencvVersion) << ','
               << csvCell(config.powerMode) << ',' << csvCell(config.category) << ','
               << csvCell(config.model) << ',' << csvCell(config.precision) << ','
               << config.coresetRatio << ',' << csvCell(config.bankPrecision) << ','
               << csvCell(config.nnBackend) << ',' << csvCell(config.runId) << ',' << index << ','
               << (config.decisionEnabled ? "true" : "false") << ',' << config.threshold << ','
               << csvCell(config.thresholdSpace) << ',' << csvCell(config.thresholdSource) << ','
               << csvCell(metadata.configPath) << ',' << csvCell(config.enginePath) << ','
               << csvCell(config.memoryBankPath) << ',' << csvCell(metadata.imagePath) << ','
               << metadata.configSha256 << ',' << metadata.engineSha256 << ','
               << metadata.memoryBankSha256 << ',' << metadata.imageSha256 << ','
               << sample.preprocessMs << ',' << sample.h2dMs << ','
               << sample.trtMs << ',' << sample.d2hMs << ',' << sample.reshapeMs << ','
               << sample.nnMs << ',' << sample.postprocessMs << ',' << sample.totalMs << ','
               << fps << ',' << bankMb << ',' << engineMb << '\n';
    }
}
