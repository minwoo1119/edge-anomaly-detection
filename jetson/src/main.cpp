#include "Benchmark.hpp"
#include "CudaNearestNeighborSearch.hpp"
#include "MemoryBank.hpp"
#include "NearestNeighborSearch.hpp"
#include "NpyWriter.hpp"
#include "PatchCorePostprocessor.hpp"
#include "Preprocessor.hpp"
#include "RuntimeConfig.hpp"
#include "TensorRTInferencer.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
struct CommandLine {
    std::string configPath;
    std::string imagePath;
    std::string heatmapPath;
    std::string benchmarkCsvPath;
    std::string embeddingPath;
    std::string anomalyMapPath;
};

CommandLine parseCommandLine(int argc, char* argv[]) {
    CommandLine commandLine;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            std::cout << "Usage: edge_anomaly --config <config.yaml> --image <image> "
                      << "[--heatmap <output.png>] [--benchmark-csv <output.csv>] "
                      << "[--dump-embedding <output.npy>] [--dump-map <output.npy>]\n";
            std::exit(0);
        }
        if ((argument == "--config" || argument == "--image" || argument == "--heatmap"
             || argument == "--benchmark-csv" || argument == "--dump-embedding"
             || argument == "--dump-map") && index + 1 >= argc) {
            throw std::invalid_argument("Missing value for argument: " + argument);
        }
        if (argument == "--config") commandLine.configPath = argv[++index];
        else if (argument == "--image") commandLine.imagePath = argv[++index];
        else if (argument == "--heatmap") commandLine.heatmapPath = argv[++index];
        else if (argument == "--benchmark-csv") commandLine.benchmarkCsvPath = argv[++index];
        else if (argument == "--dump-embedding") commandLine.embeddingPath = argv[++index];
        else if (argument == "--dump-map") commandLine.anomalyMapPath = argv[++index];
        else throw std::invalid_argument("Unknown argument: " + argument);
    }
    if (commandLine.configPath.empty() || commandLine.imagePath.empty()) {
        throw std::invalid_argument("--config and --image are required.");
    }
    return commandLine;
}

void validateTensorShapes(const TensorRTInferencer& inferencer, const RuntimeConfig& config, const MemoryBank& bank) {
    const auto& input = inferencer.inputShape();
    if (input.size() != 4 || input[0] != 1 || input[1] != 3
        || input[2] != config.inputHeight || input[3] != config.inputWidth) {
        throw std::runtime_error("Engine input must be [1, 3, input_height, input_width].");
    }
    const auto& output = inferencer.outputShape();
    if (output.size() != 4 || output[0] != 1 || output[1] <= 0 || output[2] <= 0 || output[3] <= 0) {
        throw std::runtime_error("Engine output must be a static [1, C, H, W] tensor.");
    }
    if (static_cast<std::size_t>(output[1]) != bank.dimensions()) {
        throw std::runtime_error("Engine embedding channels do not match memory bank dimensions.");
    }
}

void saveHeatmap(const cv::Mat& anomalyMap, const std::string& path) {
    double minimum = 0.0;
    double maximum = 0.0;
    cv::minMaxLoc(anomalyMap, &minimum, &maximum);
    cv::Mat normalized;
    if (maximum > minimum) {
        anomalyMap.convertTo(normalized, CV_8UC1, 255.0 / (maximum - minimum), -255.0 * minimum / (maximum - minimum));
    } else {
        normalized = cv::Mat::zeros(anomalyMap.size(), CV_8UC1);
    }
    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_JET);
    if (!cv::imwrite(path, colored)) {
        throw std::runtime_error("Failed to save anomaly heatmap: " + path);
    }
}

struct InferenceRun {
    PatchCoreResult result;
    StageTimings timings;
    std::vector<float> embedding;
};

InferenceRun runInference(
    const cv::Mat& image,
    const Preprocessor& preprocessor,
    TensorRTInferencer& inferencer,
    const PatchCorePostprocessor& postprocessor,
    const MemoryBank& memoryBank,
    const INearestNeighborSearch& nearestNeighborSearch
) {
    const auto totalStart = std::chrono::steady_clock::now();
    const auto preprocessStart = totalStart;
    const std::vector<float> input = preprocessor.preprocess(image);
    const auto preprocessEnd = std::chrono::steady_clock::now();
    TensorRTTimings trtTimings;
    std::vector<float> embedding = inferencer.infer(input, &trtTimings);
    const auto& outputShape = inferencer.outputShape();
    PostprocessTimings postprocessTimings;
    PatchCoreResult result = postprocessor.process(
        embedding,
        static_cast<std::size_t>(outputShape[1]),
        static_cast<std::size_t>(outputShape[2]),
        static_cast<std::size_t>(outputShape[3]),
        memoryBank,
        nearestNeighborSearch,
        &postprocessTimings
    );
    const auto totalEnd = std::chrono::steady_clock::now();
    const auto milliseconds = [](const auto& start, const auto& end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };
    StageTimings timings;
    timings.preprocessMs = milliseconds(preprocessStart, preprocessEnd);
    timings.h2dMs = trtTimings.h2dMs;
    timings.trtMs = trtTimings.inferenceMs;
    timings.d2hMs = trtTimings.d2hMs;
    timings.reshapeMs = postprocessTimings.reshapeMs;
    timings.nnMs = postprocessTimings.nearestNeighborMs;
    timings.postprocessMs = postprocessTimings.postprocessMs;
    timings.totalMs = milliseconds(totalStart, totalEnd);
    return {std::move(result), timings, std::move(embedding)};
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
        const CommandLine commandLine = parseCommandLine(argc, argv);
        const RuntimeConfig config = RuntimeConfig::load(commandLine.configPath);
        const cv::Mat image = cv::imread(commandLine.imagePath, cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("Failed to load image: " + commandLine.imagePath);

        const MemoryBank memoryBank = MemoryBank::loadNpy(config.memoryBankPath);
        TensorRTInferencer inferencer(config.enginePath);
        validateTensorShapes(inferencer, config, memoryBank);
        const Preprocessor preprocessor(config.inputWidth, config.inputHeight);
        const auto& outputShape = inferencer.outputShape();
        std::unique_ptr<INearestNeighborSearch> nearestNeighborSearch;
        if (config.nnBackend == "cuda") {
            nearestNeighborSearch = std::make_unique<CudaBruteForceSearch>(
                memoryBank,
                static_cast<std::size_t>(outputShape[2] * outputShape[3])
            );
        } else {
            nearestNeighborSearch = std::make_unique<CpuBruteForceSearch>();
        }
        const PatchCorePostprocessor postprocessor(
            config.inputWidth, config.inputHeight, config.numNeighbors, config.gaussianSigma
        );

        InferenceRun run;
        if (commandLine.benchmarkCsvPath.empty()) {
            run = runInference(
                image,
                preprocessor,
                inferencer,
                postprocessor,
                memoryBank,
                *nearestNeighborSearch
            );
        } else {
            for (int iteration = 0; iteration < config.warmup; ++iteration) {
                runInference(
                    image,
                    preprocessor,
                    inferencer,
                    postprocessor,
                    memoryBank,
                    *nearestNeighborSearch
                );
            }
            std::vector<StageTimings> samples;
            samples.reserve(static_cast<std::size_t>(config.repeats));
            for (int iteration = 0; iteration < config.repeats; ++iteration) {
                run = runInference(
                    image,
                    preprocessor,
                    inferencer,
                    postprocessor,
                    memoryBank,
                    *nearestNeighborSearch
                );
                samples.push_back(run.timings);
            }
            printBenchmarkSummary(samples);
            writeBenchmarkCsv(
                commandLine.benchmarkCsvPath,
                config,
                samples,
                memoryBank.sizeBytes(),
                static_cast<std::size_t>(std::filesystem::file_size(config.enginePath))
            );
        }

        const PatchCoreResult& result = run.result;

        if (!commandLine.embeddingPath.empty()) {
            writeFloatNpy(
                commandLine.embeddingPath,
                run.embedding.data(),
                run.embedding.size(),
                {
                    1,
                    static_cast<std::size_t>(outputShape[1]),
                    static_cast<std::size_t>(outputShape[2]),
                    static_cast<std::size_t>(outputShape[3])
                }
            );
        }
        if (!commandLine.anomalyMapPath.empty()) {
            if (!result.anomalyMap.isContinuous()) {
                throw std::runtime_error("Anomaly map output is not contiguous.");
            }
            writeFloatNpy(
                commandLine.anomalyMapPath,
                result.anomalyMap.ptr<float>(),
                result.anomalyMap.total(),
                {
                    static_cast<std::size_t>(result.anomalyMap.rows),
                    static_cast<std::size_t>(result.anomalyMap.cols)
                }
            );
        }

        if (!commandLine.heatmapPath.empty()) saveHeatmap(result.anomalyMap, commandLine.heatmapPath);
        std::cout << "category=" << config.category << '\n'
                  << "score=" << result.score << '\n'
                  << "threshold=" << config.threshold << '\n'
                  << "prediction=" << (result.score >= config.threshold ? "NG" : "OK") << '\n'
                  << "embedding_shape=1x" << outputShape[1] << 'x' << outputShape[2] << 'x' << outputShape[3] << '\n'
                  << "memory_bank_shape=" << memoryBank.rows() << 'x' << memoryBank.dimensions() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
