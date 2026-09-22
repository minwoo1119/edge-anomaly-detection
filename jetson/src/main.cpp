#include "MemoryBank.hpp"
#include "NearestNeighborSearch.hpp"
#include "PatchCorePostprocessor.hpp"
#include "Preprocessor.hpp"
#include "RuntimeConfig.hpp"
#include "TensorRTInferencer.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct CommandLine {
    std::string configPath;
    std::string imagePath;
    std::string heatmapPath;
};

CommandLine parseCommandLine(int argc, char* argv[]) {
    CommandLine commandLine;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            std::cout << "Usage: edge_anomaly --config <config.yaml> --image <image> "
                      << "[--heatmap <output.png>]\n";
            std::exit(0);
        }
        if ((argument == "--config" || argument == "--image" || argument == "--heatmap")
            && index + 1 >= argc) {
            throw std::invalid_argument("Missing value for argument: " + argument);
        }
        if (argument == "--config") commandLine.configPath = argv[++index];
        else if (argument == "--image") commandLine.imagePath = argv[++index];
        else if (argument == "--heatmap") commandLine.heatmapPath = argv[++index];
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
        const CpuBruteForceSearch nearestNeighborSearch;
        const PatchCorePostprocessor postprocessor(
            config.inputWidth, config.inputHeight, config.numNeighbors, config.gaussianSigma
        );

        const std::vector<float> input = preprocessor.preprocess(image);
        const std::vector<float> embedding = inferencer.infer(input);
        const auto& outputShape = inferencer.outputShape();
        const PatchCoreResult result = postprocessor.process(
            embedding,
            static_cast<std::size_t>(outputShape[1]),
            static_cast<std::size_t>(outputShape[2]),
            static_cast<std::size_t>(outputShape[3]),
            memoryBank,
            nearestNeighborSearch
        );

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
