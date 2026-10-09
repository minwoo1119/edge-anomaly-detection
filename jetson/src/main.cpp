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

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
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
    std::string inputTensorPath;
    std::string patchScoresPath;
    std::string patchEmbeddingsPath;
    std::string nearestIndicesPath;
    std::string rawScorePath;
    std::string benchmarkReadyPath;
    std::string benchmarkStartPath;
    BenchmarkMetadata benchmarkMetadata;
    std::string runIdOverride;
    bool preprocessOnly{false};
};

CommandLine parseCommandLine(int argc, char* argv[]) {
    CommandLine commandLine;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--help") {
            std::cout << "Usage: edge_anomaly --config <config.yaml> --image <image> "
                      << "[--heatmap <output.png>] [--benchmark-csv <output.csv>] "
                      << "[--dump-input <output.npy>] [--dump-embedding <output.npy>] "
                      << "[--dump-map <output.npy>] [--dump-patch-scores <output.npy>] "
                      << "[--dump-patches <output.npy>] "
                      << "[--dump-nn-indices <output.npy>] [--dump-score <output.npy>] "
                      << "[--run-id <id>] [--config-sha256 <hash>] [--engine-sha256 <hash>] "
                      << "[--memory-bank-sha256 <hash>] [--image-sha256 <hash>] "
                      << "[--benchmark-ready <path>] [--benchmark-start <path>] "
                      << "[--preprocess-only]\n";
            std::exit(0);
        }
        if ((argument == "--config" || argument == "--image" || argument == "--heatmap"
             || argument == "--benchmark-csv" || argument == "--dump-embedding"
             || argument == "--dump-map" || argument == "--dump-input"
             || argument == "--dump-patch-scores" || argument == "--dump-nn-indices"
             || argument == "--dump-score" || argument == "--dump-patches"
             || argument == "--run-id" || argument == "--config-sha256"
             || argument == "--engine-sha256" || argument == "--memory-bank-sha256"
             || argument == "--image-sha256" || argument == "--benchmark-ready"
             || argument == "--benchmark-start") && index + 1 >= argc) {
            throw std::invalid_argument("Missing value for argument: " + argument);
        }
        if (argument == "--config") commandLine.configPath = argv[++index];
        else if (argument == "--image") commandLine.imagePath = argv[++index];
        else if (argument == "--heatmap") commandLine.heatmapPath = argv[++index];
        else if (argument == "--benchmark-csv") commandLine.benchmarkCsvPath = argv[++index];
        else if (argument == "--dump-embedding") commandLine.embeddingPath = argv[++index];
        else if (argument == "--dump-map") commandLine.anomalyMapPath = argv[++index];
        else if (argument == "--dump-input") commandLine.inputTensorPath = argv[++index];
        else if (argument == "--dump-patch-scores") commandLine.patchScoresPath = argv[++index];
        else if (argument == "--dump-patches") commandLine.patchEmbeddingsPath = argv[++index];
        else if (argument == "--dump-nn-indices") commandLine.nearestIndicesPath = argv[++index];
        else if (argument == "--dump-score") commandLine.rawScorePath = argv[++index];
        else if (argument == "--run-id") commandLine.runIdOverride = argv[++index];
        else if (argument == "--config-sha256") commandLine.benchmarkMetadata.configSha256 = argv[++index];
        else if (argument == "--engine-sha256") commandLine.benchmarkMetadata.engineSha256 = argv[++index];
        else if (argument == "--memory-bank-sha256") commandLine.benchmarkMetadata.memoryBankSha256 = argv[++index];
        else if (argument == "--image-sha256") commandLine.benchmarkMetadata.imageSha256 = argv[++index];
        else if (argument == "--benchmark-ready") commandLine.benchmarkReadyPath = argv[++index];
        else if (argument == "--benchmark-start") commandLine.benchmarkStartPath = argv[++index];
        else if (argument == "--preprocess-only") commandLine.preprocessOnly = true;
        else throw std::invalid_argument("Unknown argument: " + argument);
    }
    if (commandLine.configPath.empty() || commandLine.imagePath.empty()) {
        throw std::invalid_argument("--config and --image are required.");
    }
    if (commandLine.preprocessOnly && commandLine.inputTensorPath.empty()) {
        throw std::invalid_argument("--preprocess-only requires --dump-input.");
    }
    if (commandLine.benchmarkReadyPath.empty() != commandLine.benchmarkStartPath.empty()) {
        throw std::invalid_argument(
            "--benchmark-ready and --benchmark-start must be provided together."
        );
    }
    if (!commandLine.benchmarkReadyPath.empty() && commandLine.benchmarkCsvPath.empty()) {
        throw std::invalid_argument("Benchmark synchronization requires --benchmark-csv.");
    }
    commandLine.benchmarkMetadata.configPath = commandLine.configPath;
    commandLine.benchmarkMetadata.imagePath = commandLine.imagePath;
    return commandLine;
}

void synchronizeBenchmarkStart(const CommandLine& commandLine) {
    if (commandLine.benchmarkReadyPath.empty()) return;
    const std::filesystem::path ready(commandLine.benchmarkReadyPath);
    const std::filesystem::path start(commandLine.benchmarkStartPath);
    if (ready.has_parent_path()) std::filesystem::create_directories(ready.parent_path());
    if (std::filesystem::exists(ready) || std::filesystem::exists(start)) {
        throw std::runtime_error("Benchmark synchronization files already exist.");
    }
    {
        std::ofstream signal(ready);
        if (!signal) throw std::runtime_error("Failed to write benchmark ready signal.");
        signal << "warmup_complete\n";
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (!std::filesystem::exists(start)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("Timed out waiting for benchmark start signal.");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void validateBenchmarkInvocation(const CommandLine& commandLine) {
    if (commandLine.benchmarkCsvPath.empty()) return;
    if (std::string(benchmarkBuildType()) != "Release") {
        throw std::runtime_error(
            "Benchmarking requires a Release build; current build type is "
            + std::string(benchmarkBuildType())
        );
    }
    if (benchmarkGitDirty()) {
        std::cerr << "WARNING: benchmark binary was built from a dirty worktree.\n";
    }
    const BenchmarkMetadata& metadata = commandLine.benchmarkMetadata;
    if (metadata.configSha256.empty() || metadata.engineSha256.empty()
        || metadata.memoryBankSha256.empty() || metadata.imageSha256.empty()) {
        throw std::runtime_error(
            "Benchmarking requires config, engine, memory-bank, and image SHA-256 metadata. "
            "Use jetson/scripts/run_benchmark.sh."
        );
    }
    const auto validSha256 = [](const std::string& value) {
        return value.size() == 64
            && std::all_of(value.begin(), value.end(), [](unsigned char character) {
                return std::isxdigit(character) != 0;
            });
    };
    if (!validSha256(metadata.configSha256) || !validSha256(metadata.engineSha256)
        || !validSha256(metadata.memoryBankSha256) || !validSha256(metadata.imageSha256)) {
        throw std::runtime_error("Benchmark artifact hashes must be 64-character SHA-256 values.");
    }
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
    std::vector<float> input;
    std::vector<float> embedding;
};

struct PreparedInput {
    std::vector<float> values;
    std::chrono::steady_clock::time_point started;
    double preprocessMs{0.0};
};

PreparedInput prepareInput(const cv::Mat& image, const Preprocessor& preprocessor) {
    const auto start = std::chrono::steady_clock::now();
    std::vector<float> values = preprocessor.preprocess(image);
    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    return {std::move(values), start, elapsed};
}

class PreprocessPipeline {
public:
    PreprocessPipeline(const cv::Mat& image, const Preprocessor& preprocessor)
        : image_(image), preprocessor_(preprocessor), worker_(&PreprocessPipeline::work, this) {}

    ~PreprocessPipeline() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        condition_.notify_all();
        worker_.join();
    }

    PreprocessPipeline(const PreprocessPipeline&) = delete;
    PreprocessPipeline& operator=(const PreprocessPipeline&) = delete;

    void request() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (requested_ || ready_) {
            throw std::logic_error("Preprocessing pipeline queue is full.");
        }
        requested_ = true;
        condition_.notify_all();
    }

    PreparedInput take() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return ready_ || error_ != nullptr; });
        if (error_ != nullptr) std::rethrow_exception(error_);
        PreparedInput output = std::move(*result_);
        result_.reset();
        ready_ = false;
        return output;
    }

private:
    void work() noexcept {
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] { return stop_ || requested_; });
                if (stop_) return;
                requested_ = false;
            }
            try {
                PreparedInput prepared = prepareInput(image_, preprocessor_);
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (stop_) return;
                    result_.emplace(std::move(prepared));
                    ready_ = true;
                }
                condition_.notify_all();
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    error_ = std::current_exception();
                }
                condition_.notify_all();
                return;
            }
        }
    }

    const cv::Mat& image_;
    const Preprocessor& preprocessor_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::optional<PreparedInput> result_;
    std::exception_ptr error_;
    bool requested_{false};
    bool ready_{false};
    bool stop_{false};
    std::thread worker_;
};

InferenceRun runPreparedInference(
    PreparedInput prepared,
    TensorRTInferencer& inferencer,
    const PatchCorePostprocessor& postprocessor,
    const MemoryBank& memoryBank,
    const INearestNeighborSearch& nearestNeighborSearch,
    bool gpuResidentNn
) {
    std::vector<float> input = std::move(prepared.values);
    const auto totalStart = prepared.started;
    TensorRTTimings trtTimings;
    const auto& outputShape = inferencer.outputShape();
    PostprocessTimings postprocessTimings;
    std::vector<float> embedding;
    PatchCoreResult result;
    if (gpuResidentNn) {
        const auto* cudaSearch = dynamic_cast<const CudaBruteForceSearch*>(
            &nearestNeighborSearch
        );
        if (cudaSearch == nullptr) {
            throw std::logic_error("GPU-resident NN requires CudaBruteForceSearch.");
        }
        inferencer.inferToDevice(input, &trtTimings);
        result = postprocessor.processDeviceNchw(
            inferencer.deviceOutput(),
            static_cast<std::size_t>(outputShape[1]),
            static_cast<std::size_t>(outputShape[2]),
            static_cast<std::size_t>(outputShape[3]),
            memoryBank,
            *cudaSearch,
            &postprocessTimings
        );
    } else {
        embedding = inferencer.infer(input, &trtTimings);
        result = postprocessor.process(
            embedding,
            static_cast<std::size_t>(outputShape[1]),
            static_cast<std::size_t>(outputShape[2]),
            static_cast<std::size_t>(outputShape[3]),
            memoryBank,
            nearestNeighborSearch,
            &postprocessTimings
        );
    }
    const auto totalEnd = std::chrono::steady_clock::now();
    StageTimings timings;
    timings.preprocessMs = prepared.preprocessMs;
    timings.h2dMs = trtTimings.h2dMs;
    timings.trtMs = trtTimings.inferenceMs;
    timings.d2hMs = trtTimings.d2hMs;
    timings.reshapeMs = postprocessTimings.reshapeMs;
    timings.nnMs = postprocessTimings.nearestNeighborMs;
    timings.postprocessMs = postprocessTimings.postprocessMs;
    timings.totalMs = std::chrono::duration<double, std::milli>(
        totalEnd - totalStart
    ).count();
    timings.pipelineIntervalMs = timings.totalMs;
    return {std::move(result), timings, std::move(input), std::move(embedding)};
}

InferenceRun runInference(
    const cv::Mat& image,
    const Preprocessor& preprocessor,
    TensorRTInferencer& inferencer,
    const PatchCorePostprocessor& postprocessor,
    const MemoryBank& memoryBank,
    const INearestNeighborSearch& nearestNeighborSearch,
    bool gpuResidentNn
) {
    return runPreparedInference(
        prepareInput(image, preprocessor),
        inferencer,
        postprocessor,
        memoryBank,
        nearestNeighborSearch,
        gpuResidentNn
    );
}
}  // namespace

int main(int argc, char* argv[]) {
    try {
        const CommandLine commandLine = parseCommandLine(argc, argv);
        RuntimeConfig config = RuntimeConfig::load(commandLine.configPath);
        if (!commandLine.runIdOverride.empty()) config.runId = commandLine.runIdOverride;
        validateBenchmarkInvocation(commandLine);
        const cv::Mat image = cv::imread(commandLine.imagePath, cv::IMREAD_COLOR);
        if (image.empty()) throw std::runtime_error("Failed to load image: " + commandLine.imagePath);

        const Preprocessor preprocessor(config.inputWidth, config.inputHeight);
        if (commandLine.preprocessOnly) {
            const std::vector<float> input = preprocessor.preprocess(image);
            writeFloatNpy(
                commandLine.inputTensorPath,
                input.data(),
                input.size(),
                {
                    1,
                    3,
                    static_cast<std::size_t>(config.inputHeight),
                    static_cast<std::size_t>(config.inputWidth)
                }
            );
            std::cout << "preprocessed_input=" << commandLine.inputTensorPath << '\n'
                      << "input_shape=1x3x" << config.inputHeight << 'x' << config.inputWidth << '\n';
            return 0;
        }

        const MemoryBank memoryBank = MemoryBank::loadNpy(config.memoryBankPath);
        TensorRTInferencer inferencer(config.enginePath, config.optimizationStageIndex());
        validateTensorShapes(inferencer, config, memoryBank);
        const auto& outputShape = inferencer.outputShape();
        std::unique_ptr<INearestNeighborSearch> nearestNeighborSearch;
        if (config.nnBackend == "cuda" || config.nnBackend == "cuda_warp" || config.nnBackend == "cuda_tiled" || config.nnBackend == "cuda_tiled_cached" || config.nnBackend == "cuda_tiled_async") {
            nearestNeighborSearch = std::make_unique<CudaBruteForceSearch>(
                memoryBank,
                static_cast<std::size_t>(outputShape[2] * outputShape[3]),
                config.nnBackend == "cuda_warp",
                config.nnBackend == "cuda_tiled",
                config.nnBackend == "cuda_tiled_cached",
                config.nnBackend == "cuda_tiled_async"
            );
        } else {
            nearestNeighborSearch = std::make_unique<CpuBruteForceSearch>(
                config.nnBackend == "openmp"
            );
        }
        const PatchCorePostprocessor postprocessor(
            config.inputWidth, config.inputHeight, config.numNeighbors, config.gaussianSigma
        );
        const bool gpuResidentNn = config.optimizationStageIndex() >= 5;

        InferenceRun run;
        if (commandLine.benchmarkCsvPath.empty()) {
            run = runInference(
                image,
                preprocessor,
                inferencer,
                postprocessor,
                memoryBank,
                *nearestNeighborSearch,
                gpuResidentNn
            );
        } else {
            for (int iteration = 0; iteration < config.warmup; ++iteration) {
                runInference(
                    image,
                    preprocessor,
                    inferencer,
                    postprocessor,
                    memoryBank,
                    *nearestNeighborSearch,
                    gpuResidentNn
                );
            }
            synchronizeBenchmarkStart(commandLine);
            std::vector<StageTimings> samples;
            samples.reserve(static_cast<std::size_t>(config.repeats));
            if (config.optimizationStageIndex() == 6) {
                PreprocessPipeline pipeline(image, preprocessor);
                PreparedInput current = prepareInput(image, preprocessor);
                pipeline.request();
                runPreparedInference(
                    std::move(current), inferencer, postprocessor, memoryBank,
                    *nearestNeighborSearch, gpuResidentNn
                );
                current = pipeline.take();
                auto previousCompletion = std::chrono::steady_clock::now();
                for (int iteration = 0; iteration < config.repeats; ++iteration) {
                    pipeline.request();
                    run = runPreparedInference(
                        std::move(current), inferencer, postprocessor, memoryBank,
                        *nearestNeighborSearch, gpuResidentNn
                    );
                    const auto completion = std::chrono::steady_clock::now();
                    run.timings.pipelineIntervalMs =
                        std::chrono::duration<double, std::milli>(
                            completion - previousCompletion
                        ).count();
                    previousCompletion = completion;
                    samples.push_back(run.timings);
                    current = pipeline.take();
                }
            } else {
                for (int iteration = 0; iteration < config.repeats; ++iteration) {
                    run = runInference(
                        image,
                        preprocessor,
                        inferencer,
                        postprocessor,
                        memoryBank,
                        *nearestNeighborSearch,
                        gpuResidentNn
                    );
                    samples.push_back(run.timings);
                }
            }
            printBenchmarkSummary(samples);
            writeBenchmarkCsv(
                commandLine.benchmarkCsvPath,
                config,
                commandLine.benchmarkMetadata,
                samples,
                memoryBank.rows(),
                memoryBank.sizeBytes(),
                static_cast<std::size_t>(std::filesystem::file_size(config.enginePath))
            );
        }

        if (gpuResidentNn
            && (!commandLine.embeddingPath.empty()
                || !commandLine.patchEmbeddingsPath.empty())) {
            run.embedding = inferencer.copyDeviceOutputToHost();
        }
        if (gpuResidentNn && !commandLine.patchEmbeddingsPath.empty()) {
            run.result.patchEmbeddings = PatchCorePostprocessor::nchwToPatchMajor(
                run.embedding,
                static_cast<std::size_t>(outputShape[1]),
                static_cast<std::size_t>(outputShape[2]),
                static_cast<std::size_t>(outputShape[3])
            );
        }
        const PatchCoreResult& result = run.result;

        if (!commandLine.inputTensorPath.empty()) {
            writeFloatNpy(
                commandLine.inputTensorPath,
                run.input.data(),
                run.input.size(),
                {
                    1,
                    3,
                    static_cast<std::size_t>(config.inputHeight),
                    static_cast<std::size_t>(config.inputWidth)
                }
            );
        }
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
        if (!commandLine.patchScoresPath.empty()) {
            writeFloatNpy(
                commandLine.patchScoresPath,
                result.patchScores.data(),
                result.patchScores.size(),
                {
                    static_cast<std::size_t>(outputShape[2]),
                    static_cast<std::size_t>(outputShape[3])
                }
            );
        }
        if (!commandLine.patchEmbeddingsPath.empty()) {
            writeFloatNpy(
                commandLine.patchEmbeddingsPath,
                result.patchEmbeddings.data(),
                result.patchEmbeddings.size(),
                {
                    static_cast<std::size_t>(outputShape[2] * outputShape[3]),
                    static_cast<std::size_t>(outputShape[1])
                }
            );
        }
        if (!commandLine.nearestIndicesPath.empty()) {
            std::vector<std::uint64_t> indices;
            indices.reserve(result.nearestIndices.size());
            for (const std::size_t index : result.nearestIndices) {
                indices.push_back(static_cast<std::uint64_t>(index));
            }
            writeUint64Npy(
                commandLine.nearestIndicesPath,
                indices.data(),
                indices.size(),
                {
                    static_cast<std::size_t>(outputShape[2]),
                    static_cast<std::size_t>(outputShape[3])
                }
            );
        }
        if (!commandLine.rawScorePath.empty()) {
            writeFloatNpy(commandLine.rawScorePath, &result.score, 1, {1});
        }

        if (!commandLine.heatmapPath.empty()) saveHeatmap(result.anomalyMap, commandLine.heatmapPath);
        std::cout << "category=" << config.category << '\n'
                  << "score=" << result.score << '\n'
                  << "threshold_space=" << config.thresholdSpace << '\n'
                  << "threshold_source=" << config.thresholdSource << '\n';
        if (config.decisionEnabled) {
            std::cout << "threshold=" << config.threshold << '\n'
                      << "prediction=" << (result.score >= config.threshold ? "NG" : "OK") << '\n';
        } else {
            std::cout << "threshold=disabled\n"
                      << "prediction=UNAVAILABLE\n";
        }
        std::cout
                  << "embedding_shape=1x" << outputShape[1] << 'x' << outputShape[2] << 'x' << outputShape[3] << '\n'
                  << "memory_bank_shape=" << memoryBank.rows() << 'x' << memoryBank.dimensions() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
