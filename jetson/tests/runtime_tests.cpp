#include "CudaNearestNeighborSearch.hpp"
#include "MemoryBank.hpp"
#include "NearestNeighborSearch.hpp"
#include "NpyWriter.hpp"
#include "PatchCorePostprocessor.hpp"
#include "Preprocessor.hpp"
#include "RuntimeConfig.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void testNpyRoundTrip() {
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "edge_runtime_test.npy";
    const std::vector<float> values{1.0F, -2.5F, 3.25F, 4.0F, 5.0F, 6.0F};
    writeFloatNpy(path.string(), values.data(), values.size(), {2, 3});
    const MemoryBank bank = MemoryBank::loadNpy(path.string());
    std::filesystem::remove(path);
    require(bank.rows() == 2 && bank.dimensions() == 3, "NPY shape round-trip failed");
    require(bank.values() == values, "NPY values round-trip failed");
}

void testPreprocessorChannelOrderAndNormalization() {
    const cv::Mat image(1, 1, CV_8UC3, cv::Scalar(0, 128, 255));
    const Preprocessor preprocessor(1, 1);
    const std::vector<float> tensor = preprocessor.preprocess(image);
    require(tensor.size() == 3, "Preprocessor output shape is incorrect");
    const std::vector<float> expected{
        (1.0F - 0.485F) / 0.229F,
        ((128.0F / 255.0F) - 0.456F) / 0.224F,
        (0.0F - 0.406F) / 0.225F,
    };
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(
            std::abs(tensor[index] - expected[index]) < 1e-6F,
            "Preprocessor channel order or normalization is incorrect"
        );
    }
}

void testPatchLayoutAndNearestNeighbor() {
    const std::vector<float> nchw{0.0F, 3.0F, 0.0F, 4.0F};
    const std::vector<float> patches =
        PatchCorePostprocessor::nchwToPatchMajor(nchw, 2, 1, 2);
    require(
        patches == std::vector<float>({0.0F, 0.0F, 3.0F, 4.0F}),
        "NCHW to patch-major conversion failed"
    );

    const MemoryBank bank({0.0F, 0.0F, 2.0F, 4.0F}, 2, 2);
    const CpuBruteForceSearch search;
    const SearchResult nearest = search.search(patches.data(), 2, 2, bank);
    require(nearest.indices == std::vector<std::size_t>({0, 1}), "Nearest indices are incorrect");
    require(std::abs(nearest.distances[0]) < 1e-6F, "First nearest distance is incorrect");
    require(std::abs(nearest.distances[1] - 1.0F) < 1e-6F, "Second nearest distance is incorrect");
}

void testCpuCudaAgreement() {
    const MemoryBank bank({0.0F, 0.0F, 2.0F, 4.0F, -1.0F, 1.0F}, 3, 2);
    const std::vector<float> queries{0.0F, 0.0F, 3.0F, 4.0F, -2.0F, 1.0F};
    const CpuBruteForceSearch cpu;
    const CudaBruteForceSearch cuda(bank, 3);
    const SearchResult cpuResult = cpu.search(queries.data(), 3, 2, bank);
    const SearchResult cudaResult = cuda.search(queries.data(), 3, 2, bank);
    require(cpuResult.indices == cudaResult.indices, "CPU/CUDA nearest indices differ");
    for (std::size_t index = 0; index < cpuResult.distances.size(); ++index) {
        require(
            std::abs(cpuResult.distances[index] - cudaResult.distances[index]) < 1e-5F,
            "CPU/CUDA nearest distances differ"
        );
    }
}

void testWarpCudaSearch() {
    // Tail dimensions, partial bank traversal, ties across warps, and NCHW input.
    constexpr std::size_t dimensions = 65;
    constexpr std::size_t rows = 17;
    constexpr std::size_t count = 3;
    std::vector<float> values(rows * dimensions);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t d = 0; d < dimensions; ++d)
            values[r * dimensions + d] = static_cast<float>(r + d % 7);
    std::copy_n(values.data(), dimensions, values.data() + 8 * dimensions);
    const MemoryBank bank(values, rows, dimensions);
    std::vector<float> queries(count * dimensions), nchw(queries.size());
    for (std::size_t q = 0; q < count; ++q)
        for (std::size_t d = 0; d < dimensions; ++d) {
            queries[q * dimensions + d] = values[q * 5 * dimensions + d] + 0.125F;
            nchw[d * count + q] = queries[q * dimensions + d];
        }
    const CpuBruteForceSearch cpu;
    const auto expected = cpu.search(queries.data(), count, dimensions, bank);
    const CudaBruteForceSearch optimized(bank, count, true);
    const auto actual = optimized.search(queries.data(), count, dimensions, bank);
    require(actual.indices == expected.indices, "Warp CUDA nearest indices differ");
    require(actual.indices[0] == 0, "Warp CUDA must select lowest tied index");
    CudaBuffer device(nchw.size() * sizeof(float));
    checkCuda(cudaMemcpy(device.data(), nchw.data(), nchw.size() * sizeof(float),
                         cudaMemcpyHostToDevice), "Warp test H2D failed");
    const auto resident = optimized.searchDeviceNchw(
        static_cast<const float*>(device.data()), dimensions, 1, count, bank);
    require(resident.nearest.indices == expected.indices, "Warp NCHW indices differ");
    for (std::size_t q = 0; q < count; ++q) {
        require(std::abs(actual.distances[q] - expected.distances[q]) < 1e-5F,
                "Warp CUDA distance differs");
        require(std::abs(resident.nearest.distances[q] - expected.distances[q]) < 1e-5F,
                "Warp NCHW distance differs");
    }
}

void testCpuCudaTieBreaking() {
    const MemoryBank bank({1.0F, 2.0F, 1.0F, 2.0F, 4.0F, 5.0F}, 3, 2);
    const std::vector<float> queries{1.0F, 2.0F};
    const CpuBruteForceSearch cpu;
    const CudaBruteForceSearch cuda(bank, 1);
    const SearchResult cpuResult = cpu.search(queries.data(), 1, 2, bank);
    const SearchResult cudaResult = cuda.search(queries.data(), 1, 2, bank);
    require(cpuResult.indices[0] == 0, "CPU tie-breaking must select the lowest bank index");
    require(cudaResult.indices[0] == 0, "CUDA tie-breaking must select the lowest bank index");
}

void testDeviceNchwNearestNeighbor() {
    const MemoryBank bank({0.0F, 0.0F, 2.0F, 4.0F, -1.0F, 1.0F}, 3, 2);
    const std::vector<float> patchMajor{0.0F, 0.0F, 3.0F, 4.0F, -2.0F, 1.0F};
    const std::vector<float> nchw{0.0F, 3.0F, -2.0F, 0.0F, 4.0F, 1.0F};
    CudaBuffer device(nchw.size() * sizeof(float));
    checkCuda(
        cudaMemcpy(
            device.data(), nchw.data(), nchw.size() * sizeof(float), cudaMemcpyHostToDevice
        ),
        "Test embedding H2D failed"
    );
    const CpuBruteForceSearch cpu;
    const SearchResult cpuResult = cpu.search(patchMajor.data(), 3, 2, bank);
    const CudaBruteForceSearch cuda(bank, 3);
    const auto deviceResult = cuda.searchDeviceNchw(
        static_cast<const float*>(device.data()), 2, 1, 3, bank
    );
    require(cpuResult.indices == deviceResult.nearest.indices, "Device NCHW indices differ");
    for (std::size_t index = 0; index < cpuResult.distances.size(); ++index) {
        require(
            std::abs(cpuResult.distances[index] - deviceResult.nearest.distances[index]) < 1e-5F,
            "Device NCHW distances differ"
        );
    }
    require(
        deviceResult.maximumDistanceQuery == std::vector<float>({3.0F, 4.0F}),
        "Device NCHW maximum-distance query gather failed"
    );
}

void testMemoryBankRejectsNonFiniteValues() {
    bool rejected = false;
    try {
        const MemoryBank bank({0.0F, std::numeric_limits<float>::infinity()}, 1, 2);
        (void)bank;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Memory bank must reject non-finite values");
}

void testPostprocessing() {
    const std::vector<float> nchw{0.0F, 3.0F, 0.0F, 4.0F};
    const MemoryBank bank({0.0F, 0.0F, 2.0F, 4.0F}, 2, 2);
    const CpuBruteForceSearch search;
    const PatchCorePostprocessor postprocessor(4, 2, 1, 0.0);
    const PatchCoreResult result = postprocessor.process(nchw, 2, 1, 2, bank, search);
    require(std::abs(result.score - 1.0F) < 1e-6F, "Image score is incorrect");
    require(result.anomalyMap.rows == 2 && result.anomalyMap.cols == 4, "Anomaly map shape is incorrect");
    require(result.patchScores.size() == 2, "Patch score count is incorrect");
    require(
        result.patchEmbeddings == std::vector<float>({0.0F, 0.0F, 3.0F, 4.0F}),
        "Postprocessor did not preserve patch-major embeddings"
    );
    for (int row = 0; row < result.anomalyMap.rows; ++row) {
        require(std::abs(result.anomalyMap.at<float>(row, 0) - 0.0F) < 1e-6F, "Nearest map resize is incorrect");
        require(std::abs(result.anomalyMap.at<float>(row, 1) - 0.0F) < 1e-6F, "Nearest map resize is incorrect");
        require(std::abs(result.anomalyMap.at<float>(row, 2) - 1.0F) < 1e-6F, "Nearest map resize is incorrect");
        require(std::abs(result.anomalyMap.at<float>(row, 3) - 1.0F) < 1e-6F, "Nearest map resize is incorrect");
    }
}

void testOptimizationStageIndex() {
    RuntimeConfig config;
    for (int stage = 0; stage <= 6; ++stage) {
        config.optimizationStage = "S" + std::to_string(stage);
        require(config.optimizationStageIndex() == stage, "Optimization stage parsing failed");
    }
}
}  // namespace

int main() {
    try {
        testNpyRoundTrip();
        testPreprocessorChannelOrderAndNormalization();
        testPatchLayoutAndNearestNeighbor();
        testCpuCudaAgreement();
        testCpuCudaTieBreaking();
        testWarpCudaSearch();
        testDeviceNchwNearestNeighbor();
        testMemoryBankRejectsNonFiniteValues();
        testPostprocessing();
        testOptimizationStageIndex();
        std::cout << "All runtime tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Runtime test failed: " << error.what() << '\n';
        return 1;
    }
}
