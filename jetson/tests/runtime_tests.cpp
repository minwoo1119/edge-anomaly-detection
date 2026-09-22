#include "MemoryBank.hpp"
#include "NearestNeighborSearch.hpp"
#include "NpyWriter.hpp"
#include "PatchCorePostprocessor.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
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

void testPostprocessing() {
    const std::vector<float> nchw{0.0F, 3.0F, 0.0F, 4.0F};
    const MemoryBank bank({0.0F, 0.0F, 2.0F, 4.0F}, 2, 2);
    const CpuBruteForceSearch search;
    const PatchCorePostprocessor postprocessor(4, 2, 1, 0.0);
    const PatchCoreResult result = postprocessor.process(nchw, 2, 1, 2, bank, search);
    require(std::abs(result.score - 1.0F) < 1e-6F, "Image score is incorrect");
    require(result.anomalyMap.rows == 2 && result.anomalyMap.cols == 4, "Anomaly map shape is incorrect");
    require(result.patchScores.size() == 2, "Patch score count is incorrect");
}
}  // namespace

int main() {
    try {
        testNpyRoundTrip();
        testPatchLayoutAndNearestNeighbor();
        testPostprocessing();
        std::cout << "All runtime tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Runtime test failed: " << error.what() << '\n';
        return 1;
    }
}
