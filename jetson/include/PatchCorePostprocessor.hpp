#pragma once

#include "MemoryBank.hpp"
#include "NearestNeighborSearch.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>


struct PatchCoreResult {
    float score{0.0F};
    cv::Mat anomalyMap;
    std::vector<float> patchEmbeddings;
    std::vector<float> patchScores;
    std::vector<std::size_t> nearestIndices;
};

struct PostprocessTimings {
    double reshapeMs{0.0};
    double nearestNeighborMs{0.0};
    double postprocessMs{0.0};
};


class PatchCorePostprocessor {
public:
    PatchCorePostprocessor(
        int outputWidth,
        int outputHeight,
        std::size_t numNeighbors = 9,
        double gaussianSigma = 4.0
    );

    PatchCoreResult process(
        const std::vector<float>& nchwEmbedding,
        std::size_t channels,
        std::size_t featureHeight,
        std::size_t featureWidth,
        const MemoryBank& memoryBank,
        const INearestNeighborSearch& search,
        PostprocessTimings* timings = nullptr
    ) const;

    static std::vector<float> nchwToPatchMajor(
        const std::vector<float>& nchwEmbedding,
        std::size_t channels,
        std::size_t height,
        std::size_t width
    );

private:
    float weightedImageScore(
        const std::vector<float>& queries,
        std::size_t dimensions,
        const SearchResult& nearest,
        const MemoryBank& memoryBank
    ) const;

    int outputWidth_;
    int outputHeight_;
    std::size_t numNeighbors_;
    double gaussianSigma_;
};
