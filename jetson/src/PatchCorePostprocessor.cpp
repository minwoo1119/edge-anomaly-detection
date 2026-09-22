#include "PatchCorePostprocessor.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>


PatchCorePostprocessor::PatchCorePostprocessor(
    int outputWidth,
    int outputHeight,
    std::size_t numNeighbors,
    double gaussianSigma
)
    : outputWidth_(outputWidth),
      outputHeight_(outputHeight),
      numNeighbors_(numNeighbors),
      gaussianSigma_(gaussianSigma) {
    if (outputWidth_ <= 0 || outputHeight_ <= 0) {
        throw std::invalid_argument("Postprocessor output size must be positive.");
    }
    if (numNeighbors_ == 0) {
        throw std::invalid_argument("PatchCore numNeighbors must be positive.");
    }
    if (gaussianSigma_ < 0.0) {
        throw std::invalid_argument("Gaussian sigma cannot be negative.");
    }
}


std::vector<float> PatchCorePostprocessor::nchwToPatchMajor(
    const std::vector<float>& nchwEmbedding,
    std::size_t channels,
    std::size_t height,
    std::size_t width
) {
    if (channels == 0 || height == 0 || width == 0) {
        throw std::invalid_argument("Embedding dimensions must be positive.");
    }
    const std::size_t patchCount = height * width;
    if (nchwEmbedding.size() != channels * patchCount) {
        throw std::invalid_argument("Embedding values do not match NCHW shape.");
    }

    std::vector<float> patches(patchCount * channels);
    for (std::size_t patch = 0; patch < patchCount; ++patch) {
        for (std::size_t channel = 0; channel < channels; ++channel) {
            patches[patch * channels + channel] =
                nchwEmbedding[channel * patchCount + patch];
        }
    }
    return patches;
}


float PatchCorePostprocessor::weightedImageScore(
    const std::vector<float>& queries,
    std::size_t dimensions,
    const SearchResult& nearest,
    const MemoryBank& memoryBank
) const {
    if (nearest.distances.empty()) {
        throw std::invalid_argument("Cannot score an empty patch set.");
    }

    const auto maximum = std::max_element(
        nearest.distances.begin(),
        nearest.distances.end()
    );
    const std::size_t patchIndex = static_cast<std::size_t>(
        std::distance(nearest.distances.begin(), maximum)
    );
    const float maximumDistance = *maximum;
    if (numNeighbors_ == 1 || memoryBank.rows() == 1) {
        return maximumDistance;
    }

    const std::size_t nearestBankIndex = nearest.indices.at(patchIndex);
    const float* anchor = memoryBank.row(nearestBankIndex);
    const std::size_t neighbors = std::min(numNeighbors_, memoryBank.rows());

    std::vector<std::pair<float, std::size_t>> anchorDistances;
    anchorDistances.reserve(memoryBank.rows());
    for (std::size_t bankIndex = 0; bankIndex < memoryBank.rows(); ++bankIndex) {
        const float* candidate = memoryBank.row(bankIndex);
        float squaredDistance = 0.0F;
        for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
            const float difference = anchor[dimension] - candidate[dimension];
            squaredDistance += difference * difference;
        }
        anchorDistances.emplace_back(squaredDistance, bankIndex);
    }

    std::partial_sort(
        anchorDistances.begin(),
        anchorDistances.begin() + static_cast<std::ptrdiff_t>(neighbors),
        anchorDistances.end(),
        [](const auto& left, const auto& right) {
            return left.first < right.first;
        }
    );

    const float* anomalousPatch = queries.data() + patchIndex * dimensions;
    std::vector<float> supportDistances(neighbors);
    for (std::size_t neighbor = 0; neighbor < neighbors; ++neighbor) {
        const float* support = memoryBank.row(anchorDistances[neighbor].second);
        float squaredDistance = 0.0F;
        for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
            const float difference = anomalousPatch[dimension] - support[dimension];
            squaredDistance += difference * difference;
        }
        supportDistances[neighbor] = std::sqrt(squaredDistance);
    }

    const float largest = *std::max_element(
        supportDistances.begin(),
        supportDistances.end()
    );
    float exponentialSum = 0.0F;
    for (const float distance : supportDistances) {
        exponentialSum += std::exp(distance - largest);
    }
    const float nearestProbability =
        std::exp(supportDistances.front() - largest) / exponentialSum;
    return (1.0F - nearestProbability) * maximumDistance;
}


PatchCoreResult PatchCorePostprocessor::process(
    const std::vector<float>& nchwEmbedding,
    std::size_t channels,
    std::size_t featureHeight,
    std::size_t featureWidth,
    const MemoryBank& memoryBank,
    const INearestNeighborSearch& search
) const {
    std::vector<float> queries = nchwToPatchMajor(
        nchwEmbedding,
        channels,
        featureHeight,
        featureWidth
    );
    SearchResult nearest = search.search(
        queries.data(),
        featureHeight * featureWidth,
        channels,
        memoryBank
    );

    const float score = weightedImageScore(
        queries,
        channels,
        nearest,
        memoryBank
    );

    cv::Mat patchMap(
        static_cast<int>(featureHeight),
        static_cast<int>(featureWidth),
        CV_32FC1,
        nearest.distances.data()
    );
    cv::Mat anomalyMap;
    cv::resize(
        patchMap,
        anomalyMap,
        cv::Size(outputWidth_, outputHeight_),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );
    if (gaussianSigma_ > 0.0) {
        cv::GaussianBlur(
            anomalyMap,
            anomalyMap,
            cv::Size(0, 0),
            gaussianSigma_,
            gaussianSigma_,
            cv::BORDER_REFLECT_101
        );
    }

    PatchCoreResult result;
    result.score = score;
    result.anomalyMap = std::move(anomalyMap);
    result.patchScores = std::move(nearest.distances);
    result.nearestIndices = std::move(nearest.indices);
    return result;
}
