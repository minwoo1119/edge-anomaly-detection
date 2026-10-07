#include "Preprocessor.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
struct ResizeWeights {
    int start;
    std::vector<float> values;
};

// Match tensor bilinear resize with antialias=True and align_corners=False.
std::vector<ResizeWeights> resizeWeights(int sourceSize, int targetSize) {
    const float scale = static_cast<float>(sourceSize) / targetSize;
    const float support = std::max(scale, 1.0F);
    std::vector<ResizeWeights> result;
    result.reserve(targetSize);
    for (int output = 0; output < targetSize; ++output) {
        const float center = scale * (output + 0.5F);
        const int start = std::max(static_cast<int>(center - support + 0.5F), 0);
        const int end = std::min(static_cast<int>(center + support + 0.5F), sourceSize);
        ResizeWeights weights{start, {}};
        float sum = 0.0F;
        for (int input = start; input < end; ++input) {
            const float weight = std::max(0.0F,
                1.0F - std::abs((input + 0.5F - center) / support));
            weights.values.push_back(weight);
            sum += weight;
        }
        for (float& weight : weights.values) weight /= sum;
        result.push_back(std::move(weights));
    }
    return result;
}

cv::Mat resizeAntialiased(const cv::Mat& source, int width, int height) {
    const auto horizontal = resizeWeights(source.cols, width);
    const auto vertical = resizeWeights(source.rows, height);
    cv::Mat intermediate(source.rows, width, CV_32FC3);
    for (int y = 0; y < source.rows; ++y) {
        for (int x = 0; x < width; ++x) {
            cv::Vec3f value(0, 0, 0);
            const auto& weights = horizontal[x];
            for (std::size_t i = 0; i < weights.values.size(); ++i)
                value += source.at<cv::Vec3f>(y, weights.start + i) * weights.values[i];
            intermediate.at<cv::Vec3f>(y, x) = value;
        }
    }
    cv::Mat result(height, width, CV_32FC3);
    for (int y = 0; y < height; ++y) {
        const auto& weights = vertical[y];
        for (int x = 0; x < width; ++x) {
            cv::Vec3f value(0, 0, 0);
            for (std::size_t i = 0; i < weights.values.size(); ++i)
                value += intermediate.at<cv::Vec3f>(weights.start + i, x) * weights.values[i];
            result.at<cv::Vec3f>(y, x) = value;
        }
    }
    return result;
}
} // namespace

Preprocessor::Preprocessor(
    int inputWidth,
    int inputHeight
)
    : inputWidth_(inputWidth),
      inputHeight_(inputHeight) {}


std::vector<float> Preprocessor::preprocess(
    const cv::Mat& image
) const {
    if (image.empty()) {
        throw std::runtime_error(
            "Input image is empty."
        );
    }

    cv::Mat rgb;
    cv::cvtColor(
        image,
        rgb,
        cv::COLOR_BGR2RGB
    );

    cv::Mat floatImage;
    rgb.convertTo(
        floatImage,
        CV_32FC3,
        1.0 / 255.0
    );
    floatImage = resizeAntialiased(floatImage, inputWidth_, inputHeight_);

    const std::vector<float> mean = {
        0.485f,
        0.456f,
        0.406f
    };

    const std::vector<float> std = {
        0.229f,
        0.224f,
        0.225f
    };

    std::vector<float> inputTensor(
        3 * inputWidth_ * inputHeight_
    );

    for (int y = 0; y < inputHeight_; ++y) {
        for (int x = 0; x < inputWidth_; ++x) {
            const cv::Vec3f& pixel =
                floatImage.at<cv::Vec3f>(y, x);

            for (int c = 0; c < 3; ++c) {
                const float normalized =
                    (pixel[c] - mean[c]) / std[c];

                const int index =
                    c * inputHeight_ * inputWidth_
                    + y * inputWidth_
                    + x;

                inputTensor[index] = normalized;
            }
        }
    }

    return inputTensor;
}
