#include "Preprocessor.hpp"

#include <opencv2/opencv.hpp>

#include <stdexcept>
#include <vector>


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

    cv::Mat resized;
    cv::resize(
        image,
        resized,
        cv::Size(
            inputWidth_,
            inputHeight_
        )
    );

    cv::Mat rgb;
    cv::cvtColor(
        resized,
        rgb,
        cv::COLOR_BGR2RGB
    );

    cv::Mat floatImage;
    rgb.convertTo(
        floatImage,
        CV_32FC3,
        1.0 / 255.0
    );

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