#include "Preprocessor.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr
            << "Usage: ./edge_anomaly <image_path>"
            << std::endl;

        return 1;
    }

    const std::string imagePath = argv[1];

    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        std::cerr
            << "Failed to load image: "
            << imagePath
            << std::endl;

        return 1;
    }

    constexpr int inputWidth = 256;
    constexpr int inputHeight = 256;

    Preprocessor preprocessor(
        inputWidth,
        inputHeight
    );

    const std::vector<float> inputTensor =
        preprocessor.preprocess(image);

    std::cout
        << "Original image: "
        << image.cols
        << " x "
        << image.rows
        << std::endl;

    std::cout
        << "Tensor elements: "
        << inputTensor.size()
        << std::endl;

    std::cout
        << "Expected elements: "
        << 3 * inputWidth * inputHeight
        << std::endl;

    return 0;
}