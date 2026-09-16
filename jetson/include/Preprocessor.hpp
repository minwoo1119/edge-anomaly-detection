#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

class Preprocessor {
public:
    Preprocessor(int inputWidth, int inputHeight);

    std::vector<float> preprocess(
        const cv::Mat& image
    ) const;

private:
    int inputWidth_;
    int inputHeight_;
};