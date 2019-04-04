#pragma once
#include <opencv2/core.hpp>

Mat_<double> vectorNorm(InputArray src, int axis);

Mat_<double> calculateAngleBetweenVectors(const std::vector<cv::Vec4d> v1,
    const std::vector<double> v2, int axis = 0, bool directed = true);
