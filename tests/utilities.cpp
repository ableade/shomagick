#include "stdafx.h"
#include "utilities.h"


Camera getPerspectiveCamera(float physicalLens, int height, int width, float k1, float k2) {
    const auto pixelFocal = physicalLens * width;

    cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) <<
        pixelFocal,
        0.,
        width / 2,
        0.,
        pixelFocal,
        height / 2,
        0.,
        0.,
        1
        );

    cv::Mat dist = (cv::Mat_<double>(4, 1) <<
        k1,
        k2,
        0.,
        0.
        );

    return { cameraMatrix, dist, height, width };
}