#pragma once
#include <opencv2/xfeatures2d.hpp>

class SmartSurfDetector : public cv::Feature2D {
public:
    SmartSurfDetector(double hessianThreshold = 3000,
        int nOctaves = 4,
        int nOctaveLayers = 2,
        bool extended = true,
        bool upright = false,
        int minFeatures = 4000
    );

    void detectAndCompute(cv::InputArray img, cv::InputArray mask,
        CV_OUT std::vector<cv::KeyPoint>& keypoints,
        cv::OutputArray descriptors,
        bool useProvidedKeypoints = false);

    static cv::Ptr<SmartSurfDetector> create(
        double hessianThreshold = 3000,
        int nOctaves = 4,
        int nOctaveLayers = 2,
        bool extended = true,
        bool upright = false,
        int minFeatures = 4000);

private:
    int minFeatures_;
    cv::Ptr<cv::xfeatures2d::SURF> surf_;
};
