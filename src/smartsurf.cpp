#include "smartsurf.h"

using cv::xfeatures2d::SURF;

SmartSurfDetector::SmartSurfDetector(
    double hessianThreshold,
    int nOctaves,
    int nOctaveLayers,
    bool extended,
    bool upright,
    int minFeatures
):minFeatures_(minFeatures) {
    surf_ = SURF::create(
        hessianThreshold, 
        nOctaves, 
        nOctaveLayers, 
        extended, 
        upright
    );
}

void SmartSurfDetector::detectAndCompute(
    cv::InputArray img, 
    cv::InputArray mask, 
    CV_OUT std::vector<cv::KeyPoint>& keypoints, 
    cv::OutputArray descriptors, 
    bool useProvidedKeypoints
)
{
    auto hessianThreshold = 3000;
    while (descriptors.rows() < minFeatures_ && hessianThreshold > 0.0001) {
        surf_->setHessianThreshold(hessianThreshold);
        surf_->detectAndCompute(img, mask, keypoints, descriptors, useProvidedKeypoints);
        hessianThreshold = (hessianThreshold * 2) / 3;
    } 
}

cv::Ptr<SmartSurfDetector> SmartSurfDetector::create(
    double hessianThreshold, 
    int nOctaves, 
    int nOctaveLayers, 
    bool extended, 
    bool upright, 
    int minFeatures
)
{
    return cv::makePtr<SmartSurfDetector>(
        hessianThreshold, 
        nOctaves,
        nOctaveLayers,
        extended,
        upright,
        minFeatures
        );
}
