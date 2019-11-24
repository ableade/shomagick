#include "cudaSurfFeatureDetector.hpp"

using std::vector;
using cv::InputArray;
using cv::OutputArray;
using cv::KeyPoint;

CudaSurfFeatureDetector::CudaSurfFeatureDetector(
    int minFeatures,
    int minHessian,
    int nOctaves,
    int nOctaveLayers,
    bool extended,
    float keyPointsRatio,
    bool upright
) :minFeatures_(minFeatures), cudaSurf_(
    minHessian,
    nOctaves,
    nOctaveLayers,
    extended,
    keyPointsRatio,
    upright
) {}

cv::Ptr<CudaSurfFeatureDetector> CudaSurfFeatureDetector::create(
    int minFeatures,
    int minHessian,
    int nOctaves,
    int nOctaveLayers,
    bool extended,
    float keyPointsRatio,
    bool upright
)
{
    return cv::makePtr<CudaSurfFeatureDetector>(
        minFeatures,
        minHessian,
        nOctaves,
        nOctaveLayers,
        extended,
        keyPointsRatio,
        upright
        );
}

void CudaSurfFeatureDetector::detect(InputArray image, vector<KeyPoint>& keypoints, InputArray mask) {
    auto hessianThreshold = 3000;
    while (keypoints.size() < minFeatures_ && cudaSurf_.hessianThreshold > 0.0001) {
        cudaSurf_.hessianThreshold = hessianThreshold;
        cudaSurf_(image.getGpuMat(), mask.getGpuMat(), keypoints);
        hessianThreshold = (hessianThreshold * 2) / 3;
    }
}

void CudaSurfFeatureDetector::compute(InputArray image, vector<KeyPoint>& keypoints, OutputArray descriptors) {
    cv::cuda::GpuMat mask;
    auto & gDesc = descriptors.getGpuMatRef();
    cudaSurf_(image.getGpuMat(), mask, keypoints, gDesc, true);
}

void CudaSurfFeatureDetector::detectAndCompute(
    InputArray image,
    cv::InputArray mask,
    vector<KeyPoint>& keypoints,
    OutputArray &descriptors,
    bool useProvidedKeypoints
) {
    auto & gDesc = descriptors.getGpuMatRef();
    auto hessianThreshold = 3000;
    while (descriptors.rows() < minFeatures_ && hessianThreshold > 0.0001) {
        cudaSurf_.hessianThreshold = hessianThreshold;
        cudaSurf_(image.getGpuMat(), mask.getGpuMat(), keypoints, gDesc, useProvidedKeypoints);
        hessianThreshold = (hessianThreshold * 2) / 3;
    }
}


