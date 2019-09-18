#include "cudasurf.hpp"

using std::vector;
using cv::InputArray;
using cv::OutputArray;
using cv::KeyPoint;

CudaSurfFeatureDetector::CudaSurfFeatureDetector(
        double hessianThreshold, 
        int nOctaves,
        int nOctaveLayers, 
        bool extended, 
        float keyPointsRatio, 
        bool upright
    ):cudaSurf_(
        hessianThreshold, 
        nOctaves,
        nOctaveLayers, 
        extended, 
        keyPointsRatio, 
        upright
    ) {}

 cv::Ptr<CudaSurfFeatureDetector> CudaSurfFeatureDetector::create(
     double hessianThreshold, 
     int nOctaves,
     int nOctaveLayers,
     bool extended, 
     float keyPointsRatio,
     bool upright
 ) 
 {
     return cv::makePtr<CudaSurfFeatureDetector>(hessianThreshold, nOctaves, nOctaveLayers, extended, keyPointsRatio, upright);
 }

 void CudaSurfFeatureDetector::detect(InputArray image, vector<KeyPoint>& keypoints, InputArray mask) {
     cudaSurf_(image.getGpuMat(), mask.getGpuMat(), keypoints);
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
){
    auto & gDesc = descriptors.getGpuMatRef();
    cudaSurf_(image.getGpuMat(), mask.getGpuMat(), keypoints, gDesc, useProvidedKeypoints);
}


