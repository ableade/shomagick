#pragma once
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>

class CudaSurfFeatureDetector : public cv::Feature2D {
    
private:
    cv::cuda::SURF_CUDA cudaSurf_;

public:
    CudaSurfFeatureDetector(
        double hessianThreshold, 
        int nOctaves =4, 
        int nOctaveLayers =2, 
        bool extended= false, 
        float keyPointsRatio = 0.01, 
        bool upright = false
    );

    static cv::Ptr<CudaSurfFeatureDetector> create( 
        double hessianThreshold = 3000, 
        int nOctaves =4, 
        int nOctaveLayers =2, 
        bool extended= false, 
        float keyPointsRatio = 0.01, 
        bool upright = false
    );

    void detect(
        cv::InputArray image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::InputArray mask = cv::noArray()
    ) override;

    void compute(
        cv::InputArray image,
        std::vector<cv::KeyPoint>& keypoints,
        cv::OutputArray descriptors
    ) override;

    void detectAndCompute(
        cv::InputArray image,
        cv::InputArray mask,
        std::vector<cv::KeyPoint>& keypoints,
        cv::OutputArray &descriptors,
        bool useProvidedKeypoints = false
    ) override;

};