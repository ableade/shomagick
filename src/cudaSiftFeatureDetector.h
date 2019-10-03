#pragma once
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include "cudaSift.h"

const int SIFT_DATA_RESERVE_SIZE = 25000;
const float SIFT_PEAK_THRESHOLD = 0.1;
const int SIFT_EDGE_THRESHOLD = 10;
const int SIFT_BIN_SIZE = 128;

class CudaSiftFeatureDetector : public cv::Feature2D {

private:
    void _downloadKeypoints(const SiftData& sd, std::vector<cv::KeyPoint>& keypoints);
    SiftPoint* _getSiftPointItr(const SiftData & sd);
    void _downloadDescriptors(const SiftData& sd, cv::Mat& descriptors);

public:
    static cv::Ptr<CudaSiftFeatureDetector> create();
    void detect(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray()) override;
    void detect(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat &mask);
    void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) override;
    void detectAndCompute(
        cv::InputArray image,
        cv::InputArray mask,
        std::vector<cv::KeyPoint>& keypoints,
        cv::OutputArray &descriptors,
        bool useProvidedKeypoints = false
    ) override;
    void detectAndCompute(cv::Mat image, cv::Mat& mask, std::vector<cv::KeyPoint>& keypoints, cv::Mat &descriptors);
};