#pragma once
/*
 * cudamatcher.hpp
 */


#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include "RobustMatcher.h"

class CUDARobustMatcher
{
public:
    CUDARobustMatcher(int numFeatures = 8000) : detector_(cv::cuda::ORB::create(numFeatures)), extractor_(cv::ORB::create(numFeatures)), ratio_(0.8f)
    {
        // ORB is the default feature
        // BruteFroce matcher with Norm Hamming is the default matcher
        matcher_ = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    }

    virtual ~CUDARobustMatcher();

    // Set the feature detector
    void setFeatureDetector(const cv::Ptr<cv::FeatureDetector> &detect) { detector_ = detect; }

    // Set the descriptor extractor
    void setDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor> &desc) { extractor_ = desc; }

    // Compute the keypoints of an image
    void computeKeyPoints(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints);

    // Compute the descriptors of an image given its keypoints
    void computeDescriptors(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);

    // Set ratio parameter for the ratio test
    void setRatio(float rat) { ratio_ = rat; }

    // Clear matches for which NN ratio is > than threshold
    // return the number of removed points
    // (corresponding entries being cleared,
    // i.e. size will be 0)
    int ratioTest(std::vector<std::vector<cv::DMatch>> &matches);

    // Insert symmetrical matches in symMatches vector
    void symmetryTest(const std::vector<std::vector<cv::DMatch>> &matches1,
        const std::vector<std::vector<cv::DMatch>> &matches2,
        std::vector<cv::DMatch> &symMatches);

    // Match feature points using ratio and symmetry test
    void robustMatch(const cv::Mat &image1, const cv::Mat &trainImage, std::vector<cv::DMatch> &matches,
        std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2);

    //Match feature point using  ratio and symmetry test
    void robustMatch(const cv::Mat descriptors1, const cv::Mat descriptors2, std::vector<cv::DMatch> &matches);
    // Match feature points using ratio test
    void fastRobustMatch(const cv::Mat &frame, std::vector<cv::DMatch> &good_matches,
        std::vector<cv::KeyPoint> &keypoints_frame,
        const cv::Mat &descriptors_model);

    void fastRobustMatch(const cv::Mat descriptors1, const cv::Mat descriptors2, std::vector<cv::DMatch> &matches);

private:
    // pointer to the feature point detector object
    cv::Ptr<cv::FeatureDetector> detector_;
    // pointer to the feature descriptor extractor object
    cv::Ptr<cv::DescriptorExtractor> extractor_;
    // pointer to the matcher object
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher_;
    // max ratio between 1st and 2nd NN
    float ratio_;
};
