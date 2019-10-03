#pragma once
/*
 * RobustMatcher.h
 *
 *  Created on: Jun 4, 2014
 *      Author: eriba
 */
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/cudafeatures2d.hpp>


class RobustMatcher
{
public:
    enum class Feature { orb, hahog, sift, surf };
    virtual ~RobustMatcher();
    // creates a robust matcher with chosen feature detection algorithm
    static cv::Ptr<RobustMatcher> create(
        Feature alg, 
        int numFeatures = 8000, 
        double ratio = 0.8
    );
    // Set the feature detector
    void setFeatureDetector(const cv::Ptr<cv::FeatureDetector> &detect) { detector_ = detect; }
    // Set the descriptor extractor
    void setDescriptorExtractor(const cv::Ptr<cv::DescriptorExtractor> &desc) { extractor_ = desc; }
    // Set the matcher
    void setDescriptorMatcher(const cv::Ptr<cv::DescriptorMatcher> &match) { matcher_ = match; }
    // Compute the keypoints of an image
    void computeKeyPoints(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints);
    // Compute the descriptors of an image given its keypoints
    void computeDescriptors(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);
    // Compute the descriptors and keypoint for an image
    void detectAndCompute(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors);
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
    void fastRobustMatch(const cv::Mat queryImg, std::vector<cv::DMatch> &good_matches,
        std::vector<cv::KeyPoint> &queryKeypoints,
        const cv::Mat &trainImg, std::vector<cv::KeyPoint> & trainKeypoints);
    void fastRobustMatch(
        const cv::Mat descriptors1,
        const cv::Mat descriptors2,
        std::vector<cv::DMatch> &matches
    );

private:
    bool cudaEnabled_ = false;
    Feature feature_;
    float ratio_;
    // pointer to the feature point detector object
    cv::Ptr<cv::FeatureDetector> detector_;
    // pointer to the feature descriptor extractor object
    cv::Ptr<cv::DescriptorExtractor> extractor_;
    // pointer to the matcher object
    cv::Ptr<cv::DescriptorMatcher> matcher_;
    cv::Ptr<cv::cuda::DescriptorMatcher> cMatcher_;
    // max ratio between 1st and 2nd NN

    RobustMatcher(
        bool cudaEnabled,
        Feature alg,
        float ratio,
        cv::Ptr<cv::FeatureDetector> detector,
        cv::Ptr<cv::FeatureDetector> extractor,
        cv::Ptr<cv::DescriptorMatcher> matcher,
        cv::Ptr<cv::cuda::DescriptorMatcher> cMatcher
    );

    static cv::Ptr<RobustMatcher> createOrbMatcher(
        const bool cudaEnabled,
        const int numFeatures,
        const double ratio
    );
    static cv::Ptr<RobustMatcher> createHahogMatcher(
        const bool cudaEnabled,
        const int numFeatures,
        const double ratio
    );
    static cv::Ptr<RobustMatcher> createSiftMatcher(
        const bool cudaEnabled,
        const int numFeatures,
        const double ratio
    );
    static cv::Ptr<RobustMatcher> createSurfMatcher(
        const bool cudaEnabled,
        const int numFeatures,
        const double ratio,
        const int minHessian = 1000
    );
};