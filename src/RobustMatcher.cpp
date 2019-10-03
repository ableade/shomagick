/*
 * RobustMatcher.cpp
 * Adebodun Adekunle
 * Based on robust matcher code from open cv tutorial 
 * See https://raw.githubusercontent.com/opencv/opencv/master/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/src/RobustMatcher.h
 */

#include "RobustMatcher.h"
#include "HahogFeatureDetector.h"
#include <iostream>
#include <time.h>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include "utilities.h"
#include "cudaSurfFeatureDetector.hpp"
#include "cudaSiftFeatureDetector.h"

using std::cout;
using std::cerr;
using std::endl;
using cv::cuda::GpuMat;
using cv::xfeatures2d::SIFT;
using cv::xfeatures2d::SURF;
using cv::Ptr;
using cv::FeatureDetector;

RobustMatcher::RobustMatcher(
    const bool cudaEnabled,
    Feature alg,
    const float ratio,
    cv::Ptr<cv::FeatureDetector> detector,
    cv::Ptr<cv::FeatureDetector> extractor,
    cv::Ptr<cv::DescriptorMatcher> matcher,
    cv::Ptr<cv::cuda::DescriptorMatcher> cMatcher
)
    : cudaEnabled_(cudaEnabled)
    , feature_(alg)
    , ratio_(ratio)
    , detector_(detector)
    , extractor_(extractor)
    , matcher_(matcher)
    , cMatcher_(cMatcher)
{
}


RobustMatcher::~RobustMatcher()
{
    // TODO Auto-generated destructor stub
}


cv::Ptr<RobustMatcher> RobustMatcher::createOrbMatcher(
    const bool cudaEnabled,
    const int numFeatures,
    const double ratio
)
{
    Ptr<cv::DescriptorMatcher> matcher;
    Ptr<cv::cuda::DescriptorMatcher> cMatcher;
    Ptr<FeatureDetector> detector;
    Ptr<FeatureDetector> extractor;
    if (cudaEnabled)
    {
        detector = cv::cuda::ORB::create(numFeatures);
        extractor = cv::cuda::ORB::create(numFeatures);
        cMatcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    }
    else
    {
        constexpr auto crossCheck = false;
        matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_HAMMING, crossCheck);
        detector = cv::ORB::create(numFeatures);
        extractor = cv::ORB::create(numFeatures);
    }

    auto pRobustMatcher = new RobustMatcher(
        cudaEnabled,
        Feature::orb,
        ratio,
        detector,
        extractor,
        matcher,
        cMatcher
    );

    return Ptr<RobustMatcher>{pRobustMatcher};

}

cv::Ptr<RobustMatcher> RobustMatcher::createHahogMatcher(const bool cudaEnabled, const int numFeatures, const double ratio)
{
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Ptr<cv::cuda::DescriptorMatcher> cMatcher;

    if (cudaEnabled)
    {
        cMatcher = cv::cuda::DescriptorMatcher::createBFMatcher();
    }
    else
    {
        matcher = cv::makePtr<cv::BFMatcher>();
    }

    auto pRobustMatcher = new RobustMatcher(
        cudaEnabled,
        Feature::hahog,
        ratio,
        HahogFeatureDetector::create(numFeatures),
        HahogFeatureDetector::create(numFeatures),
        matcher,
        cMatcher
    );
    return Ptr<RobustMatcher>{pRobustMatcher};
}

cv::Ptr<RobustMatcher> RobustMatcher::createSiftMatcher(const bool cudaEnabled, const int numFeatures, const double ratio)
{
    Ptr<FeatureDetector> detector;
    Ptr<FeatureDetector> extractor;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Ptr<cv::cuda::DescriptorMatcher> cMatcher;

    if (cudaEnabled)
    {
#ifndef  _MSC_VER
        //We are only able to compile the sift cuda detector and extractor only on linux only
        detector = CudaSiftFeatureDetector::create();
        extractor = CudaSiftFeatureDetector::create();
#endif // ! _MSC_VER
        cMatcher = cv::cuda::DescriptorMatcher::createBFMatcher();  
    }
    
    if(detector==nullptr || extractor == nullptr)
    {
        detector = SIFT::create(numFeatures);
        extractor = SIFT::create(numFeatures);
        matcher = cv::makePtr<cv::BFMatcher>();
    }

    auto pRobustMatcher = new RobustMatcher(
        cudaEnabled,
        Feature::sift,
        ratio,
        detector,
        extractor,
        matcher,
        cMatcher
    );
    return Ptr<RobustMatcher>{pRobustMatcher};
}

cv::Ptr<RobustMatcher> RobustMatcher::createSurfMatcher(const bool cudaEnabled, const int numFeatures, const double ratio, const int minHessian)
{
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Ptr<cv::cuda::DescriptorMatcher> cMatcher;
    Ptr<FeatureDetector> detector;
    Ptr<FeatureDetector> extractor;
    if (cudaEnabled)
    {
        detector = CudaSurfFeatureDetector::create();
        extractor = CudaSurfFeatureDetector::create();
        cMatcher = cv::cuda::DescriptorMatcher::createBFMatcher();
    }
    else
    {
        detector = SURF::create(minHessian);
        extractor = SURF::create(minHessian);
        matcher = cv::makePtr<cv::BFMatcher>();
    }

    auto pRobustMatcher = new RobustMatcher(
        cudaEnabled,
        Feature::surf,
        ratio,
        detector,
        extractor,
        matcher,
        cMatcher
    );
    return Ptr<RobustMatcher>{pRobustMatcher};
}

cv::Ptr<RobustMatcher> RobustMatcher::create(Feature alg, const int numFeatures, const double ratio)
{
    const auto cudaEnabled = checkIfCudaEnabled();
    if (cudaEnabled) {
        cerr << "CUDA device detected. Running CUDA \n";
        cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
    }
    else {
        cerr << "Unable to detect a CUDA device on this system. Will run in normal mode\n";
    }

    switch (alg)
    {
    case Feature::hahog:
        return RobustMatcher::createHahogMatcher(cudaEnabled, numFeatures, ratio);

    case Feature::sift:
        return RobustMatcher::createSiftMatcher(cudaEnabled, numFeatures, ratio);

    case Feature::surf:
        return RobustMatcher::createSurfMatcher(cudaEnabled, numFeatures, ratio);

    default:
        return createOrbMatcher(cudaEnabled, numFeatures, ratio);

    }
}

void RobustMatcher::computeKeyPoints(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints)
{
    if (cudaEnabled_) {
        GpuMat cudaFeatureImg;
        cudaFeatureImg.upload(image);
        detector_->detect(cudaFeatureImg, keypoints);
        return;
    }
    detector_->detect(image, keypoints);
}

void RobustMatcher::computeDescriptors(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    if (cudaEnabled_) {
        GpuMat cudaFeatureImg, cudaDescriptors;
        cudaFeatureImg.upload(image);
        extractor_->compute(cudaFeatureImg, keypoints, cudaDescriptors);
        cudaDescriptors.download(descriptors);
        return;
    }
    extractor_->compute(image, keypoints, descriptors);
}

void RobustMatcher::detectAndCompute(const cv::Mat & image, std::vector<cv::KeyPoint>& keypoints, cv::Mat & descriptors)
{
    //Only use GPU mat if the detector is not sift or hahog
    if (cudaEnabled_ && !(feature_ == Feature::sift || feature_ == Feature::hahog)) {
        GpuMat cudaFeatureImg, cudaDescriptors;
        cudaFeatureImg.upload(image);
        detector_->detectAndCompute(cudaFeatureImg, cv::noArray(), keypoints, cudaDescriptors);
        cudaDescriptors.download(descriptors);
        return;
    }
    detector_->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
}

int RobustMatcher::ratioTest(std::vector<std::vector<cv::DMatch>> &matches)
{
    int removed = 0;
    // for all matches
    for (std::vector<std::vector<cv::DMatch>>::iterator
        matchIterator = matches.begin();
        matchIterator != matches.end(); ++matchIterator)
    {
        // if 2 NN has been identified
        if (matchIterator->size() > 1)
        {
            // check distance ratio
            if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > ratio_)
            {
                matchIterator->clear(); // remove match
                removed++;
            }
        }
        else
        {                        
            matchIterator->clear(); // remove match
            removed++;
        }
    }
    return removed;
}

void RobustMatcher::symmetryTest(
    const std::vector<std::vector<cv::DMatch>> &matches1,
    const std::vector<std::vector<cv::DMatch>> &matches2,
    std::vector<cv::DMatch> &symMatches
)
{

    // for all matches image 1 -> image 2
    for (std::vector<std::vector<cv::DMatch>>::const_iterator
        matchIterator1 = matches1.begin();
        matchIterator1 != matches1.end(); ++matchIterator1)
    {

        // ignore deleted matches
        if (matchIterator1->empty() || matchIterator1->size() < 2)
            continue;

        // for all matches image 2 -> image 1
        for (std::vector<std::vector<cv::DMatch>>::const_iterator
            matchIterator2 = matches2.begin();
            matchIterator2 != matches2.end(); ++matchIterator2)
        {
            // ignore deleted matches
            if (matchIterator2->empty() || matchIterator2->size() < 2)
                continue;

            // Match symmetry test
            if ((*matchIterator1)[0].queryIdx ==
                (*matchIterator2)[0].trainIdx &&
                (*matchIterator2)[0].queryIdx ==
                (*matchIterator1)[0].trainIdx)
            {
                // add symmetrical match
                symMatches.push_back(
                    cv::DMatch((*matchIterator1)[0].queryIdx,
                    (*matchIterator1)[0].trainIdx,
                        (*matchIterator1)[0].distance));
                break; // next match in image 1 -> image 2
            }
        }
    }
}

void RobustMatcher::robustMatch(const cv::Mat &image1, const cv::Mat &trainImage, std::vector<cv::DMatch> &matches,
    std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2)
{
    cv::Mat descriptors1;
    cv::Mat descriptors2;


    detectAndCompute(image1, keypoints1, descriptors1);
    detectAndCompute(trainImage, keypoints2, descriptors2);

    robustMatch(descriptors1, descriptors2, matches);
}

void RobustMatcher::robustMatch(
    const cv::Mat descriptors1,
    const cv::Mat descriptors2,
    std::vector<cv::DMatch>& matches
)
{
    std::vector<std::vector<cv::DMatch>> matches12, matches21;

    // Symmetric matching using two nearest neighbours. Use CUDA if available
    if (cudaEnabled_) {
        cv::cuda::GpuMat gDescriptors1, gDescriptors2;
        gDescriptors1.upload(descriptors1);
        gDescriptors2.upload(descriptors2);

        // Image 1 to 2 matching
        cMatcher_->knnMatch(gDescriptors1, gDescriptors2, matches12, 2); // return 2 nearest neighbours 

        // Image 2 to 1 matching
        cMatcher_->knnMatch(gDescriptors2, gDescriptors1, matches21, 2); // return 2 nearest neighbours
    }
    else {
        // From image 1 to image 2
        matcher_->knnMatch(descriptors1, descriptors2, matches12, 2);

        // From image 2 to image 1
        matcher_->knnMatch(descriptors2, descriptors1, matches21, 2);
    }


    // 3. Remove matches for which NN ratio is > than threshold
    // clean image 1 -> image 2 matches
    ratioTest(matches12);
    // clean image 2 -> image 1 matches
    ratioTest(matches21);

    // 4. Remove non-symmetrical matches
    symmetryTest(matches12, matches21, matches);
}

void RobustMatcher::fastRobustMatch(
    const cv::Mat queryImg,
    std::vector<cv::DMatch>& goodMatches,
    std::vector<cv::KeyPoint>& queryKeypoints,
    const cv::Mat & trainImg,
    std::vector<cv::KeyPoint>& trainKeypoints
)
{
    cv::Mat descriptors1;
    cv::Mat descriptors2;
    detectAndCompute(queryImg, queryKeypoints, descriptors1);
    detectAndCompute(trainImg, trainKeypoints, descriptors2);

    robustMatch(descriptors1, descriptors2, goodMatches);
}

void RobustMatcher::fastRobustMatch(const cv::Mat descriptors1, const cv::Mat descriptors2, std::vector<cv::DMatch> &matches)
{
    matches.clear();
    std::vector<std::vector<cv::DMatch>> matches12;

    // 2. Match the two image descriptors
    std::vector<std::vector<cv::DMatch>> kmatches;
    matcher_->knnMatch(descriptors1, descriptors2, kmatches, 2);

    // 3. Remove matches for which NN ratio is > than threshold
    ratioTest(kmatches);

    // 4. Fill good matches container
    for (std::vector<std::vector<cv::DMatch>>::iterator
        matchIterator = kmatches.begin();
        matchIterator != kmatches.end(); ++matchIterator)
    {
        if (!matchIterator->empty())
            matches.push_back((*matchIterator)[0]);
    }
}