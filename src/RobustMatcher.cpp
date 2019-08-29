/*
 * RobustMatcher.cpp
 *
 *  Created on: Jun 4, 2014
 *      Author: eriba
 */

#include "RobustMatcher.h"
#include "HahogFeatureDetector.h"
#include <iostream>
#include <time.h>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
using std::cout;
using std::cerr;
using std::endl;
using cv::cuda::GpuMat;
using cv::xfeatures2d::SIFT;
using cv::xfeatures2d::SURF;

namespace
{
    bool checkIfCudaEnabled()
    {
        // ORB is the default feature detector
        auto cudaEnabled = cv::cuda::getCudaEnabledDeviceCount();
        if (cudaEnabled != -1 && cudaEnabled !=0) {
            cerr << "CUDA device detected. Running CUDA \n";
            cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
            return true;
        }
        else
        {
            return false;
        }
    }
} //namespace

RobustMatcher::RobustMatcher(
    const bool cudaEnabled,
    const double ratio,
    cv::Ptr<cv::FeatureDetector> detector,
    cv::Ptr<cv::FeatureDetector> extractor,
    cv::Ptr<cv::DescriptorMatcher> matcher,
    cv::Ptr<cv::cuda::DescriptorMatcher> cMatcher
)
    : cudaEnabled_( cudaEnabled )
    , ratio_( ratio )
    , detector_( detector )
    , extractor_( extractor )
    , matcher_( matcher )
    , cMatcher_( cMatcher )
{
}


RobustMatcher::~RobustMatcher()
{
  // TODO Auto-generated destructor stub
}

namespace
{
    cv::Ptr<RobustMatcher> createOrbMatcher(
        const bool cudaEnabled,
        const int numFeatures,
        const double ratio
    )
    {
        cv::Ptr<cv::DescriptorMatcher> matcher;
        cv::Ptr<cv::cuda::DescriptorMatcher> cMatcher;

        if ( cudaEnabled )
        {
            cMatcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
        }
        else
        {
            constexpr auto crossCheck = false;
            matcher = cv::makePtr<cv::BFMatcher>(cv::NORM_HAMMING, crossCheck);
        }

        return cv::makePtr<RobustMatcher>(
            cudaEnabled,
            ratio,
            cv::ORB::create(numFeatures),
            cv::ORB::create(numFeatures),
            matcher,
            cMatcher
        );
    }

} //namespace

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

    return cv::makePtr<RobustMatcher>(
        cudaEnabled,
        ratio,
        HahogFeatureDetector::create(numFeatures),
        HahogFeatureDetector::create(numFeatures),
        matcher,
        cMatcher
        );
}

cv::Ptr<RobustMatcher> RobustMatcher::createSiftMatcher(const bool cudaEnabled, const int numFeatures, const double ratio)
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

    return cv::makePtr<RobustMatcher>(
        cudaEnabled,
        ratio,
        SIFT::create(numFeatures),
        SIFT::create(numFeatures),
        matcher,
        cMatcher
        );
}

cv::Ptr<RobustMatcher> RobustMatcher::createSurfMatcher(const bool cudaEnabled, const int numFeatures, const double ratio, const int minHessian)
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

    return cv::makePtr<RobustMatcher>(
        cudaEnabled,
        ratio,
        SURF::create(minHessian),
        SURF::create(minHessian),
        matcher,
        cMatcher
        );
}

cv::Ptr<RobustMatcher> RobustMatcher::create(Feature alg, const int numFeatures, const double ratio )
{
    const auto cudaEnabled = checkIfCudaEnabled();

    switch ( alg )
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
    cv::Ptr<cv::cuda::ORB> cudaMatcher;
    if (cudaEnabled_ && (cudaMatcher = matcher_.dynamicCast<cv::cuda::ORB>())) {
        GpuMat cudaFeatureImg;
        cudaFeatureImg.upload(image);
        detector_->detect(cudaFeatureImg, keypoints);
        return;
    }
  detector_->detect(image, keypoints);
}

void RobustMatcher::computeDescriptors(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors)
{
    cv::Ptr<cv::cuda::ORB> cudaMatcher;
    if (cudaEnabled_ && (cudaMatcher = matcher_.dynamicCast<cv::cuda::ORB>())) {
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
    cv::Ptr<cv::cuda::ORB> cudaMatcher;
    if (cudaEnabled_ && (cudaMatcher = matcher_.dynamicCast<cv::cuda::ORB>())) {
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
    {                         // does not have 2 neighbours
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
