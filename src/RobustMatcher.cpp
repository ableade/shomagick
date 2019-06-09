/*
 * RobustMatcher.cpp
 *
 *  Created on: Jun 4, 2014
 *      Author: eriba
 */

#include "RobustMatcher.h"
#include <iostream>
#include <time.h>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>

using std::cout;
using std::cerr;
using std::endl;
using cv::cuda::GpuMat;

RobustMatcher::RobustMatcher(int numFeatures, double ratio) : detector_(cv::ORB::create(numFeatures))
, extractor_(cv::ORB::create(numFeatures)), ratio_(ratio)
{

    // ORB is the default feature detector
    if (cv::cuda::getCudaEnabledDeviceCount()) {
        cerr << "CUDA device detected. Running CUDA \n";
        cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
        cudaEnabled_ = true;
    }
    if (cudaEnabled_) {
        detector_ = cv::cuda::ORB::create(numFeatures);
        extractor_ = cv::cuda::ORB::create(numFeatures);
        
        cMatcher_ = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    } else
    {
        detector_ = cv::ORB::create(numFeatures);
        extractor_ = cv::ORB::create(numFeatures);
        // BruteFroce matcher with Norm Hamming is the default matcher
        matcher_ = cv::makePtr<cv::BFMatcher>((int)cv::NORM_HAMMING, false);
    }
}

RobustMatcher::~RobustMatcher()
{
  // TODO Auto-generated destructor stub
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
    if (cudaEnabled_) {
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
