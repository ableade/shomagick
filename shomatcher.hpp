
#ifndef SHOMATCHER_HPP_
#define SHOMATCHER_HPP_

#include "flightsession.h"
#include "detector.h"
#include <boost/filesystem.hpp>

inline double dist_sq(double *a1, double *a2, int dims)
{
  double dist_sq = 0, diff;
  while (--dims >= 0)
  {
    diff = (a1[dims] - a2[dims]);
    dist_sq += diff * diff;
  }
  return dist_sq;
}

class ShoMatcher
{
  private:
    FlightSession flight;
    Detector <SURF> surfDetector;
    Detector <SIFT> siftDetector;
    std::map<string, std::vector<string>> candidateImages;
    void *kd;
    int dimensions = 2;
    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> extractor_;
    cv::Ptr<cv::DescriptorMatcher> matcher_;

    bool _extractFeature(string fileName);
    

  public:
        ShoMatcher(FlightSession flight):flight(flight) {};
        void getCandidateMatches(double range = 0.000125);
        int extractFeatures();
        void runRobustFeatureDetection();
        void buildKdTree();
        std::map<string, std::vector<string>> getCandidateImages() const;
        void setFeatureDetector(const cv::Ptr<cv::FeatureDetector>& detector);
        void setFeatureExtractor(const cv::Ptr<cv::DescriptorExtractor>& extractor);
        void setMatcher(const cv::Ptr<cv::DescriptorMatcher>& matcher);
};
#endif
