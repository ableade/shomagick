
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
    

  public:
        ShoMatcher(FlightSession flight):flight(flight) {};
        void getCandidateMatches(double range = 0.00010);
        void runRobustFeatureDetection();
        bool generateImageFeaturesFile(string imageName);
        bool saveMatches(string fileName, std::vector<cv::DMatch> matches);
        void buildKdTree();
};
#endif
