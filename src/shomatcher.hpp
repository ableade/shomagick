#pragma once

#include "flightsession.h"
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

const int FEATURE_PROCESS_SIZE = 700;

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
    bool runCuda = true;
    void *kd;
    int dimensions = 2;
    int featureSize = 5000;
    bool cudaEnabled = false;
    std::map<std::string, std::vector<std::string>> candidateImages;
    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> extractor_;
    bool _extractFeature(std::string fileName, bool resize = false);

public:
    ShoMatcher(FlightSession flight, bool runCuda = true);
    void getCandidateMatchesUsingSpatialSearch(double range = 0.000125);
    void getCandidateMatchesFromFile(std::string candidateFile);
    int extractFeatures(bool resize = false);
    void runRobustFeatureMatching();
    void buildKdTree();
    std::map<std::string, std::vector<std::string>> getCandidateImages() const;
    void plotMatches(std::string img1, std::string img2) const;
};
