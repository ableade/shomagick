#pragma once

#include "flightsession.h"
#include <boost/filesystem.hpp>
#include "RobustMatcher.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

const int FEATURE_PROCESS_SIZE = 2500;

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
    FlightSession flight_;
    void *kd_;
    int dimensions_ = 2;
    int featureSize_;
    std::map<std::string, std::vector<std::string>> candidateImages;
    bool _extractFeature(std::string fileName);
    cv::Ptr<RobustMatcher> rMatcher_;

public:
    ShoMatcher(
        FlightSession flight, 
        int featureSize = FEATURE_PROCESS_SIZE, 
        RobustMatcher::Feature featureType = RobustMatcher::Feature::orb
    );
    void getCandidateMatchesUsingSpatialSearch(double range = 0.000125);
    void getCandidateMatchesFromFile(std::string candidateFile);
    int extractFeatures();
    void runRobustFeatureMatching();
    void buildKdTree();
    std::map<std::string, std::vector<std::string>> getCandidateImages() const;
    void plotMatches(std::string img1, std::string img2) const;
};
