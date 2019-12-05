#pragma once

#include "flightsession.h"
#include <boost/filesystem.hpp>
#include "RobustMatcher.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "point.h"
#include "kdtree.h"

const int FEATURE_PROCESS_SIZE = 2500;
const int KD_TREE_DIMENSION_SIZE = 2;

using KDTreeDataSet = std::vector<std::pair<Point<KD_TREE_DIMENSION_SIZE>, std::string>>;

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
    int featureSize_;
    std::map<std::string, std::vector<std::string> > candidateImages;
    bool _extractFeature(std::string fileName);
    cv::Ptr<RobustMatcher> rMatcher_;
    KDTree<KD_TREE_DIMENSION_SIZE, std::string> kd_;
    KDTreeDataSet kdTreeData_;
    void buildKdTree_();

public:
    ShoMatcher(
        FlightSession flight, 
        int featureSize = FEATURE_PROCESS_SIZE, 
        RobustMatcher::Feature featureType = RobustMatcher::Feature::orb
    );
    void getCandidateMatchesUsingKNNSearch(int n);
    void getCandidateMatchesFromFile(std::string candidateFile);
    int extractFeatures();
    void runRobustFeatureMatching();
    std::map<std::string, std::vector<std::string>> getCandidateImages() const;
    void plotMatches(std::string img1, std::string img2) const;
};
