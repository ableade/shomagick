#ifndef SHOMATCHER_HPP_
#define SHOMATCHER_HPP_

#include "flightsession.h"
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

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
    void *kd;
    int dimensions = 2;
    int featureSize = 5000;
    std::map<string, std::vector<string>> candidateImages;
    cv::Ptr<cv::FeatureDetector> detector_;
    cv::Ptr<cv::DescriptorExtractor> extractor_;

    bool _extractFeature(string fileName);

public:
    ShoMatcher(FlightSession flight)
        : flight(flight)
        , kd(nullptr)
        , candidateImages()
        , detector_(cv::ORB::create(4000))
        , extractor_(cv::ORB::create(4000))
    {}
    void getCandidateMatches(double range = 0.000125);
    int extractFeatures();
    void runRobustFeatureMatching();
    void buildKdTree();
    std::map<string, std::vector<string>> getCandidateImages() const;
};
#endif
