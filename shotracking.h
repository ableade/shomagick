#ifndef SHO_TRACKING_HPP__
#define SHO_TRACKING_HPP__

#include "unionfind.h"
#include "flightsession.h"
#include <map>

class ShoTracker
{
    private:
        FlightSession flight;
        std::map<string, std::vector<string>> candidateImages;
        std::map<string, std::vector<cv::DMatch>> candidateMatches;
        std::map <std::pair <string, int>, int> features;
        UnionFind uf;
        int featuresIndex;

    public:
        ShoTracker(FlightSession flight, std::map<string, std::vector<string>> candidateImages);
        void createTracks();
        std::vector<cv::DMatch> loadMatches(string fileName);
        int addFeatureToIndex(std::pair <string, int> feature);
        void mergeFeatureTracks(std::pair <string, int> feature1, std::pair <string, int> feature2);
        void createFeatureNodes();
};
#endif