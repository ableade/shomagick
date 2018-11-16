#include "shotracking.h"
#include "string"
#include <vector>

using std::pair;
using std::vector;
using cv::DMatch;
using std::map;
using std::make_pair;


ShoTracker::ShoTracker(FlightSession flight, std::map<string, std::vector<string>> candidateImages):flight(flight),candidateImages(candidateImages){
    this->featuresIndex = 0;
}


vector <cv::DMatch> ShoTracker:: loadMatches(string fileName) {
    auto imageMatchesPath = this->flight.getImageMatchesPath() / (fileName + ".xml");
    cv::FileStorage fs(imageMatchesPath.string(), cv::FileStorage::READ);
    vector<DMatch> matches;
    fs["matches"] >> matches;
    return matches;
}

void ShoTracker::createFeatureNodes() {
    //Image name and corresponding keypoint index form a single node
    for(auto it = candidateMatches.begin(); it!= candidateMatches.end(); ++it) {
        for (auto matchIt = it->second.begin(); matchIt!= it->second.end(); ++it) {
            auto leftFeature = make_pair(it->first, matchIt->trainIdx);
            auto matchImage = this->flight.getImageSet()[matchIt->imgIdx].fileName;
            auto rightFeature = make_pair(matchImage, matchIt->queryIdx);
            this->addFeatureToIndex(leftFeature);
            this->addFeatureToIndex(rightFeature);
        }
    }
    this->uf = UnionFind(this->features.size());
}

void ShoTracker::createTracks() {
    for(auto it = candidateMatches.begin(); it!= candidateMatches.end(); ++it) {
        for (auto matchIt = it->second.begin(); matchIt!= it->second.end(); ++it) {
            auto leftFeature = make_pair(it->first, matchIt->trainIdx);
            auto matchImage = this->flight.getImageSet()[matchIt->imgIdx].fileName;
            auto rightFeature = make_pair(matchImage, matchIt->queryIdx);
            this->mergeFeatureTracks(leftFeature, rightFeature);
        }
    }
}

void ShoTracker:: mergeFeatureTracks(pair <string, int> feature1, pair<string, int> feature2) {
    this->uf.unionSet(this->features[feature1], this->features[feature2]);
}

int ShoTracker::addFeatureToIndex(pair <string, int> feature) {
    auto insert = this->features.insert(make_pair(feature, this->featuresIndex));
    if (insert.second) {
     this->featuresIndex++;
 } else {
    return insert.first->second;
}
return this->featuresIndex -1;
}
