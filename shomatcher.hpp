
#ifndef SHOMATCHER_HPP_
#define SHOMATCHER_HPP_

#include "flightsession.h"
#include <utility>
#include "detector.h"
#include <iostream>
#include "kdtree.h"
#include <boost/filesystem.hpp>

using std::cout;
using cv::imread;
using namespace boost::filesystem;

template <class Dptr>
class ShoMatcher {
    private:
        FlightSession flight;
        Detector<Dptr> detector;
        map<string, vector<string>> candidateMatches;

    public:
        ShoMatcher(FlightSession flight, Detector<Dptr> detector) ):flight(flight), detector(detector) {}
        void getCandidateMatches (double range = 0.00010) {
            auto imageSet = this->flight->getCandidateMatches():
            for(int i=0; i< imageSet.size(); ++i) {
                auto count =0;
                vector<string> matchSet;
                auto currentImage = imageSet[i]->fileName;
                void *result_set;

                double pt[] = {mosaicImages[i].location.longitude, mosaicImages[i].location.latitude}; result_set = kd_nearest_range(static_cast<kdtree*>(kd), pt, range);
                double pos[dimensionality];        
                while(!kd_res_end(static_cast<kdres*>(result_set))) {
                    auto current = kd_res_item(static_cast<kdres*>(result_set), pos);
                    if (current == nullptr)
                        continue;
                    count++;
                    auto img = static_cast<Img*>(current); 
                    double dist = sqrt( dist_sq( pt, pos, dimensionality ) );
                    matchSet.push_back(img->fileName)
                    kd_res_next(static_cast<kdres*>(result_set));
                }
                cout << "Found "<<count << "candidate matches for "<<currentImage<<endl;
            }
        }

        void getCandidateKeypointsAndFeatures() {
            for (auto it = this->candidateMatches.begin(); it!= this->candidateMatches.end(); it++) {
                vector<KeyPoint> keypoints;
                Mat descriptors;
                //Have we already created this image features?
                for (auto matchesIt = it->second.begin(); matchesIt!= it->second.end(); ++ matchesIt) {
                    //get keypoints
                }
            }
        }

        bool generateImageFeaturesFile(string imageName) {
            auto imageFeaturePath = this->flight->getImageFeaturesPath() / imageName;
            if (! boost::filesystem::exists(imageFeaturePath)) {
                        Mat img = imread();
                        if (!img.empty()) {
                            myDetector(img, Mat(), keypoints, descriptors);
                            cv::FileStorage file(imageFeatureFile, cv::FileStorage::WRITE);
                            file << imageName << descriptors;
                        }
                        
                        }
        }
        
};
#endif
