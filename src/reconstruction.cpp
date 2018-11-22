#include "reconstruction.h"
#include <vector>

using std::vector;
using cv::Mat;
using cv::Mat_;

Reconstructor :: Reconstructor (FlightSession flight, TrackGraph tg): flight (flight), tg(tg)  {};

void Reconstructor::computeEssentialMatrix(string image1,  string image2,  Camera camera, int method, double tresh, double prob) {
    Mat cameraMatrix = (Mat_<double>(3,3) << 3.8123526712521689e+3,0.2592, 0 ,3.8123526712521689e+03,1944,0, 0.1);
    auto allPairMatches = this->flight.loadMatches(image1);
    auto pairwiseMatches = allPairMatches[image2];
    auto features2 = this->flight.loadFeatures(image2);
    auto features1 = this->flight.loadFeatures(image1);

    vector <cv::Point2f> impoints1, impoints2;
    //align pairwise matches
    for(auto dmatch : pairwiseMatches) {
        auto point1 = features1.first[dmatch.queryIdx].pt;
        auto point2 = features2.first[dmatch.trainIdx].pt;
        impoints1.push_back(point1);
        impoints2.push_back(point2);
    }
    Mat mask;
    Mat essentialMatrix = cv::findEssentialMat(impoints1, impoints2, cameraMatrix, method, tresh, prob, mask);
}
